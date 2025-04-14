# HFRK ReadMe

This is a document describing how to install the Hugging Face ROCm Kernels package, patch VLLM in order to use the kernels introduced in HFRK, and benchmark and test the patched model. 


## Scope

Currently, the model patched is the Llama model in FP8, and kernels have been tuned for the Llama 405B model. The kernels in HFRK can be introduced into any other FP8 model, and modified to output FP16. It is already the case for the RMS norm fused kernel.

## Setup

### 1. Install HFRK

```
git clone https://github.com/huggingface/hf-rocm-kernels  
cd hf-rocm-kernels
./script/build.sh gfx942
```

### 2. Decide which kernels to use
In the file `vllm/model_executor/models/llama.py`, after the first import section, there is a section called `HFRK Switch board`. It controls which kernels are turned on, and thus is quite usefull to benchmark end-to-end impact of any custom kernel. By default, all boolean flags are set to True, meaning all kcustom kernels are used. The only non-trivially named flag is `SKINNY_LIMIT`, which is the number of rows under which a 2D matrix is considered "skinny" and thus skinny GEMMs are used. We set this to 16, because the performance of skinny GEMM on 32 rows matrices is not yet better than torch's GEMM (which binds hipBlasLT). 

### 3. Patch VLLM
```
cp ./vllm/model_executor/models/llama.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/llama.py
cp ./vllm/model_executor/layers/quantization/fp8.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py
cp ./benchmarks/benchmark_latency.py /app/vllm/benchmarks/benchmark_latency.py
```
This does the following:
- patch the `llama.py` file so that the custom kernels are used during inference;
- patch the `fp8.py` file so that after weight loading, if a layer has both a `input_scale` and `weight_scale` attibute, we add the `combined_scale` attribute which is a product of the first two. It is helpfull for the skinny GEMM kernel;
- patch the `benchmark_latency.py` file so that instead of just running the model for one input lenght, output length and batch size, it can do this for multiple combination of the three and also run an additionnal profiling iteration for each set of batch size, input length and output length. We recommend using this script of the non-patched one because it does not require re-loading of the model each time we want to change the batch size or ask for a profile.

## Running benchmark
Once setup is done, we can run benchmarks using the patched script. We first define some parameters to match https://github.com/ROCm/MAD/blob/develop/scripts/vllm/vllm_benchmark_report.sh:
```
export VLLM_USE_TRITON_FLASH_ATTN=0
export NCCL_MIN_NCHANNELS=112
export VLLM_FP8_PADDING=1

model="/data/hub/models--amd--Llama-3.1-405B-Instruct-FP8-KV/snapshots/2505537398e7cfda52f6d666f315c03db8e4697c/"
```
You can change the `model` variable depending on where you store the model's `safetensor` files.

Then, you can run benchmarks for prefill configuration using:

```
cd /app/vllm/benchmarks
python benchmark_latency.py \
    --batch-sizes 1 2 4 8 16 32 64 128 256 --input-len 128 2048 --output-len 1 --tensor-parallel-size 8 --enforce-eager \
    --dtype "float16" --model $model --results-dir ./results/with_custom_kernels --profile
```
We recommend stopping the benchmark after batch size 64 and input length 2048 because there are only two configurations left, batch size 128 and 256 for input length 2048, and they happen to serve requests by batches of 64, so we can extrapolate easily from batch size 64 and input length 2048.  
To run decoding benchmark, we use:

```
cd /app/vllm/benchmarks
python benchmark_latency.py \
    --batch-sizes 1 2 4 8 16 32 64 128 256 --input-len 1 --output-len 128 --tensor-parallel-size 8 \
    --dtype "float16" --model $model --results-dir ./results/with_custom_kernels --profile
```

If we want to test out a different custom kernels combination, we can just modify the `HFRK Switch Board` section of `llama.py` and re-run `cp ./vllm/model_executor/models/llama.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/llama.py` .

## Running tests
We also added a test script meant to test the model's correctness for all batch size and input length combinations. To run it, use:

```
cp ./benchmarks/test_correctness.py /app/vllm/benchmarks/test_correctness.py
cd /app/vllm/benchmarks
python test_correctness.py \
    --tensor-parallel-size 8 --dtype "float16" --model $model --results-dir ~/vllm/test_results
```
You can add the `--enforce-eager` flag to see if preifll works as intended. The results are dumped into the `--result-dir` and you may change the output length using `--output-length`.
