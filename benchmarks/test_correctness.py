import dataclasses
import os
from typing import Any, List
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

PROMPT = [
    "The French Revolution (French: Révolution française) was a period of political and societal change in France which began with the Estates General of 1789 and ended with the Coup of 18 Brumaire on 9 November 1799. Many of the revolution's ideas are considered",
    "fundamental principles of liberal democracy,[1] and its values remain central to modern French political discourse.[2] The causes of the revolution were a combination of social, political, and economic factors which the ancien régime ('old regime') proved unable to manage. A financial ",
    "crisis and widespread social distress led to the convocation of the Estates General in May 1789, its first meeting since 1614. The representatives of the Third Estate broke away and re-constituted themselves as a National Assembly in June. The Storming of the Bastille in Paris on 14 July",
    " was followed by radical measures by the Assembly, among them the abolition of feudalism, state control over the Catholic Church, and a declaration of rights. The next three years were dominated by a struggle for political control. King Louis XVI's attempted flight to Varennes in June 1791",
    " further discredited the monarchy, and military defeats after the outbreak of the French Revolutionary Wars in April 1792 led to an armed insurrection on 10 August 1792. The monarchy was replaced by the French First Republic in September, and Louis XVI was executed in January 1793.",
    "After another revolt in June 1793, the constitution was suspended, and political power passed from the National Convention to the Committee of Public Safety, dominated by radical Jacobins led by Maximilien Robespierre. About 16,000 people were sentenced by the Revolutionary Tribunal ",
    "and executed in the Reign of Terror, which ended in July 1794 with the Thermidorian Reaction. Weakened by external threats and internal opposition, the Committee of Public Safety was replaced in November 1795 by the Directory. Its instability ended in the coup of 18 Brumaire and the ",
    "establishment of the Consulate, with Napoleon Bonaparte as First Consul. The Revolution resulted from multiple long-term and short-term factors, culminating in a social, economic, financial and political crisis in the late 1780s.[3][4][5] Combined with resistance to reform by the ruling",
    " elite and indecisive policy by Louis XVI and his ministers, the result was a crisis the state was unable to manage.[6][7] Between 1715 and 1789, the French population grew from 21 to 28 million, 20% of whom lived in towns or cities, Paris alone having over 600,000 inhabitants.[8] This ",
    "was accompanied by a tripling in the size of the middle class, which comprised almost 10% of the population by 1789.[9] Despite increases in overall prosperity, its benefits were largely restricted to the rentier and mercantile classes, while the living standards fell for wage labourers ",
    "and peasant farmers who rented their land.[10][11] Economic recession from 1785, combined with bad harvests in 1787 and 1788, led to high unemployment and food prices, causing a financial and political crisis.[3][12][13][14] While the state also experienced a debt crisis, the level of debt",
    " itself was not high compared with Britain's.[15] A significant problem was that tax rates varied widely from one region to another, were often different from the official amounts, and were collected inconsistently. Its complexity meant uncertainty over the amount contributed by any authorised",
    " tax caused resentment among all taxpayers.[16][a] Attempts to simplify the system were blocked by the regional Parlements which approved financial policy. The resulting impasse led to the calling of the Estates General of 1789, which became radicalised by the struggle for control of public ",
    "finances.[18] Louis XVI was willing to consider reforms, but he often backed down when faced with opposition from conservative elements within the nobility. Enlightenment critiques of social institutions were widely discussed among the educated French elite. At the same time, the American ",
    "Revolution and the European revolts of the 1780s inspired public debate on issues such as patriotism, liberty, equality, and democracy. These shaped the response of the educated public to the crisis,[19] while scandals such as the Affair of the Diamond Necklace fuelled widespread anger at ",
    "France faced a series of budgetary crises during the 18th century as revenues failed to keep pace with expenditure.[21][22] Although the economy grew solidly, the increase was not reflected in a proportional growth in taxes,[21] their collection being contracted to tax farmers who kept much"
    " of it as personal profit. As the nobility and Church benefited from many exemptions, the tax burden fell mainly on peasants.[23] Reform was difficult because new tax laws had to be registered with regional judicial bodies or parlements that were able to block them. The king could impose laws"
    " by decree, but this risked open conflict with the parlements, the nobility, and those subject to new taxes.[24] France primarily funded the Anglo-French War of 1778–1783 through loans. Following the peace, the monarchy borrowed heavily, culminating in a debt crisis. By 1788, half of state "
    "revenue was required to service its debt.[25] In 1786, the French finance minister, Calonne, proposed a package of reforms including a universal land tax, the abolition of grain controls and internal tariffs, and new provincial assemblies appointed by the king. The new taxes were rejected, "
    "first by a hand-picked Assembly of Notables dominated by the nobility, then by the parlements when submitted by Calonne's successor Brienne. The notables and parlements argued that the proposed taxes could only be approved by an Estates-General, a representative body that had last met in 1614.[26]"
    "The conflict between the Crown and the parlements became a national political crisis. Both sides issued a series of public statements, the government arguing that it was combating privilege and the parlement defending the ancient rights of the nation. Public opinion was firmly on the side of the "
    "parlements, and riots broke out in several towns. Brienne's attempts to raise new loans failed, and on 8 August 1788, he announced that the king would summon an Estates-General to convene the following May. Brienne resigned and was replaced by Jacques Necker.[27]"
]


def trim_down_prompt(prompt: str, length: int, tokenizer: Any) -> str:
    sequence_length = len(tokenizer.encode(prompt))
    while sequence_length > length:
        prompt = prompt[:-1]
        sequence_length = len(tokenizer.encode(prompt))
    # if sequence_length < length:
    #     raise ValueError(f"Prompt is too short: {sequence_length} < {length}")
    return prompt, sequence_length

def run_test(llm: LLM, input_length: int, output_length: int, batch_size: int, results_dir: str) -> None:
    # Setup test
    prompt, sequence_length = trim_down_prompt("".join(PROMPT), input_length, llm.get_tokenizer())
    results_path = os.path.join(results_dir, f"test_il_{input_length}_bs_{batch_size}.txt")
    with open(results_path, "w") as file:
        # Preface results
        file.write(f"Testing w/ input length {input_length} and batch size {batch_size}...\n")
        file.write(f"Sequence length: {sequence_length}\n")
        file.write(f"Prompt: {prompt!r}\n")
        # Generate and save results
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=output_length)
        outputs = llm.generate([prompt + "" for _ in range(batch_size)], sampling_params)
        for i, output in enumerate(outputs):
            file.write(f"{i}: {output.outputs[0].text!r}\n")

if __name__ == "__main__":
    # Prepare parser
    parser = FlexibleArgumentParser()
    parser.add_argument("--input-len", nargs="+", type=int, default=[128, 2048])
    parser.add_argument("--output-len",  type=int, default=20)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--results-dir", type=str, default="./test_results")

    # Parse arguments
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    output_len = int(args.output_len)

    # Initialize LLM
    llm = LLM(**dataclasses.asdict(EngineArgs.from_cli_args(args)))
    os.makedirs(args.results_dir, exist_ok=True)

    # Check correctness
    for input_length in args.input_len:
        for batch_size in args.batch_sizes:
            run_test(llm, input_length, output_len, batch_size, args.results_dir)
