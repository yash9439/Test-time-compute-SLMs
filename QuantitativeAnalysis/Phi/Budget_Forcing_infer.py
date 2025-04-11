from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import random
import csv

SYSTEM_PROMPT = """You are a helpful AI assistant. Analyze the user's question carefully and provide a step-by-step derivation. Respond strictly in the following format, ensuring both tags are present and complete:
<reasoning>
Provide detailed step-by-step thinking here, explaining the logic used to reach the solution.
</reasoning>
<answer>
Provide ONLY the final answer here. If the question requires a specific numerical value, provide only that number (as an integer if possible). If the question requires a formula, expression, or general method, provide that concisely. Do not include units, surrounding text, explanations, or any other content within this tag.
</answer>"""

TEMPERATURE = 0.8
MAX_TOKENS_THINKING = 4096
NUM_IGNORE = 3  # only do zero‑shot + 3 Wait passes

MODEL_NAME = "microsoft/Phi-4-mini-instruct"

model = LLM(
    MODEL_NAME,
    tensor_parallel_size=1,
    max_model_len=16384
)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

sampling_params_softEnd = SamplingParams(
    max_tokens=MAX_TOKENS_THINKING,
    min_tokens=0,
    stop_token_ids=tok("<|im_end|>")["input_ids"],
    skip_special_tokens=False,
    temperature=TEMPERATURE,
    frequency_penalty=0.1,
)
sampling_params_hardEnd = SamplingParams(
    max_tokens=MAX_TOKENS_THINKING,
    min_tokens=0,
    stop_token_ids=tok("</reasoning>")["input_ids"],
    skip_special_tokens=False,
    temperature=TEMPERATURE,
    frequency_penalty=0.1,
)

def process_datapoint(p, model, sampling_params_hardEnd, sampling_params_softEnd):
    # build prompt
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{p}<|im_end|>\n"
        f"<|im_start|>assistant\n<reasoning>"
    )
    # prepare storage for zero‑shot + 3 waits
    saved_responses = [(None, None)] * (NUM_IGNORE + 1)

    # zero‑shot pass
    o = model.generate(prompt, sampling_params=sampling_params_hardEnd)
    reasoning_trace = prompt + o[0].outputs[0].text
    o2 = model.generate(reasoning_trace + "</reasoning><answer>", sampling_params=sampling_params_softEnd)
    saved_responses[0] = (reasoning_trace + "</reasoning>", "<answer>" + o2[0].outputs[0].text)

    # NUM_IGNORE “Wait” passes
    for i in range(NUM_IGNORE):
        o = model.generate(reasoning_trace + "Wait", sampling_params=sampling_params_hardEnd)
        reasoning_trace = reasoning_trace + "Wait" + o[0].outputs[0].text
        o2 = model.generate(reasoning_trace + "</reasoning><answer>", sampling_params=sampling_params_softEnd)
        saved_responses[i+1] = (reasoning_trace + "</reasoning>", "<answer>" + o2[0].outputs[0].text)

    return saved_responses


# -----------------------------------------------------------------------------
# Process MATH-500
print("Processing MATH-500 dataset...")
df_math500 = pd.read_csv("MATH-500_test.csv")

zs, w1, w2, w3 = [], [], [], []
zs_a, w1_a, w2_a, w3_a = [], [], [], []

for _, row in tqdm(df_math500.iterrows(), total=len(df_math500), desc="MATH-500"):
    response = process_datapoint(row["problem"], model, sampling_params_hardEnd, sampling_params_softEnd)
    zs.append(response[0][0]);   w1.append(response[1][0])
    w2.append(response[2][0]);   w3.append(response[3][0])
    zs_a.append(response[0][1]); w1_a.append(response[1][1])
    w2_a.append(response[2][1]); w3_a.append(response[3][1])

df_math500["ZeroShot_ReasoningTrace"]      = zs
df_math500["Wait_1_ReasoningTrace"]        = w1
df_math500["Wait_2_ReasoningTrace"]        = w2
df_math500["Wait_3_ReasoningTrace"]        = w3
df_math500["ZeroShot_ReasoningTrace_ans"]  = zs_a
df_math500["Wait_1_ReasoningTrace_ans"]    = w1_a
df_math500["Wait_2_ReasoningTrace_ans"]    = w2_a
df_math500["Wait_3_ReasoningTrace_ans"]    = w3_a

df_math500.to_csv("math500_Phi_BudgetForcing.csv", index=False, quoting=csv.QUOTE_ALL)

# -----------------------------------------------------------------------------
# Process GPQA (multiple‑choice)
print("Processing GPQA dataset...")
df_gpqa = pd.read_csv("gpqa_main.csv")

zs, w1, w2, w3 = [], [], [], []
zs_a, w1_a, w2_a, w3_a = [], [], [], []
shuffled_opts = []

def process_gpqa_problem(row):
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    random.shuffle(choices)
    labels = ["A","B","C","D"]
    mapping = dict(zip(labels, choices))
    # figure out which label is now correct
    shuffled_label = next(l for l,t in mapping.items() if t == row["Correct Answer"])
    prompt = (
        f"{row['Question']}\n\nChoices:\n"
        + "\n".join(f"({lab}) {txt}" for lab,txt in mapping.items())
    )
    return prompt, shuffled_label

for _, row in tqdm(df_gpqa.iterrows(), total=len(df_gpqa), desc="GPQA"):
    prompt, corr_label = process_gpqa_problem(row)
    response = process_datapoint(prompt, model, sampling_params_hardEnd, sampling_params_softEnd)

    zs.append(response[0][0]);   w1.append(response[1][0])
    w2.append(response[2][0]);   w3.append(response[3][0])
    zs_a.append(response[0][1]); w1_a.append(response[1][1])
    w2_a.append(response[2][1]); w3_a.append(response[3][1])
    shuffled_opts.append(corr_label)

df_gpqa["ZeroShot_ReasoningTrace"]      = zs
df_gpqa["Wait_1_ReasoningTrace"]        = w1
df_gpqa["Wait_2_ReasoningTrace"]        = w2
df_gpqa["Wait_3_ReasoningTrace"]        = w3
df_gpqa["ZeroShot_ReasoningTrace_ans"]  = zs_a
df_gpqa["Wait_1_ReasoningTrace_ans"]    = w1_a
df_gpqa["Wait_2_ReasoningTrace_ans"]    = w2_a
df_gpqa["Wait_3_ReasoningTrace_ans"]    = w3_a
df_gpqa["shuffled_correct_option"]      = shuffled_opts

df_gpqa.to_csv("gpqa_Phi_BudgetForcing.csv", index=False, quoting=csv.QUOTE_ALL)
