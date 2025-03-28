from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported # ハードウェアがbfloat16をサポートしているかチェックする
import torch
import re
import pandas as pd
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from tqdm import tqdm

SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning> <answer> answer here </answer>."""

max_seq_length = 8192 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = max_seq_length,
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.4, 
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

# -------------------------------------------------------------------------------

def process_problem(problem_text, tokenizer, model):
    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : SYSTEM_PROMPT},
        {"role" : "user", "content" : problem_text},
    ], tokenize = False, add_generation_prompt = True)

    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = model.load_lora("outputs/checkpoint-2000"),
    )[0].outputs[0].text

    return output

# -------------------------------------------------------------------------------
# Process AIME24 dataset with tqdm progress bar

print("Processing AIME24 dataset...")
df_aime24 = pd.read_csv("AIME24_test.csv")

# Add tqdm progress bar (fix #2)
responses = []
for _, row in tqdm(df_aime24.iterrows(), total=len(df_aime24), desc="AIME24"):
    response = process_problem(row["problem"], tokenizer, model)
    responses.append(response)

df_aime24["model_response"] = responses

# Save the updated dataset as a CSV file
df_aime24.to_csv("aime24_LLama_2000Checkpoint.csv", index=False)


# -------------------------------------------------------------------------------
# Process MATH-500 dataset with tqdm progress bar

print("Processing MATH-500 dataset...")
df_math500 = pd.read_csv("MATH-500_test.csv")

# Add tqdm progress bar (fix #2)
responses = []
for _, row in tqdm(df_math500.iterrows(), total=len(df_math500), desc="MATH-500"):
    response = process_problem(row["problem"], tokenizer, model)
    responses.append(response)

df_math500["model_response"] = responses

# Save the updated dataset as a CSV file
df_math500.to_csv("math500_LLama_2000Checkpoint.csv", index=False)

# -------------------------------------------------------------------------------
# Process GPQA dataset with tqdm progress bar

print("Processing GPQA dataset...")
df_gpqa = pd.read_csv("gpqa_main.csv")

import random

def process_gpqa_problem(row, tokenizer, model):
    choices = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    
    # Shuffle choices
    random.shuffle(choices)
    
    # Map shuffled choices to labels
    choice_labels = ["A", "B", "C", "D"]
    choice_mapping = dict(zip(choice_labels, choices))
    
    # Find new correct option
    shuffled_correct_option = next(label for label, text in choice_mapping.items() if text == row["Correct Answer"])
    
    # Construct the prompt
    problem_text = f"""{row['Question']}
    
Choices:
(A) {choice_mapping['A']}
(B) {choice_mapping['B']}
(C) {choice_mapping['C']}
(D) {choice_mapping['D']}
"""
    
    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : SYSTEM_PROMPT},
        {"role" : "user", "content" : problem_text},
    ], tokenize = False, add_generation_prompt = True)

    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = model.load_lora("outputs/checkpoint-2000"),
    )[0].outputs[0].text
    
    return output, shuffled_correct_option

# Add tqdm progress bar (fix #2)
responses = []
correct_options = []
for _, row in tqdm(df_gpqa.iterrows(), total=len(df_gpqa), desc="GPQA"):
    response, correct_option = process_gpqa_problem(row, tokenizer, model)
    responses.append(response)
    correct_options.append(correct_option)

df_gpqa["model_response"] = responses
df_gpqa["shuffled_correct_option"] = correct_options

# Save the updated dataset
df_gpqa.to_csv("gpqa_LLama_2000Checkpoint.csv", index=False)
