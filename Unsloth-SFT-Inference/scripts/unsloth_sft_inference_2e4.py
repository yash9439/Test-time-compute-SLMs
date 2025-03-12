from unsloth import FastLanguageModel
import pandas as pd
import torch
from tqdm import tqdm

# Define model loading parameters
max_seq_length = 8192  # Same as your training script
dtype = None           # Auto detection
load_in_4bit = True    # Use 4bit quantization to reduce memory usage

# Load your fine-tuned model - specify the path to where you saved it
model_path = "5Epoch_2e_4"  # Change this to your model's directory name
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Set model to inference mode
FastLanguageModel.for_inference(model)

# -------------------------------------------------------------------------------

def process_problem(problem_text, tokenizer, model):
    messages = [
        {"role": "user", "content": f"{problem_text}"},
    ]
    
    # Create inputs with tokenization
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    # print(tokenizer.decode(inputs[0].tolist(), skip_special_tokens=False))
    
    # Create attention mask
    attention_mask = torch.ones_like(inputs).to("cuda")
    
    output_ids = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        max_new_tokens=8192, 
        use_cache=True, 
        temperature=0.01,  # Low temperature for more deterministic outputs
        min_p=0
    )
    
    # Decode and return output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# -------------------------------------------------------------------------------
# Process AIME24 dataset with tqdm progress bar

print("Processing AIME24 dataset...")
df_aime24 = pd.read_csv("AIME24_test.csv")  # Update path as needed

responses = []
for _, row in tqdm(df_aime24.iterrows(), total=len(df_aime24), desc="AIME24"):
    response = process_problem(row["problem"], tokenizer, model)
    # print(response)
    responses.append(response)

df_aime24["model_response"] = responses
df_aime24.to_csv("aime24_2e4_sft.csv", index=False)

# -------------------------------------------------------------------------------
# Process MATH-500 dataset with tqdm progress bar

print("Processing MATH-500 dataset...")
df_math500 = pd.read_csv("MATH-500_test.csv")  # Update path as needed

responses = []
for _, row in tqdm(df_math500.iterrows(), total=len(df_math500), desc="MATH-500"):
    response = process_problem(row["problem"], tokenizer, model)
    responses.append(response)

df_math500["model_response"] = responses
df_math500.to_csv("math500_2e4_sft.csv", index=False)

# -------------------------------------------------------------------------------
# Process GPQA dataset (with multiple choice handling)

print("Processing GPQA dataset...")
df_gpqa = pd.read_csv("gpqa_main.csv")  # Update path as needed

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
    
    messages = [{"role": "user", "content": problem_text}]
    
    # Tokenization
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    
    # Create attention mask
    attention_mask = torch.ones_like(inputs).to("cuda")

    # Model inference
    output_ids = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        max_new_tokens=8192, 
        use_cache=True, 
        temperature=0.01, 
        min_p=0
    )
    
    # Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text, shuffled_correct_option

# Process GPQA problems
responses = []
correct_options = []
for _, row in tqdm(df_gpqa.iterrows(), total=len(df_gpqa), desc="GPQA"):
    response, correct_option = process_gpqa_problem(row, tokenizer, model)
    responses.append(response)
    correct_options.append(correct_option)

df_gpqa["model_response"] = responses
df_gpqa["shuffled_correct_option"] = correct_options

# Save the updated dataset
df_gpqa.to_csv("gpqa_2e4_sft.csv", index=False)


