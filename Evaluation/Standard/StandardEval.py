from unsloth import FastLanguageModel
import pandas as pd
import torch
from tqdm import tqdm

aime24_file_name="aime24_Standard.csv"
math500_file_name="math500_Standard.csv"
gpqa_file_name="gpqa_Standard.csv"

aime24_outputfile_name =  aime24_file_name.split(".")[0] + "_Evaluation.csv"
math500_outputfile_name =  math500_file_name.split(".")[0] + "_Evaluation.csv"
gpqa_outputfile_name =  gpqa_file_name.split(".")[0] + "_Evaluation.csv"

df_aime24 = pd.read_csv(aime24_file_name)
df_math500 = pd.read_csv(math500_file_name)
df_gpqa = pd.read_csv(gpqa_file_name)


max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None          # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True   # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

FastLanguageModel.for_inference(model)

# -------------------------------------------------------------------------------

def process_problem(model_response, solution, tokenizer, model):
    # Last 100 tokens of the model response
    model_response = model_response.split()[-100:]

    messages = [
        {"role": "user", "content": f"Is the numerical answer in the text below equivalent to '{solution} ignoring formatting (like LaTeX)?\n\nText: {model_response}"},
    ]
    
    # Create inputs with tokenization
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    # Create attention mask (fix #1)
    attention_mask = torch.ones_like(inputs).to("cuda")
    
    output_ids = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,  # Add attention mask
        max_new_tokens=500, 
        use_cache=True, 
        temperature=0.01, 
        min_p=0
    )
    
    # Decode and store output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def process_problem_gpqa(model_response, expected_text, expected_option, tokenizer, model):
    # Last 100 tokens of the model response
    model_response = model_response.split()[-100:]

    messages = [
        {"role": "user", "content": f"Does the following text state that the answer is either option '{expected_option}' OR the text '{expected_text}'?  Consider variations in wording. Respond with 'YES' only if the text clearly indicates one of these as the answer, otherwise respond with 'NO'.\n\nText: {model_response}"},
    ]

    # Create inputs with tokenization
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    # Create attention mask (fix #1)
    attention_mask = torch.ones_like(inputs).to("cuda")
    
    output_ids = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,  # Add attention mask
        max_new_tokens=500, 
        use_cache=True, 
        temperature=0.01, 
        min_p=0
    )
    
    # Decode and store output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# -------------------------------------------------------------------------------
# Process AIME24 dataset with tqdm progress bar

print("Processing AIME24 dataset...")

# Add tqdm progress bar (fix #2)
responses = []
for _, row in tqdm(df_aime24.iterrows(), total=len(df_aime24), desc="AIME24"):
    response = process_problem(row["model_response"], row["solution"], tokenizer, model)
    responses.append(response)

df_aime24["LLM_as_Judge"] = responses

# Save the updated dataset as a CSV file
df_aime24.to_csv(aime24_outputfile_name, index=False)

# -------------------------------------------------------------------------------
# Process MATH-500 dataset with tqdm progress bar

print("Processing MATH-500 dataset...")

# Add tqdm progress bar (fix #2)
responses = []
for _, row in tqdm(df_math500.iterrows(), total=len(df_math500), desc="MATH-500"):
    response = process_problem(row["model_response"], row["answer"], tokenizer, model)
    responses.append(response)

df_math500["LLM_as_Judge"] = responses

# Save the updated dataset as a CSV file
df_math500.to_csv(math500_outputfile_name, index=False)

# -------------------------------------------------------------------------------
# Process GPQA dataset with tqdm progress bar

print("Processing GPQA dataset...")
# Add tqdm progress bar (fix #2)
responses = []
for _, row in tqdm(df_gpqa.iterrows(), total=len(df_gpqa), desc="GPQA"):
    response = process_problem_gpqa(row["model_response"], row["Correct Answer"], row["shuffled_correct_option"], tokenizer, model)
    responses.append(response)

df_gpqa["LLM_as_Judge"] = responses

# Save the updated dataset as a CSV file
df_gpqa.to_csv(gpqa_file_name, index=False)


# -------------------------------------------------------------------------------