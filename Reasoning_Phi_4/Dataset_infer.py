import torch
import re
import pandas as pd
from tqdm.auto import tqdm
import random
import os

# --- Import Transformers components ---
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. Configuration ---
MODEL_NAME = "microsoft/Phi-4-mini-instruct"

AIME_INPUT_CSV = "AIME24_test.csv"
MATH500_INPUT_CSV = "MATH-500_test.csv"
GPQA_INPUT_CSV = "gpqa_main.csv"

# Output filenames for Hugging Face pipeline results
AIME_OUTPUT_CSV = "aime24_Phi4mini_HF_zeroshot_single.csv"
MATH500_OUTPUT_CSV = "math500_Phi4mini_HF_zeroshot_single.csv"
GPQA_OUTPUT_CSV = "gpqa_Phi4mini_HF_zeroshot_single.csv"

# --- Check if input files exist ---
required_files = [AIME_INPUT_CSV, MATH500_INPUT_CSV, GPQA_INPUT_CSV]
for f in required_files:
    if not os.path.exists(f):
        print(f"Error: Required file not found: {f}")
        exit()

# --- 2. Model and Tokenizer Loading (Hugging Face) ---
print(f"Loading tokenizer for: {MODEL_NAME}...")
# Add pad_token if it doesn't exist, like Phi-3 requires
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Common practice
    print("Set pad_token = eos_token")
print("Tokenizer loaded.")


print(f"Loading model with Hugging Face transformers: {MODEL_NAME}...")
model_hf = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    # attn_implementation="flash_attention_2" # Optional: Use Flash Attention if available/compatible
)
print("Hugging Face Model loaded.")

# --- 3. Pipeline Setup ---
print("Creating Hugging Face text-generation pipeline...")
hf_pipeline = pipeline(
    "text-generation",
    model=model_hf,
    tokenizer=tokenizer,
)
print("Hugging Face Pipeline created.")

# --- 4. Generation Arguments (Hugging Face Pipeline) ---
generation_args_hf = {
    "max_new_tokens": 8192,
    "return_full_text": False,
    "temperature": 0.8,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id # Use the assigned pad token ID
}
print("Hugging Face generation arguments defined.")
print(f"Using deterministic generation: {not generation_args_hf['do_sample']}")


# --- 5. Helper Function ---
def extract_answer(text: str) -> str:
    """Extracts content within the <answer>...</answer> tags."""
    if not isinstance(text, str): # Handle potential errors where generation failed
        return "ERROR_INVALID_INPUT"
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback for final number extraction - adjust if needed based on model output format
    final_num_match = re.search(r"(?:answer is|is|final answer:|:)\s*([\d\.\-]+)$", text.strip(), re.IGNORECASE | re.MULTILINE)
    if final_num_match:
        return final_num_match.group(1).strip()
    # Fallback for just a letter (GPQA) if tags missing
    final_letter_match = re.search(r"\b([A-D])\s*$", text.strip())
    if final_letter_match:
         return final_letter_match.group(1)

    return "N/A" # Return "N/A" if no structured answer is found

# --- 6. System Prompt ---
SYSTEM_PROMPT = """You are a helpful AI assistant. Analyze the user's question carefully and provide a step-by-step derivation. Respond strictly in the following format, ensuring both tags are present and complete:
<reasoning>
Provide detailed step-by-step thinking here, explaining the logic used to reach the solution.
</reasoning>
<answer>
Provide ONLY the final answer here. If the question requires a specific numerical value, provide only that number (as an integer if possible). If the question requires a formula, expression, or general method, provide that concisely. Do not include units, surrounding text, explanations, or any other content within this tag. For multiple-choice questions, provide ONLY the letter of the correct choice (e.g., A, B, C, or D).
</answer>"""

# --- 7. Single Item Processing Function (using HF Pipeline) ---
def process_single_item_hf(problem_text):
    """Formats a single problem and generates a response using the HF pipeline."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    # Pipeline works well with the formatted prompt string
    prompt_str = tokenizer.apply_chat_template(
         messages,
         tokenize=False,
         add_generation_prompt=True
    )

    try:
        # Pass the single formatted prompt string and generation args
        # The pipeline expects a list, so wrap the string in a list
        outputs = hf_pipeline([prompt_str], **generation_args_hf)
        # Extract generated text (output format is List[List[Dict]])
        generated_text = outputs[0][0]['generated_text']
        return generated_text
    except Exception as e:
        print(f"\nError during Hugging Face pipeline generation for item: {problem_text[:100]}...")
        print(f"Error: {e}")
        # Return error placeholder
        return "ERROR_GENERATING_RESPONSE"

# --- 8. GPQA Specific Processing Function (Single Item using HF Pipeline) ---
def process_single_gpqa_hf(row):
    """Formats a single GPQA problem row, shuffles choices, generates response."""
    # Check for missing data first
    if pd.isna(row["Correct Answer"]) or pd.isna(row["Incorrect Answer 1"]) or pd.isna(row["Incorrect Answer 2"]) or pd.isna(row["Incorrect Answer 3"]):
        print(f"Warning: Skipping GPQA row index {row.name} due to missing data.")
        return "SKIPPED_MISSING_DATA", "N/A_MISSING_DATA"

    # Proceed if data is valid
    choices = [
        row["Correct Answer"], row["Incorrect Answer 1"],
        row["Incorrect Answer 2"], row["Incorrect Answer 3"],
    ]
    random.shuffle(choices)
    choice_labels = ["A", "B", "C", "D"]
    choice_mapping = dict(zip(choice_labels, choices))
    shuffled_correct_option_label = next(label for label, text in choice_mapping.items() if text == row["Correct Answer"])

    problem_text = f"""{row['Question']}

Choices:
(A) {choice_mapping['A']}
(B) {choice_mapping['B']}
(C) {choice_mapping['C']}
(D) {choice_mapping['D']}

Provide your reasoning and select the single best answer from the choices (A, B, C, D). Your final answer should be enclosed in <answer></answer> tags and contain ONLY the letter corresponding to the correct choice (e.g., <answer>C</answer>)."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        # Generate response for the single prompt
        outputs = hf_pipeline([prompt_str], **generation_args_hf)
        generated_text = outputs[0][0]['generated_text']
        return generated_text, shuffled_correct_option_label
    except Exception as e:
        print(f"\nError during Hugging Face pipeline GPQA generation for row index {row.name}: {row['Question'][:100]}...")
        print(f"Error: {e}")
        return "ERROR_GENERATING_RESPONSE", shuffled_correct_option_label # Return error but keep correct option if calculated

# --- 9. Main Dataset Processing Loop (Single Item Processing) ---

# --- AIME24 ---
print(f"\n--- Processing AIME24 dataset ({AIME_INPUT_CSV}) ---")
df_aime24 = pd.read_csv(AIME_INPUT_CSV)
aime_responses = []
# Iterate row by row with tqdm
for index, row in tqdm(df_aime24.iterrows(), total=len(df_aime24), desc="AIME24"):
    problem = row["problem"]
    response = process_single_item_hf(problem)
    aime_responses.append(response)

df_aime24["model_response_raw"] = aime_responses
df_aime24["model_answer_extracted"] = [extract_answer(resp) for resp in df_aime24["model_response_raw"]]
df_aime24.to_csv(AIME_OUTPUT_CSV, index=False)
print(f"AIME24 results saved to {AIME_OUTPUT_CSV}")

# --- MATH-500 ---
print(f"\n--- Processing MATH-500 dataset ({MATH500_INPUT_CSV}) ---")
df_math500 = pd.read_csv(MATH500_INPUT_CSV)
math500_responses = []
# Iterate row by row with tqdm
for index, row in tqdm(df_math500.iterrows(), total=len(df_math500), desc="MATH-500"):
    problem = row["problem"]
    response = process_single_item_hf(problem)
    math500_responses.append(response)

df_math500["model_response_raw"] = math500_responses
df_math500["model_answer_extracted"] = [extract_answer(resp) for resp in df_math500["model_response_raw"]]
df_math500.to_csv(MATH500_OUTPUT_CSV, index=False)
print(f"MATH-500 results saved to {MATH500_OUTPUT_CSV}")

# --- GPQA ---
print(f"\n--- Processing GPQA dataset ({GPQA_INPUT_CSV}) ---")
df_gpqa = pd.read_csv(GPQA_INPUT_CSV)
gpqa_responses = []
gpqa_correct_options = []
# Iterate row by row with tqdm
for index, row in tqdm(df_gpqa.iterrows(), total=len(df_gpqa), desc="GPQA"):
    # Use the GPQA specific single item processor
    response, correct_option = process_single_gpqa_hf(row)
    gpqa_responses.append(response)
    gpqa_correct_options.append(correct_option)

df_gpqa["model_response_raw"] = gpqa_responses
df_gpqa["shuffled_correct_option_label"] = gpqa_correct_options
df_gpqa["model_answer_extracted"] = [extract_answer(resp) for resp in df_gpqa["model_response_raw"]]
df_gpqa.to_csv(GPQA_OUTPUT_CSV, index=False)
print(f"GPQA results saved to {GPQA_OUTPUT_CSV}")

# --- 10. Optional: Test with initial questions (already processed one by one) ---
print("\n--- Processing Example Questions (Zero-Shot using HF Pipeline) ---")
test_questions = [
    "What is the sum of 3 and 5?",
    "How do you solve a quadratic equation?",
    "What is the derivative of x^2?",
    "Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?"
]
for question in test_questions:
    print(f"\nProcessing Question: {question}")
    # Process using the single item function
    response_text = process_single_item_hf(question)
    extracted_ans = extract_answer(response_text)
    print(f"Model Full Response:\n{response_text}")
    print(f"Extracted Answer: {extracted_ans}")

print("\n--- Finished ---")
