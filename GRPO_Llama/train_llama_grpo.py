from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported # ハードウェアがbfloat16をサポートしているかチェックする
import torch
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning> <answer> answer here </answer>."""

def extract_xml_answer(text: str) -> str:
    answer = text.split('<answer>')[-1]
    answer = answer.split('</answer>')[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if '####' not in text:
        return None
    return text.split('####')[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split='train') -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            'prompt': [{
                'role': 'system',
                'content': SYSTEM_PROMPT
            }, {
                'role': 'user',
                'content': x['question']
            }],
            'answer':
            extract_hash_answer(x['answer'])
        })  # type: ignore
    return data  # type: ignore


def correctness_reward_func(prompts, completions, answer,
                            **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f'Question:\n{q}', f'\nAnswer:\n{answer[0]}',
          f'\nResponse:\n{responses[0]}',
          f'\nExtracted:\n{extracted_responses[0]}')
    return [
        2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)
    ]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r'^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$'
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r'<reasoning>.*?</reasoning>\s*<answer>.*?</answer>'
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count('<reasoning>\n') == 1:
        count += 0.125
    if text.count('\n</reasoning>\n') == 1:
        count += 0.125
    if text.count('\n<answer>\n') == 1:
        count += 0.125
        count -= len(text.split('\n</answer>\n')[-1]) * 0.001
    if text.count('\n</answer>') == 1:
        count += 0.125
        count -= (len(text.split('\n</answer>')[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]['content'] for completion in completions]
    return [count_xml(c) for c in contents]


def length_penalty_reward_func(completions, **kwargs) -> list[float]:
    """Penalize responses longer than 2500 characters with -2 reward"""
    rewards = []
    for completion in completions:
        content = completion[0]['content']
        rewards.append(-2.0 if len(content) > 2500 else 0.0)
    return rewards


def xml_repetition_penalty_func(completions, **kwargs) -> list[float]:
    """Penalize multiple XML tag appearances with -0.5 per extra occurrence"""
    penalties = []
    tags = ['<reasoning>', '</reasoning>', '<answer>', '</answer>']
    for completion in completions:
        content = completion[0]['content']
        penalty = 0.0
        for tag in tags:
            count = content.count(tag)
            if count > 1:
                penalty += (count - 1) * 0.5  # -0.5 per extra occurrence
        penalties.append(-penalty)
    return penalties


max_seq_length = 8192 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.4, 
)

# PEFT (LoRA) 
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

dataset = get_gsm8k_questions()

training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 512,
    max_completion_length = max_seq_length,
    num_train_epochs = 1, # Set to 1 for a full training run
    # max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
)



trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        length_penalty_reward_func,  # New length penalty
        xml_repetition_penalty_func,  # New XML repetition penalty
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()
     


model.save_lora("./grpo_saved_lora")
     

