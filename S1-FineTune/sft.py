import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, DatasetDict
import transformers
import trl
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments  # Import TrainingArguments

@dataclass
class TrainingConfig:
    model_name: str = field()
    block_size: int = field() 
    use_lora: bool = field()
    lora_r: int = field()
    lora_alpha: int = field()
    lora_dropout: float = field()
    wandb_project: Optional[str] = field(default="s1")
    wandb_entity: Optional[str] = field(default="yash9439")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    # Use HfArgumentParser to parse BOTH TrainingConfig and TrainingArguments
    parser = transformers.HfArgumentParser((TrainingConfig, TrainingArguments))  # Use TrainingArguments
    config, args = parser.parse_args_into_dataclasses()

    # 4-bit quantization config (BitsAndBytesConfig)
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load the model with quantization
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",  # ESSENTIAL for multi-GPU/CPU loading
        use_cache=False,    # Recommended with gradient checkpointing
    )


    # Load and prepare the dataset
    dataset = load_dataset(config.train_file_path)
    if isinstance(dataset, DatasetDict) and 'train' not in dataset:
        raise ValueError("DatasetDict must contain a 'train' split.")
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({'train': dataset})

    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        if "Llama" in config.model_name:
            tokenizer.pad_token = "<|reserved_special_token_5|>"
        elif "Qwen" in config.model_name:
            tokenizer.pad_token = "<|fim_pad|>" # Or another appropriate special token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    # Prepare for k-bit training *BEFORE* LoRA
    if config.use_lora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

        # LoRA configuration
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    # Instruction and response templates (adjust as needed)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
    else:
        instruction_template = "user"  # Or your custom instruction template
        response_template = "assistant"

    # Data collator
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # --- Set TrainingArguments correctly ---
    args.optim = "paged_adamw_8bit"
    args.gradient_checkpointing = True  
    args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    args.remove_unused_columns = True  # Good practice
    args.report_to = ["wandb"]  # Enable WandB reporting
    args.max_seq_length = config.block_size
    args.dataset_text_field = "text"

    # Log the configuration (combine dictionaries for a complete view)
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")


    # --- Create the SFTTrainer ---
    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else None,
        args=args,             # Pass the TrainingArguments object
        data_collator=collator,
        tokenizer=tokenizer,    # Pass the tokenizer
        peft_config=peft_config if config.use_lora else None, # Pass peft config

    )

    # Train and save
    trainer.train()
    trainer.save_model()  # Save the trained model (including LoRA weights)
    tokenizer.save_pretrained(args.output_dir) # save the tokenizer

if __name__ == "__main__":
    print(torch.cuda.is_available())
    train()
