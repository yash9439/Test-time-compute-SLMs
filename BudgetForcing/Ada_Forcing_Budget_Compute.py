import argparse
import json
from functools import partial
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List, Dict, Any
from transformers import BitsAndBytesConfig

# --- Configuration ---
DEFAULT_TEMP = 0.6
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_MAX_TOKENS = 8192
# DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_PATH = "../FineTune/ckpts/s1-20250223_212750"
DEFAULT_DEVICE = "cuda"

@torch.no_grad()
def stream_generate(
    prompt: torch.Tensor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sampler,
    max_tokens: int,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    *,
    stop_tokens: Optional[List[int]] = None
) :
    """Generates text (streaming), yielding tokens."""

    # Handle the case where prompt might be None
    if prompt is not None:
        input_ids = prompt.unsqueeze(0)  # Batch dimension
    else:
        # If prompt is None, we should have past_key_values.  We can't
        # generate *anything* without either a prompt or a cache.
        if past_key_values is None:
            raise ValueError("Must provide either 'prompt' or 'past_key_values'")
        # Use a dummy input_ids.  The model *requires* input_ids, even
        # when using past_key_values.  A single pad token will do.
        input_ids = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long, device=prompt.device if prompt is not None else "cuda") # Use a default device if prompt is None


    generated_tokens = 0
    for _ in range(max_tokens):
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]  # Last token logits
        next_token_id = sampler(logits).unsqueeze(0)

        generated_tokens += 1
        if stop_tokens and next_token_id.item() in stop_tokens:
            break

        past_key_values = outputs.past_key_values
        input_ids = next_token_id

        yield {
            "text": tokenizer.decode(next_token_id.squeeze()),
            "token": next_token_id.item(),
            "generation_tokens": generated_tokens,
            "past_key_values": past_key_values,
        }


def make_prompt_cache(model):
    return None

def trim_prompt_cache(past_key_values, num_to_trim):
    """Trims the past key values by removing the first 'num_to_trim' tokens."""
    if past_key_values is None or num_to_trim == 0:
        return past_key_values

    # Import DynamicCache here to avoid circular imports if necessary
    from transformers.cache_utils import DynamicCache

    if isinstance(past_key_values, DynamicCache):
        new_cache = DynamicCache()
        for layer_idx in range(len(past_key_values)):
            key = past_key_values.key_cache[layer_idx]
            value = past_key_values.value_cache[layer_idx]
            # Trim the first 'num_to_trim' tokens from the sequence dimension (axis=2)
            key_trimmed = key[:, :, num_to_trim:, :]
            value_trimmed = value[:, :, num_to_trim:, :]
            new_cache.update(key_trimmed, value_trimmed, layer_idx)
        return new_cache
    else:
        # Handle legacy tuple format
        new_cache = []
        for layer_cache in past_key_values:
            k_cache, v_cache = layer_cache
            # Trim the first 'num_to_trim' tokens along the sequence dimension (axis=2)
            new_k = k_cache[:, :, num_to_trim:, :] if k_cache is not None else None
            new_v = v_cache[:, :, num_to_trim:, :] if v_cache is not None else None
            new_cache.append((new_k, new_v))
        return tuple(new_cache)


def load(model_path: str, adapter_path: Optional[str] = None, tokenizer_config: Dict = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads model and tokenizer with 4-bit quantization"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **(tokenizer_config or {}))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if adapter_path:
        print("[WARNING] Adapter loading with 4-bit quantization may have compatibility issues")
        # model.load_adapter(adapter_path)

    return model, tokenizer


def make_sampler(temp: float, top_p: float):
    """Creates a sampling function."""
    def sampler(logits: torch.Tensor) -> torch.Tensor:
        if temp <= 1e-5:
            return torch.argmax(logits, dim=-1)

        probs = torch.softmax(logits / temp, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs <= (1 - top_p)
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = True
            sorted_probs = sorted_probs.masked_fill(~mask, -float('inf'))
            probs.scatter_(-1, sorted_indices, sorted_probs)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    return sampler




def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH, help="Adapter path")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP, help="Temperature")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Top-p")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device (cuda/cpu)")

    try:
        if '-f' in sys.argv:
            sys.argv.remove('-f')
        args = parser.parse_args()
    except SystemExit as e:
        print(f"SystemExit caught: {e}")
        args = parser.parse_args([])

    torch.manual_seed(args.seed)

    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )
    model.to(args.device)
    model.eval()

    wait_token = "Wait"
    wait_token_id = tokenizer.convert_tokens_to_ids(wait_token)
    end_think_token = "</think>"
    end_think_token_id = tokenizer.convert_tokens_to_ids(end_think_token)

    think_more_prompt = torch.tensor([wait_token_id], dtype=torch.long, device=args.device)
    end_think_prompt = torch.tensor(
        tokenizer.encode(end_think_token + "\n", add_special_tokens=False),
        dtype=torch.long,
        device=args.device,
    )

    generator = partial(
        stream_generate,
        model=model,
        tokenizer=tokenizer,
        sampler=make_sampler(args.temp, args.top_p),
        stop_tokens=[tokenizer.eos_token_id, end_think_token_id, wait_token_id],
        past_key_values=None,
    )

    print(f"[INFO] Starting reasoning session with {args.model}.  Exit: 'q'.")
    while True:
        past_key_values = None  # Reset cache
        query = input(">> ")
        if query == "q":
            break
        messages = [{"role": "user", "content": query}]

        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt = tokenizer.encode(prompt, add_special_tokens=False)
        prompt = torch.tensor(prompt, dtype=torch.long, device=args.device)


        while True:
            max_tokens = args.max_tokens
            end_think_idx = None
            response_counter = 0
            for response in generator(prompt=prompt, max_tokens=max_tokens, past_key_values=past_key_values):
                past_key_values = response['past_key_values']
                response_counter += 1

                if response["token"] == wait_token_id:
                    break
                elif response["token"] == end_think_token_id:
                    end_think_idx = response_counter - 1
                print(response["text"], flush=True, end="")

            max_tokens -= response["generation_tokens"]
            prompt = None # Set prompt to None after the first inner loop

            if end_think_idx is None:
                print(end_think_token, flush=True)
                past_key_values = trim_prompt_cache(past_key_values, 1)

                for response in generator(prompt=end_think_prompt, max_tokens=max_tokens, past_key_values=past_key_values):
                    past_key_values = response['past_key_values']
                    print(response["text"], flush=True, end="")
                max_tokens -= response["generation_tokens"]


            think_more = input("\n\n\033[31mThink more? (y/n):\033[0m ")
            if think_more == "y":
                print("<think>")
                print(wait_token, flush=True, end="")
                if end_think_idx is not None:
                    num_to_trim = response_counter - end_think_idx
                else:
                    num_to_trim = 1

                max_tokens += num_to_trim
                past_key_values = trim_prompt_cache(past_key_values, num_to_trim)
                # Pass past_key_values and think_more_prompt
                for response in generator(prompt=think_more_prompt, max_tokens=max_tokens, past_key_values=past_key_values):
                    past_key_values = response['past_key_values']
                    print(response["text"], flush=True, end="")
                max_tokens -= response["generation_tokens"]

            else:
                break
        print()

if __name__ == "__main__":
    main()
