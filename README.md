# Test-Time Optimization for Small Language Models (SLMs)

**Authors:** Yash Bhaskar, Akanksha Srivastava

This repository contains the code, experiments, results, and analysis for the research project exploring test-time compute allocation strategies to improve the reasoning capabilities of Small Language Models (SLMs).

## Abstract

Small Language Models (SLMs) often struggle with complex reasoning tasks despite possessing relevant knowledge. This work explores test-time compute allocation as a critical factor influencing reasoning performance, an area currently underexplored for SLMs. We investigate techniques like budget forcing, compare Chain of Thought (CoT) with Chain of Draft (CoD) prompting, and experiment with Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) to enhance SLM reasoning. We benchmark performance on datasets like MATH-500, AIME24, and GPQA, detailing our attempts to replicate and adapt methods from larger models to SLMs (specifically Qwen-2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, and Phi-4-3.8B-mini-Instruct). Our findings highlight the potential of test-time optimization while underscoring the challenges in achieving consistent reasoning format adherence and stability with current SLMs.

## Table of Contents

-   [Introduction](#introduction)
-   [Techniques Explored](#techniques-explored)
-   [Models & Datasets](#models--datasets)
-   [Project Structure](#project-structure)
-   [Setup](#setup)
-   [Running Experiments](#running-experiments)
-   [Results & Analysis](#results--analysis)
-   [Key Findings](#key-findings)
-   [License](#license)

## Introduction

While Large Language Models (LLMs) show impressive capabilities, their reasoning performance can often be improved by allocating more computational resources ("thinking time") at inference. This aspect is less studied for SLMs. This project investigates how controlling or optimizing test-time compute can enhance the reasoning performance of SLMs, adapting and evaluating techniques proposed for larger models. We focus on making SLMs follow specific reasoning formats and inducing longer, potentially self-correcting, thought processes.

## Techniques Explored

1.  **Baseline Prompting:**
    *   Standard Prompting
    *   Zero-Shot Prompting
    *   Chain of Thought (CoT) Prompting
    *   Chain of Draft (CoD) Prompting
    *   Zero-Shot with R1-style Prompting
2.  **Supervised Fine-Tuning (SFT):** Fine-tuning instruction models (Qwen-1.5B) on the S1K dataset [3] to teach the `<think>...</think><answer>...</answer>` format, exploring different learning rates.
3.  **Group Relative Policy Optimization (GRPO):** Aligning models (Llama-3.2-1B) using reward functions based on correctness, format adherence, and length, trained on GSM8K.
4.  **Budget Forcing:** Implementing mechanisms to control reasoning length:
    *   Maximum Budget (Stopping Early)
    *   Minimum Budget (Forcing Longer Thinking / "Wait" prompt)
5.  **Progressive Reasoning Expansion:** A custom method developed to forcefully ensure the reasoning format (`<reasoning>...</reasoning><answer>...</answer>`) and progressively extend thinking time by inserting "Wait" prompts. Applied directly to Phi-4-mini.

## Models & Datasets

**Models:**

*   Qwen-2.5-1.5B-Instruct
*   Llama-3.2-1B-Instruct
*   Phi-4-3.8B-mini-Instruct
*   Deepseek-R1-Distill-Qwen-1.5B (Evaluated for format adherence)

**Benchmark Datasets:**

*   MATH-500
*   AIME24
*   GPQA

**Training Datasets:**

*   S1K [3] (Used for SFT)
*   GSM8K (Used for GRPO)

## Project Structure

```
.
├── assets/                 # Images used in reports/presentations
├── BudgetForcing-Implementation/ # Code for Budget Forcing / Progressive Reasoning Expansion
│   └── Ada_Forcing_Budget_Compute.py # Implementation script
├── Evaluation/             # Contains raw evaluation CSVs and some eval scripts (potentially older/redundant, see QuantitativeAnalysis)
│   ├── COD/
│   ├── COT/
│   ├── SFT/
│   ├── Standard/
│   └── ZeroShot/
├── GRPO_Llama/             # Experiments with GRPO on Llama-3.2-1B
│   ├── 2000Checkpoint_Temp0.8/ # Inference results from a GRPO checkpoint
│   ├── grpo_saved_lora/        # Saved LoRA adapter from GRPO training
│   ├── outputs/                # Training checkpoints and logs for GRPO
│   ├── train_llama_grpo.py     # GRPO training script (likely using Unsloth/TRL)
│   └── wandb/                  # Weights & Biases logs for GRPO runs
├── LICENSE                 # Project License
├── Presentation.pdf/pptx   # Presentation slides
├── Prompting-Baseline/     # Baseline prompting experiments (Standard, ZeroShot, CoT, CoD, R1)
│   ├── ChainOfDraft/
│   ├── ChainOfThought/
│   ├── R1/
│   ├── Standard/
│   ├── ZeroShot/
│   ├── *.csv                   # Input benchmark datasets
│   └── */unsloth_*.py          # Inference scripts for each baseline method
│   └── */*.csv                 # Raw output results for each baseline method
├── QuantitativeAnalysis/   # Central hub for results analysis and final tables
│   ├── Eval.ipynb              # General/Older evaluation notebook
│   ├── Llama/                  # Analysis specific to Llama GRPO results
│   ├── Phi/                    # Analysis specific to Phi-4 results (ZeroShot, BudgetForcing/Progressive)
│   ├── Qwen/                   # Analysis specific to Qwen results (Baselines, SFT)
│   ├── R1/                     # Analysis specific to R1 prompting baseline
│   ├── */*.ipynb               # Jupyter notebooks for data processing and analysis
│   ├── */*_extracted.csv       # Processed/cleaned results ready for analysis
│   └── tables.pdf/table_2.pdf  # Generated tables summarizing results
├── README.md               # This file
├── Reasoning_Phi_4/        # Experiments specifically with Phi-4-mini
│   ├── Dataset_infer.py        # Inference script for Phi-4 (likely ZeroShot + BudgetForcing/Progressive)
│   └── *.csv                   # Raw output results for Phi-4 experiments
├── Report.pdf              # The final research report
├── S1-FineTune/            # SFT experiments attempting to replicate S1 paper methodology (non-Unsloth)
│   ├── sft.py                  # SFT training script
│   └── *.sh/txt                # Scripts and logs
├── Unsloth-SFT-Finetune/   # SFT experiments using the Unsloth library
│   ├── 5Epoch_*/               # Saved LoRA adapters for different learning rates
│   ├── outputs_*/              # Training checkpoints and logs for SFT runs
│   ├── unsloth_*.py            # SFT training scripts for different learning rates
│   └── *.sh/log/txt            # Execution scripts and logs
└── Unsloth-SFT-Inference/  # Inference using the SFT models trained with Unsloth
    ├── AIME24_SFT_Response/    # SFT Inference results for AIME24
    ├── GPQA_SFT_Response/      # SFT Inference results for GPQA
    ├── Math500_SFT _Response/  # SFT Inference results for MATH500
    ├── Model Weights/          # Copies of the trained LoRA adapters used for inference
    ├── scripts/                # Inference scripts for each SFT model
    └── stdout/                 # Logs from inference runs
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Test-time-compute-SLMs
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  **Install dependencies:**
    While a central `requirements.txt` is missing, key libraries used across the project include:
    *   `torch`
    *   `transformers`
    *   `datasets`
    *   `pandas`
    *   `unsloth` (for Unsloth-based SFT/Inference and potentially GRPO)
    *   `trl` (likely used for GRPO)
    *   `accelerate`
    *   `bitsandbytes` (for quantization)
    *   `wandb` (optional, for logging GRPO runs)
    *   `jupyter` (for running analysis notebooks)
    Install these as needed, e.g.:
    ```bash
    pip install torch transformers datasets pandas accelerate bitsandbytes jupyter
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" # Example for Colab
    pip install trl wandb
    ```
    Refer to individual scripts for specific imports.

## Running Experiments

*   **Baseline Prompting:** Navigate to `Prompting-Baseline/` and run the `unsloth_*.py` scripts within the respective subdirectories (e.g., `Prompting-Baseline/ChainOfThought/unsloth_cot.py`). Input data CSVs are in the parent directory.
*   **SFT Training (Unsloth):** Navigate to `Unsloth-SFT-Finetune/`. Use `bash_script.sh` or run individual `unsloth_5Epoch_*.py` scripts. Models/adapters are saved in correspondingly named directories (e.g., `5Epoch_1e_4/`).
*   **SFT Inference (Unsloth):** Navigate to `Unsloth-SFT-Inference/`. Use `bash_script.sh` or run individual scripts from the `scripts/` directory (e.g., `scripts/unsloth_sft_inference_1e4.py`). Ensure the correct model paths (adapters) from `Unsloth-SFT-Finetune/` or `Model Weights/` are referenced.
*   **GRPO Training (Llama):** Navigate to `GRPO_Llama/`. Run `train_llama_grpo.py`. Checkpoints are saved in `outputs/`. Requires GSM8K dataset access.
*   **GRPO Inference (Llama):** Use `dataset_inference_llama_grpo.py` in `GRPO_Llama/2000Checkpoint_Temp0.8/`, adapting paths as needed.
*   **Budget Forcing / Progressive Expansion (Phi-4):** Navigate to `Reasoning_Phi_4/`. Run `Dataset_infer.py`. The logic likely incorporates code concepts from `BudgetForcing-Implementation/Ada_Forcing_Budget_Compute.py`.
*   **Analysis:** Navigate to `QuantitativeAnalysis/` and run the Jupyter notebooks (`*.ipynb`) to process raw results and generate analysis/tables.

## Results & Analysis

*   **Raw experiment outputs (CSV files):** Located within the specific experiment directories (e.g., `Prompting-Baseline/ZeroShot/`, `Unsloth-SFT-Inference/AIME24_SFT_Response/`, `Reasoning_Phi_4/`).
*   **Processed results and analysis notebooks:** Found in `QuantitativeAnalysis/`, categorized by model/method. Look for `*_extracted.csv` files and `*.ipynb` notebooks.
*   **Summarized results:** Presented in `QuantitativeAnalysis/tables.pdf` and `QuantitativeAnalysis/table_2.pdf`.
*   **Detailed discussion:** See the full `Report.pdf`.

## Key Findings

*   **Format Adherence Challenge:** SLMs (especially 1B-2B range) struggle with reliably adhering to structured reasoning formats (`<think>...</think>`) required for techniques like budget forcing.
*   **Model Stability Issues:** SFT and GRPO applied to SLMs can lead to instability (loops, gibberish, excessive verbosity), particularly sensitive to hyperparameters (like learning rate) and model quirks (e.g., Qwen ignoring system prompts).
*   **GRPO Complexity:** Effective GRPO requires careful reward function design and is prone to instability or reward hacking with lower-capacity models.
*   **Instruction Following is Crucial:** Models with better inherent instruction-following (like Phi-4-mini) are more amenable to direct test-time techniques like budget forcing/progressive expansion without potentially destabilizing fine-tuning.
*   **Budget Forcing Potential:** Techniques encouraging longer thought (Minimum Budget / Progressive Reasoning Expansion) show promise for improving accuracy on capable SLMs (demonstrated with Phi-4), aligning with the idea of fostering self-correction ("Aha Moments"), but depend heavily on the base model's quality.
