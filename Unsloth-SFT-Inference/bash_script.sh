#!/bin/bash

python unsloth_sft_inference_1e4.py > SFT_Infer_5Epoch_1e_4.txt 2>&1
python unsloth_sft_inference_1e5.py > SFT_Infer_5Epoch_1e_5.txt 2>&1
python unsloth_sft_inference_1e6.py > SFT_Infer_5Epoch_1e_6.txt 2>&1
python unsloth_sft_inference_2e4.py > SFT_Infer_5Epoch_2e_4.txt 2>&1
python unsloth_sft_inference_5e6.py > SFT_Infer_5Epoch_5e_6.txt 2>&1
