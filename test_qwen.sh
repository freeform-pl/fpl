#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=test_qwen
#SBATCH --nodelist=iris-hgx-1,iris-hgx-2,iris5,iris6,iris9,iris10
#SBATCH --output slurm/%j.out

cd /iris/u/marcelto/reward_learning
export HOME=/iris/u/marcelto
eval "$(/iris/u/marcelto/miniconda3/bin/conda shell.bash hook)"
conda activate qwen310

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== Python: $(which python) ==="

echo "=== Testing imports ==="
python -c "
import sys, torch
print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)}', flush=True)
from transformers import AutoProcessor, AutoModelForImageTextToText
print('transformers ok', flush=True)
from peft import LoraConfig, get_peft_model
print('peft ok', flush=True)
from qwen_model import QwenRewardModel
print('QwenRewardModel import ok', flush=True)

# Quick smoke test: create model (downloads weights first time)
print('Creating QwenRewardModel (num_preferences=8, max_frames=4)...', flush=True)
model = QwenRewardModel(num_preferences=8, max_frames=4, use_lora=False).cuda()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model created. Trainable params: {n_params:,}', flush=True)

# Test forward pass with dummy data
print('Testing forward pass...', flush=True)
B, T, C, H, W = 1, 10, 3, 128, 128
tp = torch.randint(0, 255, (B, T, C, H, W), dtype=torch.uint8).cuda()
wr = torch.randint(0, 255, (B, T, C, H, W), dtype=torch.uint8).cuda()
mask = torch.zeros(B, T, dtype=torch.bool).cuda()
rewards = model(tp, wr, mask)
print(f'Forward pass OK. rewards shape={rewards.shape}, values={rewards.detach().cpu().numpy()}', flush=True)

# Test backward pass
print('Testing backward pass...', flush=True)
loss = rewards.sum()
loss.backward()
print('Backward pass OK', flush=True)

# Test LoRA variant
print('Creating QwenRewardModel with LoRA...', flush=True)
del model
torch.cuda.empty_cache()
model_lora = QwenRewardModel(num_preferences=8, max_frames=4, use_lora=True).cuda()
n_trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model_lora.parameters())
print(f'LoRA model. Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)', flush=True)

rewards_lora = model_lora(tp, wr, mask)
print(f'LoRA forward OK. rewards shape={rewards_lora.shape}', flush=True)
rewards_lora.sum().backward()
print('LoRA backward OK', flush=True)

# Test checkpoint save/load
print('Testing checkpoint save/load...', flush=True)
sd = model_lora.get_checkpoint_state_dict()
print(f'Checkpoint state dict keys: {len(sd)} (should be small for LoRA)', flush=True)

print('\\n=== ALL TESTS PASSED ===', flush=True)
"
echo "=== Test script done ==="
