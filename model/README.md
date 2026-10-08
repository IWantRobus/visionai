# VISIONAI - Custom Diffusion Model (DDPM), built from scratch
Owner: Tioluwalope D.O. Adesuyan

## What this is
Two diffusion models trained from raw noise, written from scratch in PyTorch (no pretrained parts):

1. MNIST model (visionai_weights.pt) - unconditional digit generator
2. CIFAR-10 model (cifar_ckpt.pt) - CLASS-CONDITIONAL generator trained on 50,000
   real internet-collected photographs. Ask it for any of: airplane, automobile,
   bird, cat, deer, dog, frog, horse, ship, truck.

## Key files
- cifar_ddpm.py - CIFAR-10 class-conditional U-Net DDPM (train + sample)
- ddpm.py - shared core: cosine noise schedule, time embeddings, residual blocks
- generate.py - sample from the MNIST model
- cifar_samples.png / samples.png - generated image grids
- cifar_training_log.txt - full overnight training log (4000 steps, self-resuming)

## Train on a GPU (recommended, ~100x faster)
Run cifar_ddpm.py in Google Colab (Runtime -> GPU) or any CUDA machine:
    python cifar_ddpm.py 60    # trains 60 min, checkpoints auto-save
    python cifar_ddpm.py sample
Load cifar_data.pt (produced by hf_dl.py) or let it read the standard CIFAR batches.

## Resume training
Both trainers auto-resume from their checkpoints. The supervisor scripts restart
training if the process dies - designed for environments that kill long jobs.
