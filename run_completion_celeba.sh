#!/bin/bash
# Launch completions together
sbatch -p chromium --gres=gpu:1 complete_job_celeba dcgan &\
sbatch -p chromium --gres=gpu:1 complete_job_celeba dcgan-gp &\
sbatch -p chromium --gres=gpu:1 complete_job_celeba dcgan_sim &\
sbatch -p chromium --gres=gpu:1 complete_job_celeba dragan &\
sbatch -p chromium --gres=gpu:1 complete_job_celeba dragan_bn &\
sbatch -p chromium --gres=gpu:1 complete_job_celeba dcgan-cons &\
sbatch -p chromium --gres=gpu:1 complete_job_celeba wgan &\
sbatch -p chromium --gres=gpu:1 complete_job_celeba wgan-gp
