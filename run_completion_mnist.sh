#!/bin/bash
# Launch completions together
sbatch -p chromium --gres=gpu:1 complete_job_mnist dcgan &\
sbatch -p chromium --gres=gpu:1 complete_job_mnist dcgan-gp &\
sbatch -p chromium --gres=gpu:1 complete_job_mnist dcgan_sim &\
sbatch -p calcium --gres=gpu:1 complete_job_mnist dragan &\
sbatch -p calcium --gres=gpu:1 complete_job_mnist dragan_bn &\
sbatch -p calcium --gres=gpu:1 complete_job_mnist dcgan-cons &\
sbatch -p calcium --gres=gpu:1 complete_job_mnist wgan &\
sbatch -p calcium --gres=gpu:1 complete_job_mnist wgan-gp
