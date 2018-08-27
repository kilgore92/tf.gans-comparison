#!/bin/bash
# Launch completions in parallel
sbatch -p all --gres=gpu:1 complete_job dcgan &\
sbatch -p all --gres=gpu:1 complete_job dcgan-gp &\
sbatch -p all --gres=gpu:1 complete_job dcgan_sim &\
sbatch -p all --gres=gpu:1 complete_job dragan &\
sbatch -p all --gres=gpu:1 complete_job dragan_bn &\
sbatch -p all --gres=gpu:1 complete_job dcgan-cons &\
sbatch -p all --gres=gpu:1 complete_job wgan &\
sbatch -p all --gres=gpu:1 complete_job wgan-gp
