#!/bin/bash
# Launch completions in parallel
sbatch -p all --gres=gpu:1 underfit_job_celeba dcgan &\
sbatch -p all --gres=gpu:1 underfit_job_celeba dcgan-gp &\
sbatch -p all --gres=gpu:1 underfit_job_celeba dcgan_sim &\
sbatch -p all --gres=gpu:1 underfit_job_celeba dragan &\
sbatch -p all --gres=gpu:1 underfit_job_celeba dragan_bn &\
sbatch -p all --gres=gpu:1 underfit_job_celeba dcgan-cons &\
sbatch -p all --gres=gpu:1 underfit_job_celeba wgan &\
sbatch -p all --gres=gpu:1 underfit_job_celeba wgan-gp
