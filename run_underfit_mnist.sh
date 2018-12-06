#!/bin/bash
# Launch completions in parallel
sbatch -p all --gres=gpu:1 underfit_job_mnist dcgan &\
sbatch -p all --gres=gpu:1 underfit_job_mnist dcgan_sim &\
sbatch -p all --gres=gpu:1 underfit_job_mnist dcgan-gp &\
sbatch -p all --gres=gpu:1 underfit_job_mnist dragan &\
sbatch -p all --gres=gpu:1 underfit_job_mnist dragan_bn &\
sbatch -p all --gres=gpu:1 underfit_job_mnist dcgan-cons &\
sbatch -p all --gres=gpu:1 underfit_job_mnist wgan &\
sbatch -p all --gres=gpu:1 underfit_job_mnist wgan-gp
