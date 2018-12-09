#!/bin/bash
# Launch completions in parallel
sbatch -p chromium underfit_job_mnist dcgan &\
sbatch -p chromium underfit_job_mnist dcgan_sim &\
sbatch -p chromium underfit_job_mnist dcgan-gp &\
sbatch -p chromium underfit_job_mnist dragan &\
sbatch -p chromium underfit_job_mnist dragan_bn &\
sbatch -p chromium underfit_job_mnist dcgan-cons &\
sbatch -p chromium underfit_job_mnist wgan &\
sbatch -p chromium underfit_job_mnist wgan-gp
