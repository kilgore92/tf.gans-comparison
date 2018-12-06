#!/bin/bash
sbatch -p chromium --gres=gpu:1 train.py --model dcgan --dataset mnist --renew & \
sbatch -p chromium --gres=gpu:1 train.py --model dcgan-gp --dataset mnist --renew & \
sbatch -p chromium --gres=gpu:1 train.py --model dcgan_sim --dataset mnist --renew --simultaneous & \
sbatch -p chromium --gres=gpu:1 train.py --model dragan --dataset mnist --renew & \
sbatch -p chromium --gres=gpu:1 train.py --model dragan_bn --dataset mnist --renew & \
sbatch -p chromium --gres=gpu:1 train.py --model dcgan-cons --dataset mnist --renew --simultaneous & \
sbatch -p chromium --gres=gpu:1 train.py --model wgan --dataset mnist --renew & \
sbatch -p chromium --gres=gpu:1 train.py --model wgan-gp --dataset mnist --renew
