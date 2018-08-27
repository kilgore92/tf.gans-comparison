#!/bin/bash
sbatch -p all --gres=gpu:1 train.py --model dcgan --dataset celeba --renew & \
sbatch -p all --gres=gpu:1 train.py --model dcgan-gp --dataset celeba --renew & \
sbatch -p all --gres=gpu:1 train.py --model dcgan_sim --dataset celeba --renew --simultaneous & \
sbatch -p all --gres=gpu:1 train.py --model dragan --dataset celeba --renew & \
sbatch -p all --gres=gpu:1 train.py --model dragan_bn --dataset celeba --renew & \
sbatch -p all --gres=gpu:1 train.py --model dcgan-cons --dataset celeba --renew --simultaneous & \
sbatch -p all --gres=gpu:1 train.py --model wgan --dataset celeba --renew & \
sbatch -p all --gres=gpu:1 train.py --model wgan-gp --dataset celeba --renew
