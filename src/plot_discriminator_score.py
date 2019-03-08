#!/usr/bin/anaconda3/bin/python3
import numpy as np
import glob, os, sys
sys.path.append(os.path.join(os.getcwd(),'src'))
from argparse import ArgumentParser
import utils, config
import shutil
import scipy.misc
import cv2
import pickle
import config
import pandas as pd
from calculate_angle_metric import read_dict,merge_and_save,central_angle_metric,read_and_crop_image
import matplotlib.pyplot as plt

models = ['dcgan','wgan','dcgan-gp','wgan-gp','dcgan-cons','dragan','dragan_bn','dcgan_sim']

def plot_discriminator_score(dataset):
    """
    Plots trend for the discriminator/critic score during inpainting
    process. Used for analyzing effect of the G-loss term in the
    inpainting loss

    """

    inp_dir = os.path.join(os.getcwd(),'completions',dataset.lower())
    for model in models:
        print('Plotting trends for {}'.format(model))
        sys.stdout.flush()
        file_dir = os.path.join(inp_dir,model.lower(),'center')
        results_dir = os.path.join(file_dir,'disc_score_trends')
        if os.path.exists(results_dir) is True:
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        for batch_id in range(2):
            pkl_file_path = os.path.join(file_dir,'disc_scores_batch_{}.pkl'.format(batch_id))
            with open(pkl_file_path,'rb') as f:
                disc_scores = pickle.load(f)

            _,batch_size = disc_scores.shape
            for idx in range(batch_size):
                file_idx = batch_id*batch_size + idx
                disc_score_list = disc_scores[:,idx]
                critic = ((model == 'wgan') or (model == 'wgan-gp'))
                create_plot(disc_score_list=disc_score_list,image_idx=file_idx,save_folder=results_dir,critic=critic)


def create_plot(disc_score_list,image_idx,save_folder,critic=False,nIter=1500):
        plt.figure(figsize=(20,10))
        plt.plot(disc_score_list)
        plt.xlabel('Optimization Iterations',fontsize=20)
        if critic == True:
            plt.ylabel('Wasserstein Distance estimated by Critic',fontsize=20)
        else:
            plt.ylabel('Real Image probability estimated by Discriminator',fontsize=20)

        if critic == False:
            plt.ylim(0,1)
        else:
            plt.ylim(0,5)

        fname=os.path.join(save_folder,'inp_trends_{}.jpg'.format(image_idx))
        plt.savefig(fname)
        plt.close('all')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset',help='Name of dataset on which inpainting is done',default='celeba')
    args = parser.parse_args()
    plot_discriminator_score(dataset=args.dataset)


