#!/usr/bin/anaconda3/bin/python3
import numpy as np
import glob, os, sys
sys.path.append(os.path.join(os.getcwd(),'src'))
from argparse import ArgumentParser
import utils, config
import shutil
import pickle
import config
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

models = ['dcgan','wgan','dcgan-gp','wgan-gp','dcgan-cons','dragan','dragan_bn','dcgan_sim']

def plot_discriminator_score(dataset):
    """
    Plots trend for the discriminator/critic score during inpainting
    process. Used for analyzing effect of the G-loss term in the
    inpainting loss

    """

    inp_dir = os.path.join(os.getcwd(),'completions',dataset.lower())
    imagesdb_dir = os.path.join(os.getcwd(),'imagesdb',dataset.lower())
    for model in models:
        print('Plotting trends for {}'.format(model))
        sys.stdout.flush()
        file_dir = os.path.join(inp_dir,model.lower(),'center')
        for batch_id in range(2):
            pkl_file_path = os.path.join(file_dir,'disc_scores_batch_{}.pkl'.format(batch_id))
            with open(pkl_file_path,'rb') as f:
                disc_scores = pickle.load(f)

            _,batch_size = disc_scores.shape
            for idx in range(batch_size):
                file_idx = batch_id*batch_size + idx
                results_dir = os.path.join(imagesdb_dir,str(file_idx),'discriminator_scores')
                if os.path.exists(results_dir) is False:
                    os.makedirs(results_dir)
                disc_score_list = disc_scores[:,idx]
                critic = ((model == 'wgan') or (model == 'wgan-gp'))
                fname = os.path.join(results_dir,'{}_disc_scores.jpg'.format(model.lower()))
                create_plot(disc_score_list=disc_score_list,image_idx=file_idx,fname=fname,critic=critic)


def create_plot(disc_score_list,image_idx,fname,critic=False,nIter=1500):
        plt.figure(figsize=(25,15))
        plt.plot(disc_score_list)
        plt.xlabel('Optimization Iterations',fontsize=40)
        if critic == True:
            plt.ylabel('Wasserstein Distance',fontsize=40)
        else:
            plt.ylabel('Discriminator Score',fontsize=40)

        if critic == False:
            plt.ylim(0,1)
        else:
            plt.ylim(0,5)
        plt.tick_params(labelsize=30)

        plt.savefig(fname)
        plt.close('all')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset',help='Name of dataset on which inpainting is done',default='celeba')
    args = parser.parse_args()
    plot_discriminator_score(dataset=args.dataset)


