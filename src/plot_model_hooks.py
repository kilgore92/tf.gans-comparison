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

def plot_model_hooks(dataset,run):
    """
    Plots trend for the discriminator/critic score during inpainting
    process. Used for analyzing effect of the G-loss term in the
    inpainting loss
    +
    Gradients of loss components w.r.t z during the inpainting process

    """

    inp_dir = os.path.join(os.getcwd(),'completions',dataset.lower())
    imagesdb_dir = os.path.join(os.getcwd(),'imagesdb',dataset.lower(),'run_{}'.format(run))
    for model in models:
        print('Plotting trends for {}'.format(model))
        sys.stdout.flush()
        file_dir = os.path.join(inp_dir,model.lower(),'center')
        for batch_id in range(2):
            pkl_file_path = os.path.join(file_dir,'disc_scores_batch_{}.pkl'.format(batch_id))
            with open(pkl_file_path,'rb') as f:
                disc_scores = pickle.load(f)

            # Read in the loss gradients recorded for the contextual and perceptual losses respectively
            c_grad_file_path = os.path.join(file_dir,'c_loss_grad_batch_{}.pkl'.format(batch_id))
            with open(c_grad_file_path,'rb') as f:
                c_loss_grads = pickle.load(f)

            p_grad_file_path = os.path.join(file_dir,'p_loss_grad_batch_{}.pkl'.format(batch_id))
            with open(p_grad_file_path,'rb') as f:
                p_loss_grads = pickle.load(f)

            _,batch_size = disc_scores.shape

            print(c_loss_grads.shape)
            print(p_loss_grads.shape)
            print(disc_scores.shape)

            for idx in range(batch_size):
                file_idx = batch_id*batch_size + idx

                results_dir = os.path.join(imagesdb_dir,str(file_idx),'discriminator_scores')
                if os.path.exists(results_dir) is False:
                    os.makedirs(results_dir)

                loss_grads_dir = os.path.join(imagesdb_dir,str(file_idx),'loss_grads')
                if os.path.exists(loss_grads_dir) is False:
                    os.makedirs(loss_grads_dir)

                disc_score_list = disc_scores[:,idx]
                c_loss_grad_list = c_loss_grads[:,idx]
                p_loss_grads_list = p_loss_grads[:,idx]

                critic = ((model == 'wgan') or (model == 'wgan-gp'))
                fname = os.path.join(results_dir,'{}_disc_scores.jpg'.format(model.lower()))
                create_disc_plot(disc_score_list=disc_score_list,fname=fname,critic=critic)

                fname = os.path.join(loss_grads_dir,'{}_loss_grads.jpg'.format(model.lower()))
                create_loss_grad_plot(c_loss_grad_list,p_loss_grads_list,fname)


def create_disc_plot(disc_score_list,fname,critic=False):
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

def create_loss_grad_plot(c_loss_grad_list,p_loss_grad_list,fname):
        plt.figure(figsize=(25,15))
        plt.plot(c_loss_grad_list,'b',label='Contextual Loss Gradient')
        plt.plot(p_loss_grad_list,'g',label='Perceptual Loss Gradient')
        plt.xlabel('Optimization Iterations',fontsize=40)
        plt.ylabel('Gradient of Loss Components',fontsize=40)
        ax = plt.gca()
        ax.legend(loc='best')
        plt.tick_params(labelsize=30)

        plt.savefig(fname)
        plt.close('all')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset',help='Name of dataset on which inpainting is done',default='celeba')
    parser.add_argument('--run',help='Experiment Run',default='1')
    args = parser.parse_args()
    plot_model_hooks(dataset=args.dataset,run=args.run)


