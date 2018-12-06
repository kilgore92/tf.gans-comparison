import numpy as np
import pickle
from pylab import *
import shutil
import os
from argparse import ArgumentParser

def plot_grads(f,model,dataset):

    """
    f : Dictionary containing gradient values at different iterations
        for generator and discriminator

    """
    disc_grads = np.asarray(f['Discrminator'],dtype=np.float32) # 'Discriminator' is mis-spelt due to a typo in the training script
    gen_grads = np.asarray(f['Generator'],dtype=np.float32)

    x_axis = [i*10 for i in range(len(disc_grads))] # X-axis containing iteration number

    plot(x_axis, disc_grads, color='red', linewidth=2.5, linestyle='--', label='Discriminator Gradients')
    plot(x_axis, gen_grads, color='blue', linewidth=2.5, linestyle='-', label='Generator Gradients')

    legend()
    if os.path.exists('grad_figures') is False:
        os.makedirs('grad_figures')

    filename = 'grads_' + str(dataset) + '_' + str(model) +'.png'
    save_path = os.path.join('grad_figures',filename)
    savefig(save_path)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='Name of the GAN model', required='true')
    parser.add_argument('--dataset', type=str, help='Name of dataset',default='celeba')

    args = parser.parse_args()

    pkl_path = os.path.join(os.getcwd(),'checkpoints',args.dataset,args.model.lower(),'grads.pkl')

    try:
        with open(pkl_path,'rb') as f:
            grads = pickle.load(f)
    except:
        print('File not found')

    plot_grads(f=grads,model=args.model.lower(),dataset=args.dataset)








