#!/usr/bin/anaconda3/bin/python3
# coding: utf-8
import numpy as np
import os
from argparse import ArgumentParser
import shutil
import random
import sys

models = ['dcgan','wgan','dcgan-gp','wgan-gp','dcgan-cons','dragan','dragan_bn','dcgan_sim']

def build_parser():

    """
    Reads the files created by the reconstruction API to create a flat hierarchy
    Each folder generated contains one original image + images in-painted by different models

    """
    parser = ArgumentParser()
    parser.add_argument('--outDir',type=str,default = 'imagesdb')
    parser.add_argument('--rootDir',type=str,default='completions')
    parser.add_argument('--dataset',type=str,required=True)
    parser.add_argument('--nImages',type=int,default=1000)
    parser.add_argument('--mask',type=str,default='center')
    args = parser.parse_args()
    return args

def create_database(args):
    """
    Merge in-paintings from different models into a hierarchical folder
    structure

    """
    outDir = os.path.join(args.outDir,args.dataset.lower())

    if (os.path.exists(outDir)):
        # Remove it
        shutil.rmtree(outDir)

    os.makedirs(outDir)

    source_dirs = []
    for model in models:
        dir_path = os.path.join(os.getcwd(),str(args.rootDir),str(args.dataset.lower()),model,str(args.mask))
        source_dirs.append(dir_path)

    if args.dataset == 'celeba':
        src_gen_folder = 'gen_images'
    else:
        src_gen_folder = 'gen_images_overlay' # Poisson Blending not possible for single channel images FIXME

    for idx in range(args.nImages):
        curr_out_dir = os.path.join(outDir,'{}'.format(idx))
        os.makedirs(curr_out_dir)
        original_image = os.path.join(source_dirs[0],'{}'.format(idx),'original.jpg') # Copy the image from one of the source directories
        # Copy over the original image
        shutil.copy2(original_image,curr_out_dir)
        # Copy over the masked image
        masked_image = os.path.join(source_dirs[0],'{}'.format(idx),'masked.jpg') # Copy the image from one of the source directories
        shutil.copy2(masked_image,curr_out_dir)

        # Make the sub-folder
        genDir = os.path.join(curr_out_dir,'gen')
        os.makedirs(genDir)

        for source_dir in source_dirs:
            curr_image_file = os.path.join(source_dir,'{}'.format(idx),src_gen_folder,'gen_1400.jpg')
            model_name = source_dir.split('/')[-2]
            dst = os.path.join(curr_out_dir,'gen','{}.jpg'.format(model_name))
            shutil.copy2(curr_image_file,dst)

if __name__ == '__main__':

    args = build_parser()
    create_database(args)
