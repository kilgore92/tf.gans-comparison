#!/usr/bin/anaconda3/bin/python3
# coding: utf-8
import numpy as np
import os
import argparse
import shutil
import random
import sys

def create_database():

    """
    Reads the files created by the reconstruction API to create a flat hierarchy
    Each folder generated contains one original image + images in-painted by different models

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--outDir',type=str,default = 'database')
    parser.add_argument('--rootDir',type=str,default='completions_stochastic_center')
    parser.add_argument('--dataset',type=str,default = 'celeba')
    parser.add_argument('--nImages',type=int,default=1000)
    args = parser.parse_args()

    if (os.path.exists(args.outDir)):
        # Remove it
        shutil.rmtree(args.outDir)

    os.makedirs(args.outDir)

    models = ['dcgan','wgan','dcgan-gp','wgan-gp','dcgan-cons','dragan']
    source_dirs = []
    for model in models:
        dir_path = os.path.join(os.getcwd(),str(args.rootDir),str(model),str(args.dataset))
        source_dirs.append(dir_path)


    for idx in range(args.nImages):
        curr_out_dir = os.path.join(args.outDir,'{}'.format(idx))
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
            curr_image_file = os.path.join(source_dir,'{}'.format(idx),'gen_images','gen_1400.jpg')
            model_name = source_dir.split('/')[-2]
            dst = os.path.join(curr_out_dir,'gen','{}.jpg'.format(model_name))
            shutil.copy2(curr_image_file,dst)


def sample_test_files(dir_path,n_samples=1000):
    """
    Given a director path, get a random sample of files
    of size n_samples in a list
    """
    files = os.listdir(dir_path)
    complete_paths = [os.path.join(dir_path,f) for f in files]
    test_samples =  random.sample(complete_paths,n_samples)
    return test_samples




def split_train_test():
    """
    Split train and test set

    """
    base_dir = '/mnt/server-home/TUE/s162156/datasets/celebA'
    train_file_path = os.path.join(base_dir,'celebA')

    test_samples = sample_test_files(train_file_path)
    target_dir = os.path.join(base_dir,'celebA_test')

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    os.makedirs(target_dir)

    for f in test_samples:
        fname = f.split('/')[-1]
        new_path = os.path.join(target_dir,fname)
        shutil.move(f,new_path)


if __name__ == '__main__':
    split_train_test()



