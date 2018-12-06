#!/usr/bin/anaconda3/bin/python3
# coding: utf-8
import numpy as np
import os
from argparse import ArgumentParser
import shutil
import random
import sys

def build_parser():

    """
    Reads the files created by the reconstruction API to create a flat hierarchy
    Each folder generated contains one original image + images in-painted by different models

    """
    parser = ArgumentParser()
    parser.add_argument('--dataset',type=str,default = 'celeba')
    parser.add_argument('--nImages',type=int,default=1000)
    args = parser.parse_args()
    return args


def sample_test_files(dir_path,n_samples=1000):
    """
    Given a director path, get a random sample of files
    of size n_samples in a list
    """
    files = os.listdir(dir_path)
    complete_paths = [os.path.join(dir_path,f) for f in files]
    test_samples =  random.sample(complete_paths,n_samples)
    return test_samples




def split_train_test(args):
    """
    Split train and test set

    """
    train_file_path = os.path.join(os.path.expanduser('~'),'datasets','{}'.format(args.dataset),'{}_train'.format(args.dataset))

    target_dir = os.path.join(os.path.expanduser('~'),'datasets','{}'.format(args.dataset),'{}_test'.format(args.dataset))

    if os.path.exists(target_dir):

        # Copy back the files into the train dir before removing this
        complete_paths = [os.path.join(target_dir,f) for f in os.listdir(target_dir)]
        for f in complete_paths:
            new_path = os.path.join(train_file_path,f.split('/')[-1])
            shutil.move(f,new_path)

        # remove the test directory
        shutil.rmtree(target_dir)

    os.makedirs(target_dir)

    # Sample from 'full' dataset
    test_samples = sample_test_files(train_file_path)

    for f in test_samples:
        fname = f.split('/')[-1]
        new_path = os.path.join(target_dir,fname)
        shutil.move(f,new_path)



if __name__ == '__main__':

    args = build_parser()
    split_train_test(args)



