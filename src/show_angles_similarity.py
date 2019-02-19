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

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset',type=str,help='Dataset to analyze',default='celeba')
    parser.add_argument('--emb',type=str,help='Root dir where the embedding dictionaries are saved',default='/home/TUE/s162156/facenet/facenet/embeddings')
    return parser

def find_training_image(emb_test,training_emb_dict,target_angle=60):
    """
    Given the test embedding, returns the training image
    whose embedding is at the specified image w.r.t test embedding

    Returns the path to the first image found+number of training images at that angle

    """
    example_training_image_path = None
    num_training_images = 0
    for t_image_path,emb_training in training_emb_dict.items():
        flag=False
        angle = central_angle_metric(emb_test,emb_training)
        if angle < target_angle:
            if flag is False:
                example_training_image_path = t_image_path
                flag = True
            num_training_images += 1
    return example_training_image_path,num_training_images

def calculate_angles_test_train(emb_test,training_emb_dict):
    """
    Create a list with all distances of each test image to all training images
    to get an idea of the range

    """
    angles_list = []
    for t_image_path,emb_training in training_emb_dict.items():
        angles_list.append(central_angle_metric(emb_test,emb_training))

    return angles_list

def create_angles_histogram(args):
    """
    Creates histogram of train image angles w.r.t each test image

    """

    training_emb_dict,test_emb_dict,_ = read_dict(root_dir = args.emb,model=None,dataset=args.dataset)
    out_dir = os.path.join(os.getcwd(),'angle_metric_histogram')

    if os.path.exists(out_dir) is True:
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for test_image_path,emb_test in test_emb_dict.items():
        angles_list = calculate_angles_test_train(emb_test,training_emb_dict)
        test_image_idx = test_image_path.split('/')[-2]
        fname = os.path.join(out_dir,'hist_img_{}.jpg'.format(test_image_idx))
        plt.figure()
        plt.hist(x=np.asarray(angles_list),bins=18)
        plt.savefig(fname)
        plt.close('all')
        print('Histogram plotted for test image {}'.format(test_image_idx))
        sys.stdout.flush()



def show_images_angle_similarity(args):
    """
    Displays Test Image + 30 deg example + 60 deg example + 90 deg example + 120 def example + 150 deg example + 180 def example

    """
    training_emb_dict,test_emb_dict,_ = read_dict(root_dir = args.emb,model=None,dataset=args.dataset)
    train_images_dict = {} # Indexed by test image path, value is a list [#images at 30, at 60 and so on...]

    out_dir = os.path.join(os.getcwd(),'angle_metric_sanity')

    if os.path.exists(out_dir) is True:
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for test_image_path,emb_test in test_emb_dict.items():
        image_list = []
        train_image_hist = [] # [0] : #training images at 30 degrees, [1] :# at 60 and so on ...
        image_list.append(read_and_crop_image(test_image_path,args.dataset)) # Add test image

        example_training_path,num_training_images = find_training_image(emb_test=emb_test,training_emb_dict=training_emb_dict,target_angle=60)
        print('For {} :: {} training images found at 60 degrees'.format(test_image_path,num_training_images))
        sys.stdout.flush()
        if example_training_path is not None:
            image_list.append(read_and_crop_image(example_training_path,args.dataset))

        train_images_dict[test_image_path] = num_training_images

        #Save the array of images
        test_image_idx = test_image_path.split('/')[-2]
        merge_and_save(image_list=image_list,idx=test_image_idx,root_dir=out_dir,dataset=args.dataset)

    #Save the dict

    with open(os.path.join(out_dir,'test_image_angle_hist.pkl'),'wb') as f:
        pickle.dump(train_images_dict,f)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    show_images_angle_similarity(args)
    #create_angles_histogram(args)


