import tensorflow as tf
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cosine
import glob, os, sys
from argparse import ArgumentParser
import utils, config
import shutil
import scipy.misc
from eval import get_all_checkpoints
from convert import center_crop
import cv2
import pickle

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', help='Provide model for which analysis should be performed. eg: DCGAN/WGAN/...', required=True)
    return parser



def find_closest_vector(z_inpainting,z_train_dict):
    """
    Finds closest z-vector from the training data
    for a given z_inapainting vector
    Args : z_inpainting : z-vector for an in-painted image
           z_train_dict : Dictionary containing training_image:z mapping

    Returns : cosine similarity between z_inpainting and z_train_closest
              z_train_closest

    """
    min_cosine = 5.0
    min_training_image = ''
    for t_image_path,z_training in z_train_dict.items():
        cosine_distance = cosine(z_inpainting,z_training)
        if cosine_distance < min_cosine:
            min_cosine = cosine_distance
            min_training_image = t_image_path

    return min_training_image,min_cosine

def analyze_vectors(model):

    """
    Returns closest training image embedding for each
    inpainted image

    """
    with open('latent_space_inpaint_{}.pkl'.format(model.upper()),'rb') as f:
        z_inpainting_dict = pickle.load(f)

    with open('latent_space_test_{}.pkl'.format(model.upper()),'rb') as f:
        z_test_dict = pickle.load(f)

    with open('latent_space_test_{}.pkl'.format(model.upper()),'rb') as f:
        z_train_dict = pickle.load(f)

    recall = []
    generalize = []

    for test_image_path,z_inpainting in z_inpainting_dict.items():
        row = []
        z_test = z_test_dict[test_image_path]
        test_cosine = cosine(z_inpainting,z_test)
        train_image_min,train_cosine = find_closest_vector(z_inpainting,z_train_dict)
        row.append(test_image_path)
        row.append(train_image_min)
        row.append(test_cosine)
        row.append(train_cosine)
        print('Analysis done for image : {} train_cosine : {} test_cosine : {}'.format(test_image_path,train_cosine,test_cosine))
        if test_cosine < train_cosine: # Inpainting latent vector closer to test latent vector
            generalize.append(row)
        else: # Inpainting latent vector closer to training image latent vector
            recall.append(row)

    print('Generalized inpaintings : {}'.format(len(generalize)))
    print('Recalled inpaintings : {}'.format(len(recall)))




if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    analyze_vectors(args.model)

