import tensorflow as tf
from tqdm import tqdm
import numpy as np
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
    parser.add_argument('--model', help=models_str, required=True)
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


def analyze_vectors(model):

    """
    Returns closest training image embedding for each
    inpainted image

    """





if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    analyze_vectors(args.model)

