from argparse import ArgumentParser
import scipy.misc
import numpy as np
import pandas as pd
import pickle
import sys
import os
sys.path.append(os.getcwd())
from underfit import read_dict,find_closest_training_image
from convert import center_crop


test_image_root = '/home/TUE/s162156/gans_compare/tf.gans-comparison/imagesdb'
crop_l = 16
crop_u = 48

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--emb',type=str,help='Root dir where the embedding dictionaries are saved',default='/home/ibhat/facenet/facenet/embeddings')
    return parser

def synthetic_exp(emb_path):
    """
    1. Overfitting GAN : Inpainting produced using nearest neighbour of the test image in the training data
    2. Generalizing GAN : Inpainting produced is the test image => Inn : Nearest neighbour to test image

    We uses the find_closest_training_image() function to find nearest neighbour, the only difference between the 2 experiments is
    a sign flip on the generalization gap metric

    """
    train_emb_dict,test_emb_dict,_ = read_dict(root_dir=emb_path,model=None)

    nn_distances_overfit = []
    nn_distances_gen = []
    for path,emb in test_emb_dict.items():
       path_min_train,min_cosine,_ = find_closest_training_image(emb_inpainting=emb,train_emb_dict = train_emb_dict)
       print('For test image : {}, closest train image : {}. Cosine = {}'.format(path,path_min_train,min_cosine))
       sys.stdout.flush()
       nn_distances_overfit.append(min_cosine)
       nn_distances_gen.append(-min_cosine)
       # Create the synthetic inpainting for the overfitting GAN
       create_synthetic_inpainting(test_path=path,train_path=path_min_train)


    # Create pandas dataframe
    col_stacked_data = np.column_stack((np.asarray(nn_distances_overfit),np.asarray(nn_distances_gen)))
    df = pd.Dataframe(data=col_stacked_data,columns = ['Overfitting','Generalizing'])
    df.to_pickle(path='synthetic.pkl')




def create_synthetic_inpainting(test_path,train_path):
    """
    Read test-image and train image
    Replace center pixels of test with that of nearest neighbour from train set

    """
    train_img = scipy.misc.imread(train_path)
    train_crop = center_crop(train_img,output_size=[64,64])
    test_img = scipy.misc.imread(test_path)
    test_img[crop_l:crop_u,crop_l:crop_u,:] = train_crop[crop_l:crop_u,crop_l:crop_u,:]
    test_image_id = test_path.split('/')[-2]
    save_path = os.path.join(test_image_root,test_image_id,'gen','syn.jpg')
    scipy.misc.imsave(save_path,test_img)

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    synthetic_exp(emb_path = args.emb)








