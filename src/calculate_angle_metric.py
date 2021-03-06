#!/usr/bin/anaconda3/bin/python3
import numpy as np
from scipy.spatial.distance import cosine
import glob, os, sys
sys.path.append(os.path.join(os.getcwd(),'src'))
from argparse import ArgumentParser
import utils, config
import shutil
import scipy.misc
from convert import center_crop
import cv2
import pickle
import config
from complete import rescale_image
from complete import blend_images
import pandas as pd
from math import degrees


image_size = 64


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,help='Provide model for which analysis should be performed. eg: DCGAN/WGAN/...', required=True)
    parser.add_argument('--dataset',type=str,help='Dataset to analyze',default='celeba')
    parser.add_argument('--emb',type=str,help='Root dir where the embedding dictionaries are saved',default='/home/TUE/s162156/facenet/facenet/embeddings')
    parser.add_argument('--mask',type=str,help='Inpainting mask',default='center')
    return parser



def find_closest_training_image(emb_inpainting,train_emb_dict):
    """
    Finds closest image from training set w.r.t G(z_inpainting) based on the learned metric

    Args : emb_inpainting : Embedding for an in-painted image
           train_emb_dict : Dictionary containing training_image:embedding mapping

    Returns : minimum value of cosine,training image path for the minimum value


    """
    min_central_angle = 180.0
    min_training_image_path = ''
    for t_image_path,emb_training in train_emb_dict.items():
        central_angle = central_angle_metric(emb_inpainting,emb_training)

        if central_angle < 0: #Numerical error sometimes results in the cosine being slightly larger than 1, leading to negative "distance"
            cosine_distance = 0

        if central_angle < min_central_angle:
            min_central_angle = central_angle
            min_training_image_path = t_image_path
            min_training_image_emb = emb_training

    return min_training_image_path,min_central_angle,min_training_image_emb

def find_avg_test_image_angle(emb_test,train_emb_dict):
    """
    Finds average distance for a given test image w.r.t the training
    dataset

    """
    avg_angle = 0.0
    for t_image_path,emb_training in train_emb_dict.items():
        central_angle = central_angle_metric(emb_test,emb_training)
        avg_angle += central_angle
        print(central_angle)

    avg_angle = avg_angle/len(train_emb_dict)
    return avg_angle



def central_angle_metric(u,v,r=1.0):
    """
    For 2 vectors u,v lying on a (hyper-)sphere of radius r,
    this function returns the smallest central angle [0,180] (in degrees) between them

    This function has been copied over from ~/facenet/facenet/src/gans_compare.py -- FIXME

    """
    dot_product = np.dot(u,v)
    central_angle = np.arccos(np.divide(dot_product,r*r))
    if central_angle <= np.pi:
        return degrees(central_angle)
    else:
        return degrees(2*np.pi - central_angle)


def merge_and_save(image_list,idx,root_dir,dataset='mnist'):

    """
    Create an image mosiac.
    Order (L-R) : Test Image - Inpainting - G(z_inpainting) - Closest Training Image

    """

    filename = os.path.join(root_dir,'3z_{}.jpg'.format(idx))
    frame_width = int(64*len(image_list))
    frame_height = 64
    if dataset == 'celeba':
        frame_channels = 3
    else:
        frame_channels = 1

    if frame_channels == 3:
        img = np.zeros((frame_height,frame_width,frame_channels))

        for image,index in zip(image_list,range(len(image_list))):
            x_pos = index*64
            img[0:int(frame_height),x_pos:x_pos+64,:] = image
    else:
        img = np.zeros((frame_height,frame_width))

        for image,index in zip(image_list,range(len(image_list))):
            x_pos = index*64
            img[0:int(frame_height),x_pos:x_pos+64] = image


    scipy.misc.imsave(filename,img)



def read_and_crop_image(fname,dataset='celeba'):
    """
    Read and crop image to 64x64 for display

    """
    if dataset =='celeba':
        return center_crop(im = scipy.misc.imread(fname,mode='RGB'),output_size=[64,64])
    else:
        return scipy.misc.imresize(arr=scipy.misc.imread(fname),size = (64,64),interp='bilinear')

def read_dict(root_dir,model,dataset='celeba'):
    """
    Reads all three embedding dictionaries

    """
    root_dir = os.path.join(root_dir,dataset)
    with open(os.path.join(root_dir,'train_{}_emb_dict.pkl'.format(dataset.lower())),'rb') as f:
        train_emb_dict = pickle.load(f)

    with open(os.path.join(root_dir,'test_{}_emb_dict.pkl'.format(dataset.lower())),'rb') as f:
        test_emb_dict = pickle.load(f)

    if model is not None:
        fname = os.path.join(root_dir,'{}_emb_dict.pkl'.format(model.lower()))
        with open(fname,'rb') as f:
            inpaint_emb_dict = pickle.load(f)
    else:
        inpaint_emb_dict = None

    return train_emb_dict,test_emb_dict,inpaint_emb_dict

def get_test_image_key(gz_path):

    """
    Given path to G(z_inpainting), return the dict key
    to fetch the embeddings for the corr. test image

    """

    idx = gz_path.split('/')[-3]
    dataset = gz_path.split('/')[-6]
    test_img_key = os.path.join('imagesdb',str(dataset),str(idx),'original.jpg')
    return test_img_key



def analyze_vectors(args):

    """

    Performs an analysis of the embeddings for the train, test and G(z_inpainting)

    """

    train_emb_dict,test_emb_dict,inpaint_emb_dict = read_dict(args.emb,args.model,args.dataset)


    outDir = os.path.join(os.getcwd(),'embeddings',args.dataset.lower(),args.model.lower())

    if os.path.exists(outDir) is True:
        shutil.rmtree(outDir)

    os.makedirs(outDir)

    # Make 2 buckets -- recalled and generalized
    recall_dir = os.path.join(outDir,'recall')
    generalized_dir = os.path.join(outDir,'generalized')
    os.makedirs(recall_dir)
    os.makedirs(generalized_dir)

    image_idx = 0
    pseudo_df = []

    closest_train_inp_image_dict = {}
    closest_train_test_image_dict = {}

    # Path for inpainted images (with blending bug fixed)
    inpaint_path_root = os.path.join(os.getcwd(),'completions',args.dataset.lower(),args.model.lower(),args.mask)

    # Iterate over the dict
    for gz_path,emb_inpainting in inpaint_emb_dict.items():
        generalized_inp = 0
        row = []
        image_list = []

        image_idx = gz_path.split('/')[-3]
        # From the inpaint file, fetch corr. source image
        test_img_key = get_test_image_key(gz_path)
        # Using the source image path, perform a look up for the embedding
        emb_test = test_emb_dict[test_img_key]
        # Cosine between source image and G(z_inpainting)
        test_inp_angle = central_angle_metric(emb_inpainting,emb_test)
        # Find the closest training image in the embedding space w.r.t G(z)
        t_image_min_path_inp, train_inp_min_angle,train_inp_min_emb = find_closest_training_image(emb_inpainting,train_emb_dict)
        # Find the closest trainig image in the embedding space w.r.t test image
        avg_test_angle = find_avg_test_image_angle(emb_test,train_emb_dict)

        # Maintain "closest" training images dictionary
        closest_train_inp_image_dict[t_image_min_path_inp] = train_inp_min_emb

        row.append(os.path.join(os.getcwd(),test_img_key))
        row.append(os.path.join(os.getcwd(),gz_path))
        row.append(t_image_min_path_inp)
        row.append(test_inp_angle)
        row.append(train_inp_min_angle)
        row.append(avg_test_angle)

        print('Analysis done for image : {} train_inp_angle : {} avg_test_angle : {} test_inp_angle : {}'.format(test_img_key,train_inp_min_angle,avg_test_angle,test_inp_angle))

        sys.stdout.flush()

        testImg = read_and_crop_image(os.path.join(os.getcwd(),test_img_key),args.dataset)
        maskImg = read_and_crop_image(os.path.join(inpaint_path_root,str(image_idx),'masked.jpg'),args.dataset)

        # Inpaintings -- Buggy - Blended - Overlay
        if args.dataset == 'celeba':
            inpImg_blend = read_and_crop_image(os.path.join(inpaint_path_root,str(image_idx),'gen_images','gen_1400.jpg'))

        inpImg_overlay = read_and_crop_image(os.path.join(inpaint_path_root,str(image_idx),'gen_images_overlay','gen_1400.jpg'),args.dataset)

        # G(z)
        Gz = read_and_crop_image(os.path.join(inpaint_path_root,str(image_idx),'gz','gz_1400.jpg'),args.dataset)

        image_list.append(testImg) # Test Image -- Complete
        image_list.append(maskImg) # Masked
        if args.dataset == 'celeba':
            image_list.append(inpImg_blend) # Inpainting -- blend
        image_list.append(inpImg_overlay) # Inpainting -- overlay
        image_list.append(Gz) # G(z_inpainting)
        image_list.append(read_and_crop_image(t_image_min_path_inp,args.dataset)) # Closest training image w.r.t emb_inpainting

        pseudo_df.append(row)

        if test_inp_angle <= train_inp_min_angle: # Inpainting latent vector closer to test latent vector
            merge_and_save(image_list=image_list,idx=image_idx,root_dir = generalized_dir,dataset=args.dataset)
            generalized_inp =+ 1
        else: # Inpainting latent vector closer to training image latent vector
            merge_and_save(image_list=image_list,idx=image_idx,root_dir = recall_dir,dataset=args.dataset)

    print('Generalized inpaintings : {}'.format(generalized_inp))

    df_results = pd.DataFrame(data=np.asarray(pseudo_df),columns=['Source Image Path','Gz Path','Closest Train Image Inpainting','Test-Gz Cosine','Train-Gz Cosine','Avg Test Angle'])
    df_results.to_csv(os.path.join(outDir,'emb_results.csv'))


    # Save the dict
    with open(os.path.join(outDir,'closest_train_inp_emb.pkl'),'wb') as f:
        pickle.dump(closest_train_inp_image_dict,f)

def generate_inpainting(testImg,mask,Gz):
    """
    Given test image,G(z_inpainting) and mask
    return the inpainting

    """
    inpainting = blend_images(image=testImg,gen_image=Gz,mask=np.multiply(255,1.0-mask),rescale=False)
    return inpainting



if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()
    analyze_vectors(args)

