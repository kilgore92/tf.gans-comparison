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
import config
from complete import rescale_image
from complete import blend_images


image_size = 64

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,help='Provide model for which analysis should be performed. eg: DCGAN/WGAN/...', required=True)
    parser.add_argument('--dataset',type=str,help='Dataset to analyze',default='celeba')
    parser.add_argument('--gpu',type=str,help='GPU to select,valid options 0 or 1',default="1")
    return parser



def find_closest_vector(z_inpainting,z_train_dict):
    """
    Finds closest z-vector from the training data
    for a given z_inpainting vector
    Args : z_inpainting : z-vector for an in-painted image
           z_train_dict : Dictionary containing training_image:z mapping

    Returns : cosine similarity between z_inpainting and z_train_closest
              z_train_closest

    """
    min_cosine = 5.0
    min_training_image_path = ''
    for t_image_path,z_training in z_train_dict.items():
        cosine_distance = cosine(z_inpainting,z_training)
        if cosine_distance < min_cosine:
            min_cosine = cosine_distance
            min_training_image_path = t_image_path
            min_train_z = z_training

    return min_training_image_path,min_train_z,min_cosine

def merge_and_save(image_list,idx,root_dir):

    """
    Create an image mosiac.
    Order (L-R) : Test Image - Inpainting - Closest Training Image - G(z_inpainting) - G(z_test) - G(min_train_z)

    """

    filename = os.path.join(root_dir,'3z_{}.jpg'.format(idx))
    frame_width = int(64*len(image_list))
    frame_height = 64
    frame_channels = 3
    img = np.zeros((frame_height,frame_width,frame_channels))

    for image,index in zip(image_list,range(len(image_list))):
        x_pos = index*64
        img[0:int(frame_height),x_pos:x_pos+64,:] = image

    scipy.misc.imsave(filename,img)


def generate_image(model,sess,z):
    """
    For a given z, return G(z)

    """
    g_img = sess.run(model.fake_sample,feed_dict = {model.z : z})
    return rescale_image(g_img[0])


def read_and_crop_image(fname):
    """
    Read and crop image to 64x64 for display

    """
    return center_crop(im = scipy.misc.imread(fname,mode='RGB'),output_size=[64,64])

def analyze_vectors(args):

    """
    Returns closest training image embedding for each
    inpainted image

    """
    with open('latent_space_inpaint_{}.pkl'.format(args.model.upper()),'rb') as f:
        z_inpainting_dict = pickle.load(f)

    with open('latent_space_test_{}.pkl'.format(args.model.upper()),'rb') as f:
        z_test_dict = pickle.load(f)

    with open('latent_space_train_{}.pkl'.format(args.model.upper()),'rb') as f:
        z_train_dict = pickle.load(f)

    root_dir = os.path.join(os.getcwd(),'3z_{}'.format(args.model))
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)

    os.makedirs(root_dir)

    # Make 2 buckets -- recalled and generalized
    recall_dir = os.path.join(root_dir,'recall')
    generalized_dir = os.path.join(root_dir,'generalized')
    os.makedirs(recall_dir)
    os.makedirs(generalized_dir)

    recall = []
    generalize = []
    image_idx = 0

    model = get_model(mname=args.model)

    config = tf.ConfigProto(device_count = {'GPU': 0})
    config.gpu_options.visible_device_list = ""


    # Create center mask
    image_shape = [image_size,image_size,3]
    patch_size = image_size//2
    crop_pos = (image_size - patch_size)/2
    l = int(crop_pos)
    u = int(crop_pos + patch_size)
    mask = np.ones(image_shape)
    mask[l:u, l:u, :] = 0.0

    with tf.device('/cpu:0'):
        with tf.Session(config=config) as sess:
            #Load the GAN model
            restorer = tf.train.Saver()
            checkpoint_dir = os.path.join(os.getcwd(),'checkpoints',args.dataset.lower(),args.model.lower())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                restorer.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Invalid checkpoint directory')
                assert(False)

            # Iterate over the dict
            for test_image_path,z_inpainting in z_inpainting_dict.items():
                row = []
                image_list = []
                z_test = z_test_dict[test_image_path]
                test_cosine = cosine(z_inpainting,z_test)
                train_image_min,min_train_z,train_cosine = find_closest_vector(z_inpainting,z_train_dict)
                row.append(test_image_path)
                row.append(train_image_min)
                row.append(test_cosine)
                row.append(train_cosine)
                print('Analysis done for image : {} train_cosine : {} test_cosine : {}'.format(test_image_path,train_cosine,test_cosine))

                testImg = read_and_crop_image(test_image_path)
                Gz_inpainting = generate_image(model=model,sess=sess,z=z_inpainting.reshape(1,100))
                Gz_inpainting = Gz_inpainting.reshape(image_size,image_size,3)

                image_list.append(testImg) # Test Image -- Complete
                image_list.append(generate_inpainting(testImg,mask,Gz_inpainting)) # Inpainted image
                image_list.append(read_and_crop_image(train_image_min)) # Closest training image w.r.t z_inpainting

                # Generated Images -- from the 3z's
                image_list.append(Gz_inpainting) # G(z_inpainting)
                image_list.append(generate_image(model=model,sess=sess,z = z_test.reshape(1,100))) # G(z_test)
                image_list.append(generate_image(model=model,sess=sess,z=min_train_z.reshape(1,100))) # G(z_training_min)
                image_idx += 1

                if test_cosine <= train_cosine: # Inpainting latent vector closer to test latent vector
                    generalize.append(row)
                    merge_and_save(image_list=image_list,idx=image_idx,root_dir = generalized_dir)
                else: # Inpainting latent vector closer to training image latent vector
                    recall.append(row)
                    merge_and_save(image_list=image_list,idx=image_idx,root_dir = recall_dir)

            print('Generalized inpaintings : {}'.format(len(generalize)))
            print('Recalled inpaintings : {}'.format(len(recall)))

def generate_inpainting(testImg,mask,Gz):
    """
    Given test image,G(z_inpainting) and mask
    return the inpainting

    """
    inpainting = blend_images(image=testImg,gen_image=Gz,mask=np.multiply(255,1.0-mask),rescale=False)
    return inpainting


def get_model(mname):
    """
    Instantiate the GAN model

    """
    model = config.get_model(mtype=mname.upper(),name=mname.lower(),training=False)
    return model

def save_gz(args):
    """
    From a pickled dictionary (image_id:z_inpainting),
    save the G(z_inpainting) images in a folder

    """
    # Read in the picked dict
    filename = 'latent_space_inpaint_'+'{}.pkl'.format(args.model.upper())
    with open(filename,'rb') as f:
        z_dict = pickle.load(f)

    # Create output directory
    dir_name = 'gz_{}'.format(args.model.upper())
    dir_path = os.path.join(os.getcwd(),dir_name)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    # Get the saved GAN model
    model = get_model(mname=args.model)

    config = tf.ConfigProto(device_count = {'GPU': 0})
    config.gpu_options.visible_device_list = ""

    with tf.device('/cpu:0'):
        with tf.Session(config=config) as sess:
            #Load the GAN model
            restorer = tf.train.Saver()
            checkpoint_dir = os.path.join(os.getcwd(),'checkpoints',args.dataset.lower(),args.model.lower())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                restorer.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Invalid checkpoint directory')
                assert(False)

            for idx,z in z_dict.items():
                image_file = str(idx) + '.jpg'
                image_path = os.path.join(dir_path,image_file)
                gz = generate_image(model=model,sess=sess,z=z.reshape(1,100))
                scipy.misc.imsave(image_path,gz)





if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    save_gz(args)

