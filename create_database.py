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
    parser.add_argument('--dataset',type=str,default = 'celeba')
    parser.add_argument('--nImages',type=int,default=1000)
    args = parser.parse_args()

    if (os.path.exists(args.outDir)):
        # Remove it
        shutil.rmtree(args.outDir)

    os.makedirs(args.outDir)

    models = ['dcgan','wgan','dcgan-gp','wgan-gp','dcgan-cons','dcgan-gp']
    source_dirs = []
    for model in models:
        dir_path = os.path.join(os.getcwd(),'completions_stochastic',str(model),str(args.dataset))
        source_dirs.append(dir_path)


    for idx in range(args.nImages):
        curr_out_dir = os.path.join(args.outDir,'{}'.format(idx))
        os.makedirs(curr_out_dir)
        original_image = os.path.join(source_dirs[0],'{}'.format(idx),'original.jpg') # Copy the image from one of the source directories
        # Copy over the original image
        shutil.copy2(original_image,curr_out_dir)

        # Make the sub-folder
        genDir = os.path.join(curr_out_dir,'gen')
        os.makedirs(genDir)

        for source_dir in source_dirs:
            curr_image_file = os.path.join(source_dir,'{}'.format(idx),'gen_images','gen_1400.jpg')
            model_name = source_dir.split('/')[-2]
            dst = os.path.join(curr_out_dir,'gen','{}.jpg'.format(model_name))
            shutil.copy2(curr_image_file,dst)

if __name__ == '__main__':
    create_database()



