#!/usr/bin/anaconda3/bin/python3
# coding: utf-8
import tensorflow as tf
import numpy as np
import glob, os, sys
sys.path.append(os.path.join(os.getcwd(),'src'))
from argparse import ArgumentParser
import utils, config
import shutil
import scipy.misc
from convert import center_crop
import cv2
import pickle
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

slim = tf.contrib.slim

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=512, help='default: 128', type=int)
    parser.add_argument('--num_threads', default=4, help='# of data read threads (default: 4)', type=int)
    models_str = ' / '.join(config.model_zoo)
    parser.add_argument('--model', help=models_str, required=True) # DRAGAN, CramerGAN
    parser.add_argument('--name', help='default: name=model')
    parser.add_argument('--images', '-D', help='Path to folder containing images', required=True)
    parser.add_argument('--image_size',default=64,type = int)
    parser.add_argument('--dataset',help='Name of dataset on which inpainting is done',default='celeba',required='true')
    parser.add_argument('--maskType',help='center/left/right/random/bottom',default='center')
    parser.add_argument('--nIter',help='Number of iteration to perform for each in-painting',default=1500,type=int)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--clipping',type=str,help='Options: standard or stochastic',default='stochastic')
    return parser


def blend_images(image,gen_image,mask,rescale=True):
    # TODO : Re-factor code to enable blending for non-center masks

    """
    Blend generated patch into the masked image
    using the OpenCV implementation of Poisson blending

    """
    if rescale is True:
        gen_image = rescale_image(gen_image)

    if rescale is True:
        image = rescale_image(image)

    image = np.array(image,dtype = np.uint8)
    mask = np.array(mask,dtype=np.uint8)
    gen_image = np.array(gen_image,dtype = np.uint8)
    center = (image.shape[0]//2,image.shape[1]//2)

    blended_image = cv2.seamlessClone(gen_image,image,mask,center,cv2.NORMAL_CLONE)
    return blended_image

def rescale_image(image):
    image = 255*((image+1.)/2.)
    return image

def get_image_paths(image_dir):
    """

    Returns a list of paths for images to be in-painted

    """
    path_list = [os.path.join(image_dir,f) for f in os.listdir(image_dir)]
    return path_list

def read_image(image_path,image_shape,crop=True,n_channels=3):

    im = scipy.misc.imread(image_path)
    if crop is True:
        im = center_crop(im,image_shape)
    im = np.array(im).astype(np.float32)
    im = scipy.misc.imresize(im,(image_shape[0],image_shape[1]))
    im = np.reshape(im,(image_shape[0],image_shape[1],n_channels))
    return (im/127.5 - 1)

def save_image(image,path,n_channels=3):
    if n_channels == 1:
        image = np.reshape(image,(image.shape[0],image.shape[1]))

    scipy.misc.imsave(path,image)


def complete(args):
    """
    Performs in-painting over images using a pre-trained
    GAN model : http://arxiv.org/abs/1607.07539

    """


    image_paths = get_image_paths(args.images)
    nImgs = len(image_paths)

    print('Images found : {}'.format(nImgs))

    if args.dataset == 'celeba':
        crop = True
        n_channels = 3
    else:
        n_channels = 1
        crop = False

    image_shape = [int(args.image_size),int(args.image_size),n_channels]

    batch_idxs = int(np.ceil(nImgs/args.batch_size))

    maskType = args.maskType

    folder_name = os.path.join('completions',args.dataset.lower(),args.model.lower())
    dumpDir = os.path.join(folder_name,maskType)


    if os.path.exists(dumpDir):
        shutil.rmtree(dumpDir)
    os.makedirs(dumpDir)


    if maskType == 'random':
        fraction_masked = 0.2
        mask = np.ones(image_shape)
        mask[np.random.random(image_shape[:2]) < fraction_masked] = 0.0
    elif maskType == 'center': # Center mask removes 25% of the image
        patch_size = args.image_size//2
        crop_pos = (args.image_size - patch_size)/2
        mask = np.ones(image_shape)
        sz = args.image_size
        l = int(crop_pos)
        u = int(crop_pos + patch_size)
        mask[l:u, l:u, :] = 0.0
    elif maskType == 'left':
        mask = np.ones(image_shape)
        c = args.image_size // 2
        mask[:,:c,:] = 0.0
    elif maskType == 'full':
        mask = np.ones(image_shape)
    elif maskType == 'grid':
        mask = np.zeros(image_shape)
        mask[::4,::4,:] = 1.0
    elif maskType == 'bottom':
        mask = np.ones(image_shape)
        bottom_half = int(args.image_size/2)
        mask[bottom_half:args.image_size,:,:] = 0.0

    else:
        print('Invalid mask type provided')
        assert(False)

    tf_config = tf.ConfigProto()

    with tf.Session(config=tf_config) as sess:
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        # Load the checkpoint file into the model
        # Get the model
        # If Training is set to false, the discriminator ops graph is not built.
        # The discriminator graph is used to compute the in-painting loss. Hacked it now, please FIX THIS - TODO

        if args.model.lower() == 'dragan' or args.model.lower()=='dcgan-cons': # Pick the non-BN version of DRAGAN and DCGAN-CONS
            model = config.get_model(args.model.upper(),args.model.lower(), training=True,batch_norm=False,image_shape=image_shape)
        else:
            model = config.get_model(args.model.upper(),args.model.lower(), training=True,image_shape=image_shape)

        restorer = tf.train.Saver()
        checkpoint_dir = os.path.join(os.getcwd(),'checkpoints',args.dataset.lower(),args.model.lower())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            restorer.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Invalid checkpoint directory')
            assert(False)

        critic = False

        if model.name == 'wgan' or model.name == 'wgan-gp':
            critic = True

        for idx in range(0, batch_idxs):
            l = idx*args.batch_size
            u = min((idx+1)*args.batch_size, nImgs)
            batchSz = u-l
            batch_files = image_paths[l:u]
            batch = [read_image(batch_file,[args.image_size,args.image_size],n_channels=n_channels,crop=crop) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            masked_images = np.multiply(batch_images, mask)
            if batchSz < args.batch_size:
                padSz = ((0, int(args.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)
                masked_images = np.multiply(batch_images, mask)


            zhats = np.random.uniform(-1, 1, size=(args.batch_size, model.z_dim))

            disc_score_tracker = [] #tracks the discriminator/critic score for every image in the batch
            perceptual_loss_grad = []
            contextual_loss_grad = []
            # Variables for ADAM
            m = 0
            v = 0

            for file_idx in range(len(batch_images)):
                folder_idx = l + file_idx

                outDir = os.path.join(dumpDir,'{}'.format(folder_idx))
                os.makedirs(outDir) # Directory that stores real and masked images, different for each real image

                if n_channels == 3:
                    genDir = os.path.join(outDir,'gen_images') # Directory that stores iterations of in-paintings
                    os.makedirs(genDir)

                genDir_overlay = os.path.join(outDir,'gen_images_overlay') # Directory that stores iterations of in-paintings
                os.makedirs(genDir_overlay)

                gzDir = os.path.join(outDir,'gz')
                os.makedirs(gzDir)

                save_image(image=batch_images[file_idx,:,:,:],path=os.path.join(outDir,'original.jpg'),n_channels=n_channels)
                save_image(image=masked_images[file_idx,:,:,:],path=os.path.join(outDir,'masked.jpg'),n_channels=n_channels)


            for i in range(args.nIter):
                fd = {
                    model.z: zhats,
                    model.mask: mask,
                    model.X: batch_images,
                    }
                run = [model.complete_loss, model.perceptual_loss , model.contextual_loss,model.grad_complete_loss, model.G,model.grad_norm_perceptual_loss,model.grad_norm_contextual_loss]
                complete_loss,perceptual_loss,contextual_loss, g, G_imgs, grad_norm_perceptual_loss,grad_norm_contextual_loss = sess.run(run, feed_dict=fd)

                #Capture the gradient norms of both loss components
                perceptual_loss_grad.append(grad_norm_perceptual_loss[0])
                contextual_loss_grad.append(grad_norm_contextual_loss[0])

                if model.name!='wgan' and model.name!='wgan-gp':
                    disc_scores = sess.run(model.D_fake_prob,feed_dict={model.z:zhats})
                else:
                    disc_scores = sess.run(model.C_fake,feed_dict={model.z:zhats})

                disc_score_tracker.append(disc_scores.flatten())


                if i%100 == 0:
                    # Compute mean score given to this batch of images by the Discriminator
                    if model.name!='wgan' or model.name.lower()!='wgan-gp':
                        mean_disc_score = np.mean(disc_scores)
                    else:
                        mean_disc_score = np.mean(disc_scores)

                    print('Timestamp: {:%Y-%m-%d %H:%M:%S} Batch : {}/{}. Iteration : {}. Mean complete loss : {} Mean Perceptual loss : {} Mean Contextual Loss: {} Discriminator/Critic Score: {}'.format(datetime.now(),idx,batch_idxs,i, np.mean(complete_loss[0:batchSz]),perceptual_loss,np.mean(contextual_loss[0:batchSz]),mean_disc_score))

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-mask)
                    completed = []

                    #Direct overlay
                    overlay = masked_images + inv_masked_hat_images

                    #Poisson Blending
                    if n_channels ==  3:# OpenCV Poisson Blending supports only 3-channel image blending. FIXME
                        for img,indx in zip(G_imgs,range(len(G_imgs))):
                            completed.append(blend_images(image = overlay[indx,:,:,:], gen_image = img,mask = np.multiply(255,1.0-mask)))
                        completed = np.asarray(completed)

                    # Save all in-painted images of this iteration in their respective image folders
                    for  image_idx in range(args.batch_size):
                        folder_idx = l + image_idx

                        save_path_overlay = os.path.join(dumpDir,'{}'.format(folder_idx),'gen_images_overlay','gen_{}.jpg'.format(i))
                        save_path_gz = os.path.join(dumpDir,'{}'.format(folder_idx),'gz','gz_{}.jpg'.format(i))
                        overlay[image_idx,:,:,:] = rescale_image(overlay[image_idx,:,:,:])

                        if n_channels == 3:
                            save_path = os.path.join(dumpDir,'{}'.format(folder_idx),'gen_images','gen_{}.jpg'.format(i))
                            save_image(image=completed[image_idx,:,:,:],path=save_path,n_channels=n_channels)

                        save_image(image=overlay[image_idx,:,:,:],path=save_path_overlay,n_channels=n_channels)
                        save_image(image=rescale_image(G_imgs[image_idx,:,:,:]),path=save_path_gz,n_channels=n_channels)


                # Adam implementation
                m_prev = np.copy(m)
                v_prev = np.copy(v)
                m = args.beta1 * m_prev + (1 - args.beta1) * g[0]
                v = args.beta2 * v_prev + (1 - args.beta2) * np.multiply(g[0], g[0])
                m_hat = m / (1 - args.beta1 ** (i + 1))
                v_hat = v / (1 - args.beta2 ** (i + 1))
                zhats += - np.true_divide(args.lr * m_hat, (np.sqrt(v_hat) + args.eps))


                sys.stdout.flush()

                if args.clipping == 'standard':
                # Standard Clipping
                    zhats = np.clip(zhats, -1, 1)
                elif args.clipping == 'stochastic':
                # Stochastic Clipping
                    for batch_zhat,batch_id in zip(zhats,range(zhats.shape[0])):
                        for elem,elem_id in zip(batch_zhat,range(batch_zhat.shape[0])):
                            if elem > 1 or elem < -1:
                                zhats[batch_id][elem_id] = np.random.uniform(-1,1) # FIXME : There has to be a less shitty way to modify an array in-place
                else:
                    print('Invalid clipping mode')
                    assert(False)

            #Save the matrix for the batch once done
            disc_score_tracker = np.asarray(disc_score_tracker)
            perceptual_loss_grad = np.asarray(perceptual_loss_grad)
            contextual_loss_grad = np.asarray(contextual_loss_grad)

            with open(os.path.join(dumpDir,'disc_scores_batch_{}.pkl'.format(idx)),'wb') as f:
                pickle.dump(disc_score_tracker,f)

            with open(os.path.join(dumpDir,'p_loss_grad_batch_{}.pkl'.format(idx)),'wb') as f:
                pickle.dump(perceptual_loss_grad,f)

            with open(os.path.join(dumpDir,'c_loss_grad_batch_{}.pkl'.format(idx)),'wb') as f:
                pickle.dump(contextual_loss_grad,f)

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    complete(args)

