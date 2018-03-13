import tensorflow as tf
from tqdm import tqdm
import numpy as np
import inputpipe as ip
import glob, os, sys
from argparse import ArgumentParser
import utils, config
import shutil
import scipy.misc
from eval import get_all_checkpoints
from convert import center_crop
slim = tf.contrib.slim

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, help='default: 128', type=int)
    parser.add_argument('--num_threads', default=4, help='# of data read threads (default: 4)', type=int)
    models_str = ' / '.join(config.model_zoo)
    parser.add_argument('--model', help=models_str, required=True) # DRAGAN, CramerGAN
    parser.add_argument('--name', help='default: name=model')
    parser.add_argument('--images', '-D', help='Path to folder containing images', required=True)
    parser.add_argument('--image_size',default=64,type = int)
    parser.add_argument('--outDir', '-O', help='Output folder for in-painted images', required=True)
    parser.add_argument('--checkpoint_dir',help='Path to folder with pre-trained GAN',required=True)
    parser.add_argument('--maskType',help='center/left/right/random',default='center')
    parser.add_argument('--nIter',help='Number of iteration to perform for each in-painting',default=1500,type=int)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--lr', type=float, default=0.005)

    return parser

def get_image_paths(image_dir):
    """

    Returns a list of paths for images to be in-painted

    """
    path_list = [os.path.join(image_dir,f) for f in os.listdir(image_dir)]
    return path_list

def read_image(image_path,image_shape):

    im = scipy.misc.imread(image_path, mode='RGB')
    resized_image = center_crop(im,image_shape)
    resized_image = np.array(resized_image).astype(np.float32)
    return (resized_image/127.5 - 1)

def save_image(image,path):
    image = 255*(image+1.)/2.
    scipy.misc.imsave(path,image)

def complete(args):
    """
    Performs in-painting over images using a pre-trained
    GAN model
    """


    image_paths = get_image_paths(args.images)
    nImgs = len(image_paths)

    image_shape = [int(args.image_size),int(args.image_size),3]


    batch_idxs = int(np.ceil(nImgs/args.batch_size))

    if args.maskType == 'random':
        fraction_masked = 0.2
        mask = np.ones(image_shape)
        mask[np.random.random(image_shape[:2]) < fraction_masked] = 0.0
    elif args.maskType == 'center': # Center mask removes 25% of the image
        patch_size = args.image_size//2
        crop_pos = (args.image_size - patch_size)/2
        mask = np.ones(image_shape)
        sz = args.image_size
        l = int(crop_pos)
        u = int(crop_pos + patch_size)
        mask[l:u, l:u, :] = 0.0
    elif args.maskType == 'left':
        mask = np.ones(image_shape)
        c = args.image_size // 2
        mask[:,:c,:] = 0.0
    elif args.maskType == 'full':
        mask = np.ones(image_shape)
    elif args.maskType == 'grid':
        mask = np.zeros(image_shape)
        mask[::4,::4,:] = 1.0
    else:
        print('Invalid mask type provided')
        assert(False)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.visible_device_list = "0"

    with tf.Session(config=tf_config) as sess:
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        # Load the checkpoint file into the model
        # Get the model
        # If Training is set to false, the discriminator ops graph is not built.
        # This is needed for in-painting. Hacked it now, please FIX THIS - TODO
        model = config.get_model(args.model.upper(),args.model.lower(), training=True)
        restorer = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            restorer.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Invalid checkpoint directory')
            assert(False)

        if os.path.exists(args.outDir) is True:
            shutil.rmtree(args.outDir)
        os.makedirs(args.outDir)

        for idx in range(0, batch_idxs):
            l = idx*args.batch_size
            u = min((idx+1)*args.batch_size, nImgs)
            batchSz = u-l
            batch_files = image_paths[l:u]
            batch = [read_image(batch_file,[args.image_size,args.image_size]) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            masked_images = np.multiply(batch_images, mask)
            if batchSz < args.batch_size:
                padSz = ((0, int(args.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)
                masked_images = np.multiply(batch_images, mask)

            zhats = np.random.uniform(-1, 1, size=(args.batch_size, model.z_dim))
            # Variables for ADAM
            m = 0
            v = 0

            for file_idx in range(len(batch_files)):
                folder_idx = l + file_idx
                outDir = os.path.join(args.outDir,'{}'.format(folder_idx))
                os.makedirs(outDir) # Directory that stores real and masked images, different for each real image
                genDir = os.path.join(outDir,'gen_images') # Directory that stores iterations of in-paintings
                os.makedirs(genDir)
                save_image(image=batch_images[file_idx,:,:,:],path=os.path.join(outDir,'original.jpg'))
                save_image(image=masked_images[file_idx,:,:,:],path=os.path.join(outDir,'masked.jpg'))


            for i in range(args.nIter):
                fd = {
                    model.z: zhats,
                    model.mask: mask,
                    model.X: batch_images,
                    }
                run = [model.complete_loss, model.grad_complete_loss, model.fake_sample]
                loss, g, G_imgs= sess.run(run, feed_dict=fd)

                if i%100 == 0:
                    print('Batch : {}. Iteration : {}. Mean loss : {}'.format(idx,i, np.mean(loss[0:batchSz])))
                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-mask)
                    completed = masked_images + inv_masked_hat_images
                    # Save all in-painted images of this iteration in their respective image folders
                    for  image_idx in range(len(completed)):
                        folder_idx = l + image_idx
                        save_path = os.path.join(args.outDir,'{}'.format(folder_idx),'gen_images','gen_{}.jpg'.format(i))
                        save_image(image=completed[image_idx,:,:,:],path=save_path)

                # Adam implementation
                m_prev = np.copy(m)
                v_prev = np.copy(v)
                m = args.beta1 * m_prev + (1 - args.beta1) * g[0]
                v = args.beta2 * v_prev + (1 - args.beta2) * np.multiply(g[0], g[0])
                m_hat = m / (1 - args.beta1 ** (i + 1))
                v_hat = v / (1 - args.beta2 ** (i + 1))
                zhats += - np.true_divide(args.lr * m_hat, (np.sqrt(v_hat) + args.eps))
                zhats = np.clip(zhats, -1, 1)



if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    complete(args)

