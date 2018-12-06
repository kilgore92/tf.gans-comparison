#coding: utf-8
import tensorflow as tf
import numpy as np
import utils
import config
import os, glob
import scipy.misc
from argparse import ArgumentParser
import pickle
import shutil
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from complete import rescale_image
slim = tf.contrib.slim


def build_parser():
    parser = ArgumentParser()
    models_str = ' / '.join(config.model_zoo)
    parser.add_argument('--model', help=models_str, required=True)
    parser.add_argument('--name', help='default: name=model')
    parser.add_argument('--dataset', '-D', help='CelebA / LSUN', required=True)
    parser.add_argument('--batch_size',default=512, type=int,help='Batch size for generated images')
    parser.add_argument('--gpu',type=str,default="0")
    parser.add_argument('--gen',action='store_true',default=True)
    return parser


def sample_z(shape):
    return np.random.normal(size=shape)


def generate_image(model,sess,z):
    """
    For a given z, return G(z)

    """
    g_img = sess.run(model.fake_sample,feed_dict = {model.z : z})
    return rescale_image(g_img[0])

def get_all_checkpoints(ckpt_dir, force=False):
    '''
    When the learning is interrupted and resumed, all checkpoints can not be fetched with get_checkpoint_state
    (The checkpoint state is rewritten from the point of resume).
    This function fetch all checkpoints forcely when arguments force=True.
    '''

    if force:
        ckpts = os.listdir(ckpt_dir) # get all fns
        ckpts = map(lambda p: os.path.splitext(p)[0], ckpts) # del ext
        ckpts = set(ckpts) # unique
        ckpts = filter(lambda x: x.split('-')[-1].isdigit(), ckpts) # filter non-ckpt
        ckpts = sorted(ckpts, key=lambda x: int(x.split('-')[-1])) # sort
        ckpts = map(lambda x: os.path.join(ckpt_dir, x), ckpts) # fn => path
    else:
        ckpts = tf.train.get_checkpoint_state(ckpt_dir).all_model_checkpoint_paths

    return ckpts


def eval(model, name, dataset,batch_size, gpu = "1",load_all_ckpt=True,sample_dir=None):
    if name == None:
        name = model.name
    if sample_dir == None:
        dir_name = os.path.join('eval', dataset, name)
    else:
        dir_name = os.path.join(sample_dir,dataset,name)
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(gpu)
    with tf.Session(config=config) as sess:
        #Load the GAN model
        restorer = tf.train.Saver()
        checkpoint_dir = os.path.join(os.getcwd(),'checkpoints',dataset.lower(),name.lower())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            restorer.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Invalid checkpoint directory')
            assert(False)

        n_batches = 20 # Atleast 10,000 images needed to get a good FID estimate

        for batch in range(n_batches):
            z_ = sample_z([batch_size, model.z_dim])
            fake_samples = sess.run(model.fake_sample, {model.z: z_})
            # inverse transform: [-1, 1] => [0, 1]
            fake_samples = (fake_samples + 1.) / 2.
            for fake_sample,idx in zip(fake_samples,range(len(fake_samples))):
                fn = "{}_{}.jpg".format(batch,idx)
                scipy.misc.imsave(os.path.join(dir_name, fn), fake_sample)
            print('Generated {} batches'.format(batch))


def save_gz(model,mname,dataset):
    """
    From a pickled dictionary (image_id:z_inpainting),
    save the G(z_inpainting) images in a folder

    """
    # Read in the picked dict
    filename = 'latent_space_inpaint_'+'{}.pkl'.format(mname.upper())
    with open(filename,'rb') as f:
        z_dict = pickle.load(f)

    # Create output directory
    dir_name = 'gz_{}'.format(mname.upper())
    dir_path = os.path.join(os.getcwd(),dir_name)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


    tf_config = tf.ConfigProto(device_count = {'GPU': 0})
    tf_config.gpu_options.visible_device_list = ""

    with tf.device('/cpu:0'):
        with tf.Session(config=tf_config) as sess:
            #Load the GAN model
            restorer = tf.train.Saver()
            checkpoint_dir = os.path.join(os.getcwd(),'checkpoints',dataset.lower(),mname.lower())
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
'''
You can create a gif movie through imagemagick on the commandline:
$ convert -delay 20 eval/* movie.gif
'''
# def to_gif(dir_name='eval'):
#     images = []
#     for path in glob.glob(os.path.join(dir_name, '*.png')):
#         im = scipy.misc.imread(path)
#         images.append(im)

#     # make_gif(images, dir_name + '/movie.gif', duration=10, true_image=True)
#     imageio.mimsave('movie.gif', images, duration=0.2)


if __name__ == "__main__":
    parser = build_parser()
    FLAGS = parser.parse_args()
    FLAGS.model = FLAGS.model.upper()
    FLAGS.dataset = FLAGS.dataset.lower()
    if FLAGS.name is None:
        FLAGS.name = FLAGS.model.lower()
    config.pprint_args(FLAGS)
    # training=False => build generator only
    model = config.get_model(FLAGS.model, FLAGS.name, training=False)
    if FLAGS.gen is True:
        save_gz(model=model,mname=FLAGS.name,dataset=FLAGS.dataset)
    else:
        eval(model,dataset=FLAGS.dataset, name=FLAGS.name, batch_size = FLAGS.batch_size, load_all_ckpt=True)
