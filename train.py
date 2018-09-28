#!/usr/bin/anaconda3/bin/python3
# coding: utf-8
import tensorflow as tf
import numpy as np
import glob, os, sys
sys.path.append(os.getcwd())
import inputpipe as ip
from argparse import ArgumentParser
import utils, config
import shutil
import scipy.misc
import pickle
from tensorflow.examples.tutorials.mnist import input_data

MNIST_DIR = '/mnt/server-home/TUE/s162156/datasets/mnist'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', default=20, help='default: 20', type=int)
    parser.add_argument('--batch_size', default=128, help='default: 128', type=int)
    parser.add_argument('--num_threads', default=4, help='# of data read threads (default: 4)', type=int)
    models_str = ' / '.join(config.model_zoo)
    parser.add_argument('--model', help=models_str, required=True) # DRAGAN, CramerGAN
    parser.add_argument('--name', help='default: name=model')
    parser.add_argument('--dataset', '-D', help='CelebA / LSUN', required=True)
    parser.add_argument('--image_size',default=64)
    parser.add_argument('--ckpt_step', default=500, help='# of steps for saving checkpoint (default: 5000)', type=int)
    parser.add_argument('--renew', action='store_true', help='train model from scratch - \
        clean saved checkpoints and summaries', default=False)
    parser.add_argument('--simultaneous', action='store_true', help='Choose between alternate GD and simultaeneous GD', default=False)
    return parser


def input_pipeline(glob_pattern, batch_size, num_threads, num_epochs,image_size):
    tfrecords_list = glob.glob(glob_pattern)
    X = ip.shuffle_batch_join(tfrecords_list, batch_size=batch_size, num_threads=num_threads, num_epochs=num_epochs,image_size=image_size)
    return X


def sample_z(shape):
    return np.random.normal(size=shape)


def train(model, dataset,input_op, num_epochs, batch_size, n_examples, ckpt_step, renew=False,simultaneous = False):
    # n_examples = 202599 # same as util.num_examples_from_tfrecords(glob.glob('./data/celebA_tfrecords/*.tfrecord'))
    # 1 epoch = 1583 steps
    print("\n# of examples: {}".format(n_examples))
    print("steps per epoch: {}\n".format(n_examples//batch_size))
    n_critic = 5 # Critic training iterations per generator update for WGAN and WGAN-GP

    if model.name == 'dcgan-cons' or 'dcgan-cons_bn':
        store_grads = False
    else:
        store_grads = True

    summary_path = os.path.join('./summary/', dataset, model.name)
    ckpt_path = os.path.join('./checkpoints', dataset, model.name)

    if renew:
        if os.path.exists(summary_path):
            tf.gfile.DeleteRecursively(summary_path)
        if os.path.exists(ckpt_path):
            tf.gfile.DeleteRecursively(ckpt_path)
    if not os.path.exists(ckpt_path):
        tf.gfile.MakeDirs(ckpt_path)

    sample_dir = os.path.join(os.getcwd(),'samples',dataset.lower(),model.name)

    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)


    os.makedirs(sample_dir)

    if store_grads is True:
        d_grad_norm = []
        if model.name == 'dcgan': # For DCGAN, save both the value of both DRAGAN and WGAN-GP term
            d_grad_norm_gp = []

    config = tf.ConfigProto()

    if dataset == 'mnist':
        mnist = input_data.read_data_sets(MNIST_DIR,reshape=[])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # for epochs

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # https://github.com/tensorflow/tensorflow/issues/10972
        # TensorFlow 1.2 has much bugs for text summary
        # make config_summary before define of summary_writer - bypass bug of tensorboard

        # It seems that batch_size should have been contained in the model config ...
        total_steps = int(np.ceil(n_examples * num_epochs / float(batch_size))) # total global step
        config_list = [
            ('num_epochs', num_epochs),
            ('total_iteration', total_steps),
            ('batch_size', batch_size),
            ('dataset', dataset)
        ]
        model_config_list = [[k, str(w)] for k, w in sorted(model.args.items()) + config_list]
        model_config_summary_op = tf.summary.text(model.name + '/config', tf.convert_to_tensor(model_config_list),
            collections=[])
        model_config_summary = sess.run(model_config_summary_op)

        # print to console
        print("\n====== Process info =======")
        print("argv: {}".format(' '.join(sys.argv)))
        print("PID: {}".format(os.getpid()))
        print("====== Model configs ======")
        for k, v in model_config_list:
            print("{}: {}".format(k, v))
        print("===========================\n")

        sys.stdout.flush()

        summary_writer = tf.summary.FileWriter(summary_path, flush_secs=30, graph=sess.graph)
        summary_writer.add_summary(model_config_summary)
        saver = tf.train.Saver(max_to_keep=2) # save all checkpoints
        global_step = 0

        # Use a (fixed) validation z and keep checking images over the training loop to detect mode collapse/image quality

        val_z = sample_z([batch_size,model.z_dim])

        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = sess.run(model.global_step)
            print('\n[!] Restore from {} ... starting global step is {}\n'.format(ckpt.model_checkpoint_path, global_step))

        try:
            # If training process was resumed from checkpoints, input pipeline cannot detect
            # when training should stop. So we need `global_step < total_step` condition.
            while not coord.should_stop() and global_step < total_steps:
                # model.all_summary_op contains histogram summary and image summary which are heavy op
                summary_op = model.summary_op if global_step % 100 == 0 else model.all_summary_op

                if dataset == 'celeba':
                    batch_X = sess.run(input_op)
                else:
                    batch,_ = mnist.train.next_batch(batch_size)
                    print(batch[0,:])
                    batch = (batch-0.5)/0.5
                    batch = tf.image.resize_images(batch,[32,32]).eval()
                    batch_X = batch

                batch_z = sample_z([batch_size, model.z_dim])

                if simultaneous is False: #Alternating Gradient Descent
                    if model.name == 'wgan' or model.name == 'wgan-gp':
                        for step in range(n_critic): #Train critic till optimality for WGAN or WGAN-GP
                            _, summary = sess.run([model.D_train_op, summary_op], {model.X: batch_X, model.z: batch_z})
                    else:
                        _, summary = sess.run([model.D_train_op, summary_op], {model.X: batch_X, model.z: batch_z})

                    _, global_step = sess.run([model.G_train_op, model.global_step], {model.X: batch_X, model.z: batch_z})
                else: #Simultaneous Gradient Descent
                    _,_,summary,global_step = sess.run([model.D_train_op,model.G_train_op,summary_op,model.global_step], {model.X: batch_X, model.z: batch_z})

                summary_writer.add_summary(summary, global_step=global_step)

                if global_step % 10 == 0:
                    #Monitor losses
                    G_loss,D_loss = sess.run([model.G_loss,model.D_loss],feed_dict={model.X:batch_X,model.z:batch_z})
                    if store_grads is True:
                        d_grad_norm.append(sess.run(model.D_grad_norm,feed_dict = {model.X : batch_X, model.z : batch_z}))
                        if model.name == 'dcgan':
                            d_grad_norm_gp.append(sess.run(model.D_grad_norm_gp,feed_dict = {model.X : batch_X, model.z : batch_z}))

                    print('Global Step {} :: Generator loss = {} Discriminator loss = {}'.format(global_step,G_loss,D_loss))

                    sys.stdout.flush()

                    if global_step % ckpt_step == 0:
                        saver.save(sess, ckpt_path+'/'+model.name, global_step=global_step)
                        save_samples(sess=sess,val_z = val_z ,model=model,dir_name = sample_dir,global_step=global_step,shape=[16,8],dataset=dataset)

        except tf.errors.OutOfRangeError:
            print('\nDone -- epoch limit reached\n')

        finally:
            if store_grads == True: # Store gradient norms in a pkl file
                save_file = os.path.join(ckpt_path,'{}_grads.pkl'.format(model.name))
                with open(save_file,'wb') as f:
                    pickle.dump(d_grad_norm,f)

                if model.name == 'dcgan':
                    save_file = os.path.join(ckpt_path,'{}_grads_gp.pkl'.format(model.name))
                    with open(save_file,'wb') as f:
                        pickle.dump(d_grad_norm_gp,f)


            coord.request_stop()

        coord.join(threads)
        summary_writer.close()

def save_samples(sess,val_z,model,dir_name,global_step,shape,dataset):
    """
    Function to save samples during training

    """
    fake_samples = sess.run(model.fake_sample, {model.z: val_z})

    #Bring the images to original display range
    if dataset != 'mnist':
        fake_samples = 255*((fake_samples + 1.) / 2.)
    else:
        fake_samples = fake_samples*0.5 + 0.5

    merged_samples = utils.merge(fake_samples, size=shape)
    fn = "{:0>6d}.png".format(global_step)
    scipy.misc.imsave(os.path.join(dir_name, fn), merged_samples)

if __name__ == "__main__":
    parser = build_parser()
    FLAGS = parser.parse_args()
    FLAGS.model = FLAGS.model.upper()
    FLAGS.dataset = FLAGS.dataset.lower()
    if FLAGS.name is None:
        FLAGS.name = FLAGS.model.lower()
    config.pprint_args(FLAGS)

    # get information for dataset
    dataset_pattern, n_examples = config.get_dataset(FLAGS.dataset)
    # input pipeline
    if FLAGS.dataset == 'celeba':
        X = input_pipeline(dataset_pattern, batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_threads, num_epochs=FLAGS.num_epochs,image_size = int(FLAGS.image_size))
        resized_image_shape = [64,64,3]
    else: # MNIST or Fashion-MNIST
        X = None
        resized_image_shape = [32,32,1]

    batch_norm = True

    if FLAGS.name == 'dragan' or FLAGS.name == 'dcgan-cons':
        batch_norm = False

    if FLAGS.simultaneous == True and FLAGS.model == 'DCGAN':
        FLAGS.name = FLAGS.model.lower() + '_sim'

    model = config.get_model(FLAGS.model, FLAGS.name, training=True,image_shape=resized_image_shape,batch_norm=batch_norm,dataset = FLAGS.dataset)

    train(model=model, dataset=FLAGS.dataset,input_op=X, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,
        n_examples=n_examples, ckpt_step=FLAGS.ckpt_step, renew=FLAGS.renew,simultaneous = FLAGS.simultaneous)
