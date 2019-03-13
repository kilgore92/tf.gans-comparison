# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel

'''
based on DCGAN.

WGAN:
WD = max_f [ Ex[f(x)] - Ez[f(g(z))] ] where f has K-Lipschitz constraint
J = min WD (G_loss)
'''

class WGAN(BaseModel):
    def __init__(self, name, training, D_lr=5e-5, G_lr=5e-5, image_shape=[64, 64, 3], z_dim=100):
        self.ld = 10. # lambda
        self.n_critic = 5
        super(WGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr,
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)

            G = self._generator(z)
            C_real = self._critic(X)
            C_fake = self._critic(G, reuse=True)

            W_dist = tf.reduce_mean(C_real - C_fake)
            C_loss = -W_dist
            G_loss = tf.reduce_mean(-C_fake)

            C_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/critic/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/generator/')

            C_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/critic/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/generator/')

            # Gradient Penalty (GP)
            eps = tf.random_uniform(shape=[tf.shape(X)[0], 1, 1, 1], minval=0., maxval=1.)
            x_hat = eps*X + (1.-eps)*G
            C_xhat = self._critic(x_hat, reuse=True)
            C_xhat_grad = tf.gradients(C_xhat, x_hat)[0] # gradient of D(x_hat)
            C_xhat_grad_norm = tf.norm(slim.flatten(C_xhat_grad), axis=1)  # l2 norm

            # In the paper, critic networks has been trained n_critic times for each training step.
            # Here I adjust learning rate instead.
            with tf.control_dependencies(C_update_ops):
                C_train_op = tf.train.RMSPropOptimizer(learning_rate=self.D_lr).\
                    minimize(C_loss, var_list=C_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.RMSPropOptimizer(learning_rate=self.G_lr).\
                    minimize(G_loss, var_list=G_vars, global_step=global_step)

            # weight clipping
            ''' It is right that clips gamma of the batch_norm? '''

            # ver 1. clips all variables in critic
            C_clips = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in C_vars] # with gamma

            # ver 2. does not work
            # C_clips = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in C_vars if 'gamma' not in var.op.name] # without gamma

            # ver 3. works but strange
            # C_clips = []
            # for var in C_vars:
            #     if 'gamma' not in var.op.name:
            #         C_clips.append(tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)))
            #     else:
            #         C_clips.append(tf.assign(var, tf.clip_by_value(var, -1.00, 1.00)))

            with tf.control_dependencies([C_train_op]): # should be iterable
                C_train_op = tf.tuple(C_clips) # tf.group ?

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('C_loss', C_loss),
                tf.summary.scalar('W_dist', W_dist)
            ])

            # sparse-step summary
            tf.summary.image('G', G, max_outputs=self.FAKE_MAX_OUTPUT)
            # tf.summary.histogram('real_probs', D_real_prob)
            # tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = C_train_op # compatibility for train.py
            self.G_train_op = G_train_op
            self.G_loss = G_loss
            self.D_loss = C_loss
            self.G = G
            self.global_step = global_step
            self.D_grad_norm = C_xhat_grad_norm
            self.C_fake = C_fake
            self.W_dist = W_dist


            # Image In-painting
            self.mask = tf.placeholder(tf.float32, self.shape, name='mask')
            self.lam = 0.003 # Value taken from paper

            # Reduce the difference in the masked part -- TODO : Add weighting term (from paper) to the mask*image product
            self.contextual_loss = tf.reduce_mean(
                tf.contrib.layers.flatten(
                    tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.X))), 1)

            # The reconstructed/completed image must also "fool" the discriminator
            self.perceptual_loss = self.G_loss
            self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
            self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)
            self.grad_perceptual_loss = tf.gradients(self.perceptual_loss,self.z)
            self.grad_contextual_loss = tf.gradients(self.contextual_loss,self.z)
            self.grad_norm_perceptual_loss = tf.norm(self.grad_perceptual_loss,axis=1)
            self.grad_norm_contextual_loss = tf.norm(self.grad_contextual_loss,axis=1)

    def _critic(self, X, reuse=False):
        ''' K-Lipschitz function '''
        with tf.variable_scope('critic', reuse=reuse):
            net = X

            with slim.arg_scope([slim.conv2d], kernel_size=[5,5], stride=2, activation_fn=ops.lrelu,
                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                net = slim.conv2d(net, 64, normalizer_fn=None)
                expected_shape(net, [32, 32, 64])
                net = slim.conv2d(net, 128)
                expected_shape(net, [16, 16, 128])
                net = slim.conv2d(net, 256)
                expected_shape(net, [8, 8, 256])
                net = slim.conv2d(net, 512)
                expected_shape(net, [4, 4, 512])

            net = slim.flatten(net)
            net = slim.fully_connected(net, 1, activation_fn=None)

            return net

    def _generator(self, z, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4*4*1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])
            filter_num = 512
            input_size = 4
            stride = 2
            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5,5], stride=stride, padding='SAME',
                activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                while input_size < (self.shape[0]//stride):
                    net = slim.conv2d_transpose(net, filter_num)
                    expected_shape(net, [input_size*stride, input_size*stride, filter_num])
                    filter_num = filter_num//2
                    input_size = input_size*stride

                net = slim.conv2d_transpose(net, self.shape[2], activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [self.shape[0], self.shape[1], self.shape[2]])

                return net
