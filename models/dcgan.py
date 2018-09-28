# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel
import math
'''Original hyperparams:
optimizer - SGD
init - stddev 0.02
'''

class DCGAN(BaseModel):
    def __init__(self, name, training, D_lr=2e-4, G_lr=2e-4, image_shape=[64, 64, 3], z_dim=100,dataset='celeba'):
        self.beta1 = 0.5
        super(DCGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr,
            image_shape=image_shape, z_dim=z_dim,dataset=dataset)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            G = self._generator(z)
            D_real_prob, D_real_logits = self._discriminator(X)
            D_fake_prob, D_fake_logits = self._discriminator(G, reuse=True)

            G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_logits), logits=D_real_logits)
            D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_logits), logits=D_fake_logits)
            D_loss = D_loss_real + D_loss_fake

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/G/')

            # DRAGAN term
            shape = tf.shape(X)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(X, axes=[0,1,2,3])
            x_std = tf.sqrt(x_var) # magnitude of noise decides the size of local region
            noise = 0.5*x_std*eps # delta in paper
            # Author suggested U[0,1] in original paper, but he admitted it is bug in github
            # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.
            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            xhat = tf.clip_by_value(X + alpha*noise, -1., 1.) # x_hat should be in the space of X

            D_xhat_prob, D_xhat_logits = self._discriminator(xhat, reuse=True)
            # Originally, the paper suggested D_xhat_prob instead of D_xhat_logits.
            # But D_xhat_prob (D with sigmoid) causes numerical problem (NaN in gradient).
            D_xhat_grad = tf.gradients(D_xhat_logits, xhat)[0] # gradient of D(x_hat)
            D_xhat_grad_norm = tf.norm(slim.flatten(D_xhat_grad), axis=1)  # l2 norm

            # WGAN-GP term
            eps = tf.random_uniform(shape=[tf.shape(X)[0], 1, 1, 1], minval=0., maxval=1.)
            x_hat_gp = eps*X + (1.-eps)*G
            D_xhat_gp = self._discriminator(x_hat_gp, reuse=True)
            D_xhat_grad_gp = tf.gradients(D_xhat_gp, x_hat_gp)[0] # gradient of D(x_hat)
            D_xhat_grad_norm_gp = tf.norm(slim.flatten(D_xhat_grad_gp), axis=1)  # l2 norm


            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).\
                    minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                # learning rate 2e-4/1e-3
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1).\
                    minimize(G_loss, var_list=G_vars, global_step=global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('D_loss/real', D_loss_real),
                tf.summary.scalar('D_loss/fake', D_loss_fake)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.histogram('real_probs', D_real_prob)
            tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.G_loss = G_loss
            self.D_loss = D_loss
            self.fake_sample = G
            self.global_step = global_step
            self.D_grad_norm = D_xhat_grad_norm
            self.D_grad_norm_gp = D_xhat_grad_norm_gp

            # Image In-painting
            self.mask = tf.placeholder(tf.float32, self.shape, name='mask')
            self.lam = 0.003 # Value taken from paper

            # Reduce the difference in the masked part -- TODO : Add weighting term (from paper) to the mask*image product
            self.contextual_loss = tf.reduce_sum(
                tf.contrib.layers.flatten(
                    tf.abs(tf.multiply(self.mask, self.fake_sample) - tf.multiply(self.mask, self.X))), 1)

            # The reconstructed/completed image must also "fool" the discriminator
            self.perceptual_loss = self.G_loss
            self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
            self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            net = X
            width = self.shape[0]

            if self.dataset == 'celeba':
                filter_num = 64
                kernel_size = [5,5]
            else:
                filter_num = 128
                kernel_size = [4,4]

            stride = 2

            num_conv_layers = 4

            with slim.arg_scope([slim.conv2d], kernel_size=kernel_size, stride=stride, padding='SAME', activation_fn=ops.lrelu,
                normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                for layer_num in range(1,num_conv_layers + 1):
                    if layer_num == 1: # No batch norm for the first convolution
                        net = slim.conv2d(net, filter_num, normalizer_fn=None)
                    else:
                        net = slim.conv2d(net, filter_num)
                    output_dim = math.ceil(width/stride) # Since padding='SAME', refer : https://www.tensorflow.org/api_guides/python/nn#Convolution -- Ishaan
                    expected_shape(net, [output_dim, output_dim, filter_num])
                    width = width // 2
                    filter_num = filter_num*2

            net = slim.flatten(net)
            logits = slim.fully_connected(net, 1, activation_fn=None)
            prob = tf.sigmoid(logits)

            return prob, logits

    def _generator(self, z, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            net = z
            net = slim.fully_connected(net, 4*4*1024, activation_fn=tf.nn.relu)
            net = tf.reshape(net, [-1, 4, 4, 1024])
            filter_num = 512
            input_size = 4
            stride = 2

            if self.dataset == 'celeba':
                kernel_size = [5,5]
            else:
                kernel_size = [4,4]


            with slim.arg_scope([slim.conv2d_transpose], kernel_size=kernel_size, stride=stride, padding='SAME',
                activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                while input_size < (self.shape[0]//stride):
                    net = slim.conv2d_transpose(net, filter_num)
                    expected_shape(net, [input_size*stride, input_size*stride, filter_num])
                    filter_num = filter_num//2
                    input_size = input_size*stride

                net = slim.conv2d_transpose(net, self.shape[2], activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [self.shape[0], self.shape[1], self.shape[2]])

                return net
