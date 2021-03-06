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
    def __init__(self, name, training, D_lr=2e-4, G_lr=2e-4, image_shape=[64, 64, 3], z_dim=100):
        self.beta1 = 0.5
        super(DCGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr,
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None] + self.shape)
            self.z = tf.placeholder(tf.float32, [None, self.z_dim])
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.G = self._generator(self.z)
            self.D_real_prob, self.D_real_logits = self._discriminator(self.X)
            self.D_fake_prob, self.D_fake_logits = self._discriminator(self.G, reuse=True)

            #G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits,labels=tf.ones_like(self.D_fake_prob)))
            self.D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.D_real_logits), logits=self.D_real_logits)
            self.D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.D_fake_logits), logits=self.D_fake_logits)
            self.D_loss = self.D_loss_real + self.D_loss_fake

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/G/')

            # DRAGAN term
           # shape = tf.shape(self.X)
           # eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
           # x_mean, x_var = tf.nn.moments(self.X, axes=[0,1,2,3])
           # x_std = tf.sqrt(x_var) # magnitude of noise decides the size of local region
           # noise = 0.5*x_std*eps # delta in paper
           # # Author suggested U[0,1] in original paper, but he admitted it is bug in github
           # # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.
           # alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
           # xhat = tf.clip_by_value(self.X + alpha*noise, -1., 1.) # x_hat should be in the space of X

           # D_xhat_prob, D_xhat_logits = self._discriminator(xhat, reuse=True)
           # # Originally, the paper suggested D_xhat_prob instead of D_xhat_logits.
           # # But D_xhat_prob (D with sigmoid) causes numerical problem (NaN in gradient).
           # D_xhat_grad = tf.gradients(D_xhat_logits, xhat)[0] # gradient of D(x_hat)
           # D_xhat_grad_norm = tf.norm(slim.flatten(D_xhat_grad), axis=1)  # l2 norm

           # # WGAN-GP term
           # eps = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=0., maxval=1.)
           # x_hat_gp = eps*X + (1.-eps)*self.G
           # D_xhat_gp = self._discriminator(x_hat_gp, reuse=True)
           # D_xhat_grad_gp = tf.gradients(D_xhat_gp, x_hat_gp)[0] # gradient of D(x_hat)
           # D_xhat_grad_norm_gp = tf.norm(slim.flatten(D_xhat_grad_gp), axis=1)  # l2 norm


            with tf.control_dependencies(D_update_ops):
                self.D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).\
                    minimize(self.D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                # learning rate 2e-4/1e-3
                self.G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1).\
                    minimize(self.G_loss, var_list=G_vars, global_step=self.global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', self.G_loss),
                tf.summary.scalar('D_loss', self.D_loss),
                tf.summary.scalar('D_loss/real', self.D_loss_real),
                tf.summary.scalar('D_loss/fake', self.D_loss_fake)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', self.G, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.histogram('real_probs', self.D_real_prob)
            tf.summary.histogram('fake_probs', self.D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()


            # Image In-painting
            self.mask = tf.placeholder(tf.float32, self.shape, name='mask')
            self.lam = 0.003 # Value taken from paper

            # Reduce the difference in the masked part -- TODO : Add weighting term (from paper) to the mask*image product
            self.contextual_loss = tf.reduce_mean(
                tf.contrib.layers.flatten(
                    tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.X))), 1)

            # Perceptual loss -- we don't need to avg. over different inp processes, they proceed independently of each other!
            self.perceptual_loss = self.G_loss

            # The reconstructed/completed image must also "fool" the discriminator
            self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
            self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)
            self.grad_perceptual_loss = tf.gradients(self.perceptual_loss,self.z)
            self.grad_contextual_loss = tf.gradients(self.contextual_loss,self.z)
            self.grad_norm_perceptual_loss = tf.norm(self.grad_perceptual_loss,axis=1)
            self.grad_norm_contextual_loss = tf.norm(self.grad_contextual_loss,axis=1)

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            print('Bulding discriminator graph')
            net = X
            width = self.shape[0]
            filter_num = 64
            stride = 2
            num_conv_layers = 4
            with slim.arg_scope([slim.conv2d], kernel_size=[5,5], stride=stride, padding='SAME', activation_fn=ops.lrelu,
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
            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[5,5], stride=stride, padding='SAME',
                activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=self.bn_params):
                while input_size < (self.shape[0]//stride):
                    net = slim.conv2d_transpose(net, filter_num)
                    expected_shape(net, [input_size*stride, input_size*stride, filter_num])
                    filter_num = filter_num//2
                    input_size = input_size*stride

                net = slim.conv2d_transpose(net, self.shape[2] , activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [self.shape[0], self.shape[1], self.shape[2]])

                return net
