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

class DCGAN_CONS(BaseModel):
    def __init__(self, name, training, D_lr=1e-4, G_lr=1e-4, image_shape=[64, 64, 3], z_dim=100,batch_norm=False):
        self.norm_fn = None
        if batch_norm is True:
            self.norm_fn = slim.batch_norm

        super(DCGAN_CONS, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr,
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None, self.z_dim])
            global_step = tf.Variable(0, name='global_step', trainable=False)
            G = self._generator(z)
            D_real_prob, D_real_logits = self._discriminator(X)
            D_fake_prob, D_fake_logits = self._discriminator(G, reuse=True)

            # Average losses per mini-batch
            G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_logits), logits=D_real_logits)
            D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_logits), logits=D_fake_logits)
            D_loss = D_loss_real + D_loss_fake

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/D/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/G/')

            # Consensus Optimization Regularization term -- Calculated for per-batch loss
            gamma = 10

            # D_loss/G_loss is already the average loss over the batch.
            D_grad = tf.gradients(ys = D_loss, xs = D_vars) # gradient of D_loss
            G_grad = tf.gradients(ys = G_loss,xs = G_vars) # gradient of G_loss
            D_grad_sq = sum([tf.reduce_sum(tf.square(g)) for g in D_grad])
            G_grad_sq = sum([tf.reduce_sum(tf.square(g)) for g in G_grad])
            L_reg = D_grad_sq + G_grad_sq


            D_loss += gamma*L_reg
            # G_loss is used for the inpainting process, in-painting does not need the regularization term
            G_loss_train = G_loss + gamma*L_reg


            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/D/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/G/')

            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.RMSPropOptimizer(learning_rate=self.D_lr).\
                    minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.RMSPropOptimizer(learning_rate=self.G_lr).\
                    minimize(G_loss_train, var_list=G_vars, global_step=global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss_train', G_loss_train),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('D_loss/real', D_loss_real),
                tf.summary.scalar('D_loss/fake', D_loss_fake)
            ])

            # sparse-step summary
            tf.summary.image('G', G, max_outputs=self.FAKE_MAX_OUTPUT)
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
            self.G = G
            self.global_step = global_step
            self.reg_term = gamma*L_reg #For debug purposes
            self.D_fake_prob = D_fake_prob

            # Image In-painting
            self.mask = tf.placeholder(tf.float32, self.shape, name='mask')
            self.lam = 0.003 # Value taken from paper

            # Reduce the difference in the masked part -- TODO : Add weighting term (from paper) to the mask*image product
            self.contextual_loss = tf.reduce_sum(
                tf.contrib.layers.flatten(
                    tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.X))), 1)

            # The reconstructed/completed image must also "fool" the discriminator
            self.perceptual_loss = self.G_loss
            self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
            self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    def _discriminator(self, X, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            net = X
            width = self.shape[0]
            filter_num = 64
            stride = 2
            num_conv_layers = 4
            with slim.arg_scope([slim.conv2d], kernel_size=[5,5], stride=stride, padding='SAME', activation_fn=ops.lrelu,
                normalizer_fn=self.norm_fn, normalizer_params=self.bn_params):
                for layer_num in range(1,num_conv_layers + 1):
                    if layer_num == 1:
                        net = slim.conv2d(net, filter_num, normalizer_fn=None)
                    else:
                        net = slim.conv2d(net,filter_num,normalizer_fn=self.norm_fn)

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
                activation_fn=tf.nn.relu, normalizer_fn=self.norm_fn,normalizer_params = self.bn_params):
                while input_size < (self.shape[0]//stride):
                    net = slim.conv2d_transpose(net, filter_num)
                    expected_shape(net, [input_size*stride, input_size*stride, filter_num])
                    filter_num = filter_num//2
                    input_size = input_size*stride

                net = slim.conv2d_transpose(net,self.shape[2], activation_fn=tf.nn.tanh, normalizer_fn=None)
                expected_shape(net, [self.shape[0], self.shape[1],self.shape[2]])

                return net
