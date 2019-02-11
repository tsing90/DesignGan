
# coding: utf-8

# # Project

# In[1]:

import glob, os, time, random
import tensorflow as tf
import tensorlayer as tl
import numpy as np
#from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io
# from pylab import rcParams
# rcParams['figure.figsize'] = 10, 7


## create folders
save_dir = "samples"
checkpoint_dir = "checkpoint"
tl.files.exists_or_mkdir(save_dir)
tl.files.exists_or_mkdir(checkpoint_dir)

#labels = mnist.train.labels
batch_size = 128

def load_dir(path):
    images = []
    for fname in glob.glob(path):
        try:
            image = io.imread(fname).astype(np.float32)
            if len(image.shape) <= 2:
                image = np.swapaxes(np.stack((image,)*3), 0, 2)
            image = tl.prepro.imresize(image, size=[64, 64], interp='bicubic', mode=None)
            image = np.asarray(image, dtype=np.float32)
            images.append((np.array(image) / 127.5)-1)
        except Exception as e:
            print(fname, e)
    return np.array(images)

spoons = load_dir('leaves_spoons/spoons/*')
leaves = load_dir('leaves_spoons/leaves/*')

print(leaves.shape, spoons.shape)



def xavier_initializer(shape):
    return tf.random_normal(shape=shape, stddev=1/shape[0])

img_size = 64
colors = 3

# Generator
z_size = 200 # Latent vector dimension
g_w1_size = 400
g_w2_size = 200
g_out_size = img_size * img_size * colors

# Discriminator
x_size = img_size * img_size * colors
d_w1_size = 400
d_w2_size = 200
d_out_size = colors


z = tf.placeholder('float', shape=(None, z_size))
X = tf.placeholder('float', shape=(None, x_size))


# ## Weights

# In[7]:

g_weights = {
    'w1': tf.Variable(xavier_initializer(shape=(z_size, g_w1_size))),
    'b1': tf.Variable(tf.zeros(shape=[g_w1_size])),
    'w2': tf.Variable(xavier_initializer(shape=(g_w1_size, g_w2_size))),
    'b2': tf.Variable(tf.zeros(shape=[g_w2_size])),
    'out': tf.Variable(xavier_initializer(shape=(g_w2_size, g_out_size))),
    'b3': tf.Variable(tf.zeros(shape=[g_out_size])),
}

d1_weights ={
    'w1': tf.Variable(xavier_initializer(shape=(x_size, d_w1_size))),
    'b1': tf.Variable(tf.zeros(shape=[d_w1_size])),
    'w2': tf.Variable(xavier_initializer(shape=(d_w1_size, d_w2_size))),
    'b2': tf.Variable(tf.zeros(shape=[d_w2_size])),
    'out': tf.Variable(xavier_initializer(shape=(d_w2_size, d_out_size))),
    'b3': tf.Variable(tf.zeros(shape=[d_out_size])),
}

d2_weights ={
    'w1': tf.Variable(xavier_initializer(shape=(x_size, d_w1_size))),
    'b1': tf.Variable(tf.zeros(shape=[d_w1_size])),
    'w2': tf.Variable(xavier_initializer(shape=(d_w1_size, d_w2_size))),
    'b2': tf.Variable(tf.zeros(shape=[d_w2_size])),
    'out': tf.Variable(xavier_initializer(shape=(d_w2_size, d_out_size))),
    'b3': tf.Variable(tf.zeros(shape=[d_out_size])),
}


# ## Models
#
# ### Layers

# In[8]:

def conv(x, out_channels, kernel_size=5, stride=2, name='project'):
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        kernel = tf.get_variable('conv_kernel', [kernel_size, kernel_size, x_shape[-1], out_channels],
                     initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('conv_bias', [out_channels], initializer=tf.ones_initializer())

        return tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding='SAME') + bias

def deconv(x, out_channels, kernel_size=5, stride=2, name='project'):
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        kernel = tf.get_variable('deconv_kernel', [kernel_size, kernel_size, out_channels, x_shape[-1]],
             initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('deconv_bias', [out_channels], initializer=tf.ones_initializer())
        output_shape = [batch_size, x_shape[1] * stride, x_shape[2] * stride, out_channels]

        return tf.nn.conv2d_transpose(x, kernel, output_shape, [1, stride, stride, 1]) + bias

def dense(x, out_channels, name='project'):
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        W = tf.get_variable('dense_w', [x_shape[1], out_channels],
             initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('dense_bias', [out_channels], initializer=tf.ones_initializer())
        return tf.matmul(x, W) + bias

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def bn(x, name='project'):
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        beta = tf.get_variable('BnBeta', [x_shape[-1]],
                            initializer=tf.zeros_initializer())
        gamma = tf.get_variable('BnGamma', [x_shape[-1]],
                            initializer=tf.ones_initializer())
        mean, var = tf.nn.moments(x, [0])
        return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)


# In[9]:


def G(z, reuse=True):
 with tf.variable_scope('G', reuse=reuse):
     g_h1 = dense(z, out_channels=1024, name='h1')
     g_h1 = bn(g_h1, name='bn1')
     g_h1 = tf.nn.relu(g_h1)
     g_h2 = dense(g_h1, out_channels=int(img_size/4)*int(img_size/4)*128, name='h2')
     g_h2 = bn(g_h2, name='bn2')
     g_h2 = tf.nn.relu(g_h2)
     g_h3 = tf.reshape(g_h2, [-1, int(img_size/4), int(img_size/4), 128])
     g_h3 = deconv(g_h3, out_channels=64, name='h3')
     g_h3 = bn(g_h3, name='bn3')
     g_h3 = tf.nn.relu(g_h3)
     g_h4 = deconv(g_h3, out_channels=colors, name='h4')
    #  g_h4 = tf.reshape(g_h4, [-1, img_size * img_size, colors])
     return tf.nn.tanh(g_h4)

def D1(x, reuse=True):
 with tf.variable_scope('D1', reuse=reuse):
     x = tf.reshape(x, [-1, img_size, img_size, colors])
     #x = bn(x, name='bn1')
     d1_h1 = conv(x, out_channels=64, name='h1')
     #d1_h1 = bn(d1_h1, name='bn2')
     d1_h1 = lrelu(d1_h1)
     d1_h2 = conv(d1_h1, out_channels=128, name='h2')
     #d1_h2 = bn(d1_h2, name='bn3')
     d1_h2 = lrelu(d1_h2)
     d1_h2 = tl.layers.flatten_reshape(d1_h2, name='f')
    #  d1_h2 = tf.reshape(d1_h2, [-1, 128])
    #  d1_h3 = tf.nn.dropout(d1_h2, 0.5)
     d1_h3 = dense(d1_h2, out_channels=colors, name='h3')
     return d1_h3

def D2(x, reuse=True):
 with tf.variable_scope('D2', reuse=reuse):
     x = tf.reshape(x, [-1, img_size, img_size, colors])
     x = tf.image.rgb_to_grayscale(x)
     #x = bn(x, name='bn1')
     d2_h1 = conv(x, out_channels=64, name='h1')
     #d2_h1 = bn(d2_h1, name='bn2')
     d2_h1 = lrelu(d2_h1)
     d2_h2 = conv(d2_h1, out_channels=128, name='h2')
     #d2_h2 = bn(d2_h2, name='bn3')
     d2_h2 = lrelu(d2_h2)
     d2_h2 = tf.reshape(d2_h2, [-1, 128])
    #  d2_h3 = tf.nn.dropout(d2_h2, 0.5)
     d2_h2 = tl.layers.flatten_reshape(d2_h2, name='f')
     d2_h3 = dense(d2_h2, out_channels=colors, name='h3')
     return d2_h3

def generate_z(n=1):
    return np.random.normal(scale=0.1, size=(n, z_size))


G(z, reuse=False)
D1(spoons[0], reuse=False)
D2(spoons[0], reuse=False)

sample = G(z) # To be called during session


def x_entropy(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)


d1_real = D1(X)
d1_fake = D1(G(z))
d2_real = D2(X)
d2_fake = D2(G(z))

G1_objective = tf.reduce_mean(x_entropy(d1_fake, tf.ones_like(d1_fake)))
G2_objective = tf.reduce_mean(x_entropy(d2_fake, tf.ones_like(d2_fake)))

D1_obj_real = tf.reduce_mean(x_entropy(d1_real, tf.ones_like(d1_real)-0.1))
D1_obj_fake = tf.reduce_mean(x_entropy(d1_fake, tf.zeros_like(d1_fake)))
D1_objective = D1_obj_real + D1_obj_fake

D2_obj_real = tf.reduce_mean(x_entropy(d2_real, tf.ones_like(d2_real)-0.1))
D2_obj_fake = tf.reduce_mean(x_entropy(d2_fake, tf.zeros_like(d2_fake)))
D2_objective = D2_obj_real + D2_obj_fake

D1_fake_balance = tf.reduce_mean((x_entropy(d1_fake, tf.zeros_like(d1_fake)) -
                                  x_entropy(d1_fake, tf.ones_like(d1_fake)))**2)
D2_fake_balance = tf.reduce_mean((x_entropy(d2_fake, tf.zeros_like(d2_fake)) -
                                  x_entropy(d2_fake, tf.ones_like(d2_fake)))**2)


params = tf.trainable_variables()
d1_params = [v for v in params if v.name.startswith('D1/')]
d2_params = [v for v in params if v.name.startswith('D2/')]
g_params = [v for v in params if v.name.startswith('G/')]


G1_opt = tf.train.AdamOptimizer().minimize(
    G1_objective, var_list=g_params)
G2_opt = tf.train.AdamOptimizer().minimize(
    G2_objective, var_list=g_params)

D1_real_opt = tf.train.AdamOptimizer().minimize(
    D1_obj_real, var_list=d1_params)
D2_real_opt = tf.train.AdamOptimizer().minimize(
    D2_obj_real, var_list=d2_params)
D1_fake_opt = tf.train.AdamOptimizer().minimize(
    D1_obj_fake, var_list=d1_params)
D2_fake_opt = tf.train.AdamOptimizer().minimize(
    D2_obj_fake, var_list=d2_params)

# D1_balance_opt = tf.train.AdamOptimizer().minimize(
#     D1_fake_balance, var_list=d1_params)
# D2_balance_opt = tf.train.AdamOptimizer().minimize(
#     D2_fake_balance, var_list=d2_params)

D1_opt = tf.train.AdamOptimizer().minimize(
    D1_objective, var_list=d1_params)
D2_opt = tf.train.AdamOptimizer().minimize(
    D2_objective, var_list=d2_params)


# ## Training

# Hyper-parameters
# import random
# import warnings
# import matplotlib.gridspec as gridspec
# from IPython.display import clear_output
# warnings.simplefilter('error', UserWarning)

n_step = 500000
images1 = leaves
images2 = spoons

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_step):

        _, errDA = sess.run([D1_opt, D1_objective], feed_dict={
            X: images1[np.random.choice(range(len(images1)), batch_size)].reshape(batch_size, img_size * img_size * colors),
            z: generate_z(batch_size),
        })
        _, errDB = sess.run([D2_opt, D2_objective], feed_dict={
            X: images2[np.random.choice(range(len(images2)), batch_size)].reshape(batch_size, img_size * img_size * colors),
            z: generate_z(batch_size),
        })

        # G
        _, errGA = sess.run([G1_opt, G1_objective], feed_dict={
            z: generate_z(batch_size)
        })

        _, errGB = sess.run([G2_opt, G2_objective], feed_dict={
            z: generate_z(batch_size)
        })

        print("Step: [%2d/%2d] dA:%.5f dB:%.5f gA:%.5f gB:%.5f" % (i, n_step, errDA, errDB, errGA, errGB))
        if (i % 200 ==0):#((i / epochs) % 0.01 == 0):
            print(i, costs[i])
            images = sess.run(sample, feed_dict={z:generate_z(batch_size)})
            tl.vis.save_images(images[0:32], [4, 8], save_dir+'/%d_test.png' % i)
