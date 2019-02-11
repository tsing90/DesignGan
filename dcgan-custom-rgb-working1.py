
# coding: utf-8

# # Project

# In[1]:

# get_ipython().magic('matplotlib inline')


import glob, os, time, random
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io
import tensorlayer as tl

from pylab import rcParams
rcParams['figure.figsize'] = 10, 7

## create folders
save_dir = "samples"
checkpoint_dir = "checkpoint"
tl.files.exists_or_mkdir(save_dir)
tl.files.exists_or_mkdir(checkpoint_dir)

batch_size = 128

def prepro(x):
    x = tl.prepro.flip_axis(x, axis=1, is_random=True)
    # x = tl.prepro.rotation(x, rg=16, is_random=True, fill_mode='nearest')
    # x = tl.prepro.imresize(x, size=[int(64*1.2), int(64*1.2)], interp='bicubic', mode=None)
    # x = tl.prepro.crop(x, wrg=64, hrg=64, is_random=True)
    x = tl.prepro.imresize(x, size=[64, 64], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def rescale(x):
    x = tl.prepro.imresize(x, size=[64, 64], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x
###================= PREPARE DATA
def load_dir(path):
    images = []
    for fname in glob.glob(path):
        try:
            image = io.imread(fname).astype(np.float32)
            if len(image.shape) <= 2:
                image = np.swapaxes(np.stack((image,)*3), 0, 2)
            # print('a',type(images))
            image = tl.prepro.imresize(image, size=[64, 64], interp='bicubic', mode=None)
            image = np.asarray(image, dtype=np.float32)
            # image.dtype = np.float32
            # print(type(image))
            # print(image.shape, image.dtype)
            images.append((np.array(image) / 127.5)-1)
        except Exception as e:
            print(fname, e)
    return np.array(images)

image_A = load_dir('leaves_spoons/spoons/*')
image_B = load_dir('leaves_spoons/leaves/*')

print(image_A.shape, image_B.shape)


# def load_image_from_folder(path):
#     path_imgs = tl.files.load_file_list(path=path, regx='\\.jpg', printable=False)
#     path_imgs += tl.files.load_file_list(path=path, regx='\\.JPG', printable=False)
#     # return tl.vis.read_images(path_imgs, path=path, n_threads=10, printable=False)
#     images = []
#     for p in path_imgs:
#         im = tl.vis.read_image(p, path=path)
#         images.append(im)
#
#     # grey-scale to 3 channels
#     for i in range(len(images)):
#         if images[i].ndim == 2:
#             # images[i] = np.stack((images[i],)*3)
#             images[i] = np.repeat(images[i][:,:,np.newaxis], 3, axis=2)
#             # print(images[i].shape)
#             # tl.vis.save_image(images[i], '_t.png')
#     return images
#
# image_A = load_image_from_folder("leaves_spoons/spoons")
# image_B = load_image_from_folder("leaves_spoons/leaves")
#
# for i in range(len(image_A)):
#     image_A[i] = rescale(image_A[i])
# for i in range(len(image_B)):
#     image_B[i] = rescale(image_B[i])
#
# image_A = np.asarray(image_A)
# image_B = np.asarray(image_B)


###================= DEFINE MODEL
def xavier_initializer(shape):
    return tf.random_normal(shape=shape, stddev=1/shape[0])


img_size = 64
colors = 3

# Generator
z_dim = 200 # Latent vector dimension
g_w1_size = 400
g_w2_size = 200
g_out_size = img_size * img_size * colors

# Discriminator
x_size = img_size * img_size * colors
d_w1_size = 400
d_w2_size = 200
d_out_size = colors

t_z = tf.placeholder('float', shape=(None, z_dim))
t_image = tf.placeholder('float', shape=(None, img_size, img_size, 3))

g_weights = {
    'w1': tf.Variable(xavier_initializer(shape=(z_dim, g_w1_size))),
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
    #  x = tf.identity(x)
     #x = bn(x, name='bn1')
     d1_h1 = conv(x, out_channels=64, name='h1')
     #d1_h1 = bn(d1_h1, name='bn2')
     d1_h1 = lrelu(d1_h1)
     d1_h2 = conv(d1_h1, out_channels=128, name='h2')
     #d1_h2 = bn(d1_h2, name='bn3')
     d1_h2 = lrelu(d1_h2)
    #  d1_h2 = tf.reshape(d1_h2, [-1, 128])
     d1_h2 = tl.layers.flatten_reshape(d1_h2, name='f')
    #  d1_h3 = tf.nn.dropout(d1_h2, 0.5)                      # TODO NOT USING DROPOUT
     d1_h3 = dense(d1_h2, out_channels=1, name='h3')
     return d1_h3

def D2(x, reuse=True):
 with tf.variable_scope('D2', reuse=reuse):
     x = tf.reshape(x, [-1, img_size, img_size, colors])
    #  x = tf.identity(x)
     x = tf.image.rgb_to_grayscale(x)
     #x = bn(x, name='bn1')
     d2_h1 = conv(x, out_channels=64, name='h1')
     #d2_h1 = bn(d2_h1, name='bn2')
     d2_h1 = lrelu(d2_h1)
     d2_h2 = conv(d2_h1, out_channels=128, name='h2')
     #d2_h2 = bn(d2_h2, name='bn3')
     d2_h2 = lrelu(d2_h2)
    #  d2_h2 = tf.reshape(d2_h2, [-1, 128])
     d2_h2 = tl.layers.flatten_reshape(d2_h2, name='f')
    #  d2_h3 = tf.nn.dropout(d2_h2, 0.5)                  # TODO NOT USING DROPOUT
     d2_h3 = dense(d2_h2, out_channels=1, name='h3')
     return d2_h3

sample = G(t_z, reuse=False)
d1_real = D1(t_image, reuse=False)
d1_fake = D1(sample, reuse=True)
d2_real = D2(t_image, reuse=False)
d2_fake = D2(sample, reuse=True)
# print(sample)   # (128, 64, 64, 3)
# exit()

# import model
# generator = model.G#generator_txt2img_resnet
# discriminator = model.D#discriminator_dropout_nobn
#
# ## Real A, B
# n_d_A, d1_real = discriminator(t_image, is_train=True, reuse=False, name='D1', rgb2gray=True)
# n_d_B, d2_real = discriminator(t_image, is_train=True, reuse=False, name='D2')
#
# ## G for A -> A + B
# n_g = generator(t_z, is_train=True, reuse=False, name='G', batch_size=batch_size)
# sample = n_g.outputs
# # print(sample)
# # exit()
# _, d1_fake = discriminator(n_g.outputs, is_train=True, reuse=True, name='D1', rgb2gray=True)
# _, d2_fake = discriminator(n_g.outputs, is_train=True, reuse=True, name='D2')

# 问题是模型或数据

###================= DEFINE LOSS AND TRAIN OPS
# def x_entropy(x, y):
#     return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

g_loss_A_fake = tl.cost.sigmoid_cross_entropy(d1_fake, tf.ones_like(d1_fake), name='gA')
# g_loss_A_fake = tf.reduce_mean(x_entropy(d1_fake, tf.ones_like(d1_fake)))
g_loss_B_fake = tl.cost.sigmoid_cross_entropy(d2_fake, tf.ones_like(d2_fake), name='gB')
# g_loss_B_fake = tf.reduce_mean(x_entropy(d2_fake, tf.ones_like(d2_fake)))

d_loss_A_real = tl.cost.sigmoid_cross_entropy(d1_real, tf.ones_like(d1_real)-0.1, name='drealA')
# d_loss_A_real = tf.reduce_mean(x_entropy(d1_real, tf.ones_like(d1_real)-0.1))
d_loss_A_fake = tl.cost.sigmoid_cross_entropy(d1_fake, tf.zeros_like(d1_fake), name='dfakeA')
# d_loss_A_fake = tf.reduce_mean(x_entropy(d1_fake, tf.zeros_like(d1_fake)))
d_loss_A = d_loss_A_real + d_loss_A_fake

d_loss_B_real = tl.cost.sigmoid_cross_entropy(d2_real, tf.ones_like(d2_real)-0.1, name='drealB')
# d_loss_B_real = tf.reduce_mean(x_entropy(d2_real, tf.ones_like(d2_real)-0.1))
d_loss_B_fake = tl.cost.sigmoid_cross_entropy(d2_fake, tf.zeros_like(d2_fake), name='dfakeB')
# d_loss_B_fake = tf.reduce_mean(x_entropy(d2_fake, tf.zeros_like(d2_fake)))
d_loss_B = d_loss_B_real + d_loss_B_fake

# params = tf.trainable_variables()
# d_vars_A = [v for v in params if v.name.startswith('D1/')]
# d_vars_B = [v for v in params if v.name.startswith('D2/')]
# g_vars = [v for v in params if v.name.startswith('G/')]
g_vars = tl.layers.get_variables_with_name('G/', True, True)
d_vars_A = tl.layers.get_variables_with_name('D1/', True, True)
d_vars_B = tl.layers.get_variables_with_name('D2/', True, True)

# lr_init = 0.001
# beta1 = 0.9
lr_init = 0.0002
beta1 = 0.5
# decay_every = 100
# lr_decay = 0.5

with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(lr_init, trainable=False)
g_optim_A = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss_A_fake, var_list=g_vars)
g_optim_B = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss_B_fake, var_list=g_vars)

d_optim_A = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss_A, var_list=d_vars_A)
d_optim_B = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss_B, var_list=d_vars_B)

###================= TRAIN
def generate_z(n=1):
    return np.random.normal(scale=0.1, size=(n, z_dim))

n_step = 500000
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)
for i in range(n_step):
    step_time = time.time()
    # update D
    _, errDA = sess.run([d_optim_A, d_loss_A], feed_dict={
        t_image: image_A[np.random.choice(range(len(image_A)), batch_size)],
        t_z: generate_z(batch_size),
    })
    _, errDB = sess.run([d_optim_B, d_loss_B], feed_dict={
        t_image: image_B[np.random.choice(range(len(image_B)), batch_size)],
        t_z: generate_z(batch_size),
    })

    # update G
    _, errGA = sess.run([g_optim_A, g_loss_A_fake], feed_dict={
        t_z: generate_z(batch_size)
    })

    _, errGB = sess.run([g_optim_B, g_loss_B_fake], feed_dict={
        t_z: generate_z(batch_size)
    })

    print("Step [%2d/%2d] time: %.4fs dA:%.5f dB:%.5f gA:%.5f gB:%.5f" % (i, n_step, time.time()-step_time, errDA, errDB, errGA, errGB))
    if (i % 200 == 0):
        images = sess.run(sample, feed_dict={t_z:generate_z(batch_size)})
        tl.vis.save_images(images[0:32], [4, 8], save_dir+'/%d_test.png' % i)

#     # if (epoch != 0) and (epoch % 10 == 0):
#         # tl.files.save_npz(n_g.all_params, name=checkpoint_dir+'/g.npz', sess=sess)
#         # tl.files.save_npz(n_d_A.all_params, name=checkpoint_dir+'/dA.npz', sess=sess)
#         # tl.files.save_npz(n_d_B.all_params, name=checkpoint_dir+'/dB.npz', sess=sess)
