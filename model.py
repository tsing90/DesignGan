#! /usr/bin/python
# -*- coding: utf8 -*-
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os

# modified from Semantic Image Systhesis via Adversarial Learning
def encoder_generator(inputs, is_train=True, reuse=False, batch_size=None, debug=None):
    """ 64x64 --> 64x64 """
    gf_dim = 64#128
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("encoder_generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        if debug == "input_cnn_emb":                        # for image interpolation
            net_h2 = InputLayer(inputs, name='g/images')    # for image interpolation
        else:                                               # for image interpolation
            net_in = InputLayer(inputs, name='g/images')    # for image interpolation
            ## downsampling
            net_h0 = Conv2d(net_in, gf_dim, (3, 3), (1, 1), act=tf.nn.relu,
                    padding='SAME', W_init=w_init, name='g_h0/conv2d')
            net_h1 = Conv2d(net_h0, gf_dim*2, (4, 4), (2, 2), act=None,
                    padding='SAME', W_init=w_init, b_init=b_init, name='g_h1/conv2d')
            net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu,
                    is_train=is_train, gamma_init=gamma_init, name='g_h1/batchnorm')
            net_h2 = Conv2d(net_h1, gf_dim*4, (4, 4), (2, 2), act=None,
                    padding='SAME', W_init=w_init, b_init=b_init, name='g_h2/conv2d')
            net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu,
                    is_train=is_train, gamma_init=gamma_init, name='g_h2/batchnorm')   # DH add

        if debug == "return_cnn_embed": # for image interpolation
            return net_h2               # for image interpolation

        for i in range(16):
            net = Conv2d(net_h2, gf_dim*4, (3, 3), (1, 1),
                   padding='SAME', W_init=w_init, b_init=b_init, name='g_residual{}/conv2d_1'.format(i))
            net = BatchNormLayer(net, act=tf.nn.relu,
                   is_train=is_train, gamma_init=gamma_init, name='g_residual{}/batch_norm_1'.format(i))
            net = Conv2d(net, gf_dim*4, (3, 3), (1, 1),
                   padding='SAME', W_init=w_init, b_init=b_init, name='g_residual{}/conv2d_2'.format(i))
            net = BatchNormLayer(net, #act=tf.nn.relu,
                   is_train=is_train, gamma_init=gamma_init, name='g_residual{}/batch_norm_2'.format(i))
            net_h2 = ElementwiseLayer(layer=[net_h2, net], combine_fn=tf.add, name='g_residual{}/add'.format(i))
            net_h2.outputs = tf.nn.relu(net_h2.outputs)

        # Note: you can use DeConv2d instead of UpSampling2dLayer and Conv2d
        # net_h3 = DeConv2d(net_h2, gf_dim*2, (4, 4), out_size=(32, 32), strides=(2, 2),    # 16x16--32x32
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_up/decon2d_1')
        net_h3 = UpSampling2dLayer(net_h2, size=[32, 32], is_scale=False, method=1, align_corners=False, name='g_up/upsample2d_1')
        net_h3 = Conv2d(net_h3, gf_dim*2, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, b_init=b_init, name='g_up/conv2d_1')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_up/batch_norm_1')

        # net_h4 = DeConv2d(net_h3, gf_dim, (4, 4), out_size=(64, 64), strides=(2, 2),    # 32x32--64x64
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_up/decon2d_2')
        net_h4 = UpSampling2dLayer(net_h3, size=[64, 64], is_scale=False, method=1, align_corners=False, name='g_up/upsample2d_2')
        net_h4 = Conv2d(net_h4, gf_dim, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, b_init=b_init, name='g_up/conv2d_2')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_up/batch_norm_2')

        ## down to 3 channels
        network = Conv2d(net_h4, 3, (3, 3), (1, 1),
               padding='SAME', W_init=w_init, name='g_out/conv2d')

        logits = network.outputs
        network.outputs = tf.nn.tanh(network.outputs)
        # exit(network.outputs)
    return network


def discriminator(input_images, t_txt=None, is_train=True, reuse=False, name="discriminator", rgb2gray=False):
    """ 64x64 --> real/fake """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    # Discriminator with ResNet : line 197 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = 64 # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        if rgb2gray:
            input_images = tf.image.rgb_to_grayscale(input_images)
        net_in = InputLayer(input_images, name='d_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d_h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h1/conv2d')
        # net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h2/conv2d')
        # net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h3/conv2d')
        # net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')

        net = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='d_h4_res/conv2d')
        # net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d2')
        # net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm2')
        net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d3')
        # net = BatchNormLayer(net, #act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm3')
        net_h4 = ElementwiseLayer(layer=[net_h3, net], combine_fn=tf.add, name='d_h4/add')
        net_h4.outputs = tl.act.lrelu(net_h4.outputs, 0.2)

        if t_txt is not None:
            net_txt = InputLayer(t_txt, name='d_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, name='d_reduce_txt/dense')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim2')
            net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='d_txt/tile')
            net_h4_concat = ConcatLayer([net_h4, net_txt], concat_dim=3, name='d_h3_concat')
            # 243 (ndf*8 + 128 or 256) x 4 x 4
            net_h4 = Conv2d(net_h4_concat, df_dim*8, (1, 1), (1, 1),
                    padding='VALID', W_init=w_init, b_init=None, name='d_h3/conv2d_2')
            net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2')

        net_ho = Conv2d(net_h4, 1, (s16, s16), (s16, s16), padding='VALID', W_init=w_init, name='d_ho/conv2d')
        # 1 x 1 x 1
        # net_ho = FlattenLayer(net_h4, name='d_ho/flatten')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
    return net_ho, logits


def discriminator_dropout_nobn(input_images, t_txt=None, is_train=True, reuse=False, name="discriminator"):
    """ 64x64 --> real/fake """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    # Discriminator with ResNet : line 197 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = 64 # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='d_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d_h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h1/conv2d')
        net_h1.outputs = tl.act.lrelu(net_h1.outputs, 0.2)
        # net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h2/conv2d')
        net_h2.outputs = tl.act.lrelu(net_h2.outputs, 0.2)
        # net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h3/conv2d')
        net_h3.outputs = tl.act.lrelu(net_h3.outputs, 0.2)
        # net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')

        net = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='d_h4_res/conv2d')
        net.outputs = tl.act.lrelu(net.outputs, 0.2)
        # net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d2')
        net.outputs = tl.act.lrelu(net.outputs, 0.2)
        # net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm2')
        net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d3')
        # net = BatchNormLayer(net, #act=lambda x: tl.act.lrelu(x, 0.2),
        #         is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm3')
        net_h4 = ElementwiseLayer(layer=[net_h3, net], combine_fn=tf.add, name='d_h4/add')
        net_h4.outputs = tl.act.lrelu(net_h4.outputs, 0.2)

        if t_txt is not None:
            net_txt = InputLayer(t_txt, name='d_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, name='d_reduce_txt/dense')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim2')
            net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='d_txt/tile')
            net_h4_concat = ConcatLayer([net_h4, net_txt], concat_dim=3, name='d_h3_concat')
            # 243 (ndf*8 + 128 or 256) x 4 x 4
            net_h4 = Conv2d(net_h4_concat, df_dim*8, (1, 1), (1, 1),
                    padding='VALID', W_init=w_init, b_init=None, name='d_h3/conv2d_2')
            net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2')

        net_h4 = DropoutLayer(net_h4, keep=0.5, is_fix=True, is_train=True, name='DROPOUT')

        net_ho = Conv2d(net_h4, 1, (s16, s16), (s16, s16), padding='VALID', W_init=w_init, name='d_ho/conv2d')
        # 1 x 1 x 1
        # net_ho = FlattenLayer(net_h4, name='d_ho/flatten')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
    return net_ho, logits

def generator_txt2img_resnet(input_z, t_txt=None, is_train=True, reuse=False, batch_size=32):
    """ z + (txt) --> 64x64 """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    s = 64 # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    gf_dim = 128

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_z, name='g_inputz')

        if t_txt is not None:
            net_txt = InputLayer(t_txt, name='g_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, name='g_reduce_text/dense')
            net_in = ConcatLayer([net_in, net_txt], concat_dim=1, name='g_concat_z_txt')

        net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16, act=tf.identity,
                W_init=w_init, b_init=None, name='g_h0/dense')
        net_h0 = BatchNormLayer(net_h0,  #act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')
        net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')

        net = Conv2d(net_h0, gf_dim*2, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1_res/batch_norm')
        net = Conv2d(net, gf_dim*2, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d2')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1_res/batch_norm2')
        net = Conv2d(net, gf_dim*8, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d3')
        net = BatchNormLayer(net, # act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm3')
        net_h1 = ElementwiseLayer(layer=[net_h0, net], combine_fn=tf.add, name='g_h1_res/add')
        net_h1.outputs = tf.nn.relu(net_h1.outputs)

        # Note: you can also use DeConv2d to replace UpSampling2dLayer and Conv2d
        # net_h2 = DeConv2d(net_h1, gf_dim*4, (4, 4), out_size=(s8, s8), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h2/decon2d')
        net_h2 = UpSampling2dLayer(net_h1, size=[s8, s8], is_scale=False, method=1,
                align_corners=False, name='g_h2/upsample2d')
        net_h2 = Conv2d(net_h2, gf_dim*4, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2,# act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h2/batch_norm')

        net = Conv2d(net_h2, gf_dim, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h3_res/batch_norm')
        net = Conv2d(net, gf_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d2')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h3_res/batch_norm2')
        net = Conv2d(net, gf_dim*4, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d3')
        net = BatchNormLayer(net, #act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm3')
        net_h3 = ElementwiseLayer(layer=[net_h2, net], combine_fn=tf.add, name='g_h3/add')
        net_h3.outputs = tf.nn.relu(net_h3.outputs)

        # net_h4 = DeConv2d(net_h3, gf_dim*2, (4, 4), out_size=(s4, s4), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h4/decon2d'),
        net_h4 = UpSampling2dLayer(net_h3, size=[s4, s4], is_scale=False, method=1,
                align_corners=False, name='g_h4/upsample2d')
        net_h4 = Conv2d(net_h4, gf_dim*2, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

        # net_h5 = DeConv2d(net_h4, gf_dim, (4, 4), out_size=(s2, s2), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h5/decon2d')
        net_h5 = UpSampling2dLayer(net_h4, size=[s2, s2], is_scale=False, method=1,
                align_corners=False, name='g_h5/upsample2d')
        net_h5 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h5/batch_norm')

        # net_ho = DeConv2d(net_h5, c_dim, (4, 4), out_size=(s, s), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_ho/decon2d')
        net_ho = UpSampling2dLayer(net_h5, size=[s, s], is_scale=False, method=1,
                align_corners=False, name='g_ho/upsample2d')
        net_ho = Conv2d(net_ho, 3, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, name='g_ho/conv2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)
    return net_ho#, logits



def G(z, is_train=True, reuse=True, batch_size=None, name='generator'):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(z, name='in')
        n = DenseLayer(n, 1024, W_init=w_init, name='h1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn1')
        n = DenseLayer(n, 16*16*128, W_init=w_init, name='h2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn2')
        n = ReshapeLayer(n, [-1, int(64/4), int(64/4), 128], name='res')
        print(n.outputs)
        n = DeConv2d(n, 64, (5, 5), out_size=(int(64/2), int(64/2)), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='dc1')
        print(n.outputs)
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='bn3')
        n = DeConv2d(n, 3, (5, 5), out_size=(64, 64), strides=(2, 2),
                padding='SAME', batch_size=batch_size, act=tf.nn.tanh, W_init=w_init, name='dc2')
        return n

def D(x, is_train=True, reuse=True, name='dis', rgb2gray=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x : tl.act.lrelu(x, 0.2)
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        if rgb2gray:
            x = tf.image.rgb_to_grayscale(x)
        n = InputLayer(x, name='in')
        n = Conv2d(n, 64, (5, 5), (2, 2), padding='SAME', act=lrelu, W_init=w_init, name='h1')
        n = Conv2d(n, 128, (5, 5), (2, 2), padding='SAME', act=lrelu, W_init=w_init, name='h2')
        n = FlattenLayer(n, name='f')
        # n = DropoutLayer(n, keep=0.5, is_fix=True, is_train=True, name='DROPOUT')
        n = DenseLayer(n, 1, W_init=w_init, name='o')
        return n, n.outputs

if __name__ == '__main__':
    # t_image = tf.placeholder(tf.float32, [batch_size, image_size, image_size, c_dim], name='t_image')
    # t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='t_z')
    ## simple biGAN without text
    # G(z)
    # net_g, logits = generator(t_z, batch_size=batch_size)
    # net_g.print_layers()
    # # E(x)
    # net_e = encoder(net_g.outputs)
    # net_e.print_layers()
    # # # # D(x,z)
    # net_d, _ = discriminator(t_image, t_z)
    # net_d.print_layers()

    ## simple biGAN with text
    # t_txt = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='t_txt')
    # # RNN(txt)
    # net_rnn = rnn_embed(t_txt)
    # # G(z, RNN(txt))
    # net_g, logits = generator(t_z, net_rnn.outputs, batch_size=batch_size)
    # net_g.print_layers()
    # # E(x)
    # net_e = encoder(net_g.outputs)
    # net_e.print_layers()
    # # # # D(x,z)
    # net_d, _ = discriminator(t_image, t_z, net_rnn.outputs)
    # net_d.print_layers()

    ## GAN-CLS

    ## (bi)gAN
    # t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='t_z')
    # t_txt = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='t_txt')
    # net_rnn = rnn_embed(t_txt)
    # net_g1, _ = generator_txt2img_resnet(t_z, net_rnn.outputs, batch_size=batch_size)
    # net_g2, _ = stackG_256(net_g1.outputs, net_rnn.outputs, batch_size=batch_size)
    # #
    # image_size = 256
    # t_image = tf.placeholder(tf.float32, [batch_size, image_size, image_size, c_dim], name='t_image')
    # # stackGAN
    # # net_d2, _ = stackD_256(net_g2.outputs, net_rnn.outputs)
    # # bistackGAN
    # net_d2, _ = bistackGAN_D_256(net_g2.outputs, t_z, net_rnn.outputs)
    # net_d2.print_layers()
    # # net_e = encoder_256(t_image, net_rnn.outputs)
    # net_e = encoder_256(t_image, net_rnn.outputs)#, net_rnn.outputs)
    # net_e.print_layers()

    ## image captioning
    # t_image = tf.placeholder(tf.float32, [batch_size, 256, 256, c_dim], name='t_image')
    # t_txt = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='t_txt')
    # # net_cnn = cnn_encoder(t_image)
    #
    # net_vgg = vgg16_cnn(t_image)
    #
    # net_image_caption = image_caption_rnn_look(net_vgg.outputs, t_txt, is_train=False)
    #
    # if not os.path.isfile("checkpoint/vgg16_weights.npz"):
    #     print("Please download vgg16_weights.npz from : http://www.cs.toronto.edu/~frossard/post/vgg16/")
    #     exit()
    # npz = np.load('checkpoint/vgg16_weights.npz')
    #
    # # print(len(net_vgg.all_params))
    # # print(len(sorted( npz.items() )[0:25]))
    # # exit()
    #
    #
    # sess = tf.InteractiveSession()
    # tl.layers.initialize_global_variables(sess)
    #
    # # params = []
    # for idx, val in enumerate(sorted( npz.items() )[0:26]):
    #     print("  Loading %s" % str(val[1].shape))
    #     # params.append(val[1])
    #     net_vgg.all_params[idx].assign(val[1])
    # net_vgg.print_params(False)
    # # tl.files.assign_params(sess, params, net_vgg)

    # image translation 244
    t_image = tf.placeholder(tf.float32, [batch_size, 244, 244, 3], name='t_image')
    t_txt = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='t_txt')

    net_rnn = rnn_embed(t_txt, is_train=False, reuse=False)

    net_fake_image, _, kl_loss = encode_generator_244(t_image,
                    net_rnn.outputs,                              # arbitrary text instead of matching text
                    is_train=True, reuse=False)
    print(net_fake_image.outputs)
    # exit()
    net_d, disc_fake_image_logits = discriminator_txt2img_244(
                    net_fake_image.outputs,
                    net_rnn.outputs,                              # arbitrary text instead of matching text
                    is_train=True, reuse=False)
    print(net_d.outputs)
#
