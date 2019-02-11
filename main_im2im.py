#! /usr/bin/python
# -*- coding: utf8 -*-

"""


"""
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time, random, model

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


## create folders
save_dir = "samples"
checkpoint_dir = "checkpoint"
tl.files.exists_or_mkdir(save_dir)
tl.files.exists_or_mkdir(checkpoint_dir)

def train():
    ## hyperparameters
    batch_size = 32 #64
    # lr_init = 0.0001 # 0.0002
    # beta1 = 0.9     # 0.5
    lr_init = 0.0002
    beta1 = 0.5
    n_epoch = 400
    decay_every = 100
    lr_decay = 0.5
    ni = int(np.sqrt(batch_size))

    ## load data
    # im_train_A, im_train_B, im_test_A, im_test_B = tl.files.load_cyclegan_dataset(filename='summer2winter_yosemite')
    # im_train = im_train_A + im_train_B

    def load_image_from_folder(path):
        path_imgs = tl.files.load_file_list(path=path, regx='\\.jpg', printable=False)
        path_imgs += tl.files.load_file_list(path=path, regx='\\.JPG', printable=False)
        # return tl.vis.read_images(path_imgs, path=path, n_threads=10, printable=False)
        images = []
        for p in path_imgs:
            images.append(tl.vis.read_image(p, path=path))

        # grey-scale to 3 channels
        for i in range(len(images)):
            if images[i].ndim == 2:
                # images[i] = np.stack((images[i],)*3)
                images[i] = np.repeat(images[i][:,:,np.newaxis], 3, axis=2)
                # print(images[i].shape)
                # tl.vis.save_image(images[i], '_t.png')
        return np.asarray(images)

    im_A = load_image_from_folder("leaves_spoons/spoons")
    im_B = load_image_from_folder("leaves_spoons/leaves")
    n_A = len(im_A); n_B = len(im_B)
    im_train_A, im_test_A = im_A[:int(n_A*0.7)], im_A[int(n_A*0.7):]
    im_train_B, im_test_B = im_B[:int(n_B*0.7)], im_B[int(n_B*0.7):]
    # print(n_A, n_B)
    # print(im_A.shape, im_B.shape)
    # print(im_train_A.shape)
    # exit()

    ## sample/seed images
    sample_imgs_A = np.asarray(im_test_A[0:batch_size])
    sample_imgs_A = tl.prepro.threading_data(sample_imgs_A, fn=rescale)
    tl.vis.save_images(sample_imgs_A, [4, 8], save_dir+'/_sample_A.png')
    sample_imgs_B = np.asarray(im_test_B[0:batch_size])
    sample_imgs_B = tl.prepro.threading_data(sample_imgs_B, fn=rescale)
    tl.vis.save_images(sample_imgs_B, [4, 8], save_dir+'/_sample_B.png')

    ## A, B inputs
    t_image_A = tf.placeholder(tf.float32, [None, 64, 64, 3], 'imageA')
    t_image_B = tf.placeholder(tf.float32, [None, 64, 64, 3], 'imageB')

    encoder_generator = model.encoder_generator
    discriminator = model.discriminator

    ## Real A, B
    n_d_A, d_real_A_logits = discriminator(t_image_A, is_train=True, reuse=False, name='disA', rgb2gray=True) # spoons
    n_d_B, d_real_B_logits = discriminator(t_image_B, is_train=True, reuse=False, name='disB')

    ## G for A -> A + B
    n_g = encoder_generator(t_image_A, is_train=True, reuse=False)
    _, d_fake_A_logits = discriminator(n_g.outputs, is_train=True, reuse=True, name='disA', rgb2gray=True)
    _, d_fake_B_logits = discriminator(n_g.outputs, is_train=True, reuse=True, name='disB')

    ## Loss
    # d_A and d_B
    noise = tf.random_uniform(shape=(batch_size, 1, 1, 1), minval=0.1, maxval=0.4) # label smoothing regularization
    d_loss_A_real = tl.cost.sigmoid_cross_entropy(d_real_A_logits, tf.ones_like(d_real_A_logits)-noise, name='drealA')
    d_loss_B_real = tl.cost.sigmoid_cross_entropy(d_real_B_logits, tf.ones_like(d_real_B_logits)-noise, name='drealB')
    d_loss_A_fake = tl.cost.sigmoid_cross_entropy(d_fake_A_logits, tf.zeros_like(d_fake_A_logits), name='dfakeA')
    d_loss_B_fake = tl.cost.sigmoid_cross_entropy(d_fake_B_logits, tf.zeros_like(d_fake_B_logits), name='dfakeB')
    d_loss_A = d_loss_A_real + d_loss_A_fake
    d_loss_B = d_loss_B_real + d_loss_B_fake

    # g cheat 2 dis
    g_loss_A_fake = tl.cost.sigmoid_cross_entropy(d_fake_A_logits, tf.ones_like(d_fake_A_logits), name='gfakeA')
    g_loss_B_fake = tl.cost.sigmoid_cross_entropy(d_fake_B_logits, tf.ones_like(d_fake_B_logits), name='gfakeB')
    g_loss = g_loss_A_fake + g_loss_B_fake

    ## Train ops
    g_vars = tl.layers.get_variables_with_name('encoder_generator', True, True)
    d_vars_A = tl.layers.get_variables_with_name('disA', True, True)
    d_vars_B = tl.layers.get_variables_with_name('disB', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    # g_optim_mse = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    g_optim_A = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss_A_fake, var_list=g_vars)
    g_optim_B = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss_B_fake, var_list=g_vars)
    d_optim_A = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss_A, var_list=d_vars_A)
    d_optim_B = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss_B, var_list=d_vars_B)

    ## init params
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    ## load model
    tl.files.load_and_assign_npz(sess, checkpoint_dir+'/g.npz', n_g)
    tl.files.load_and_assign_npz(sess, checkpoint_dir+'/dA.npz', n_d_A)
    tl.files.load_and_assign_npz(sess, checkpoint_dir+'/dB.npz', n_d_B)

    ## save log
    log = open('results.txt', 'w')

    ## Start training
    for epoch in range(0, n_epoch+1):
        start_time = time.time()
        # change lr
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            _log = " ** new learning rate: %f" % (lr_init * new_lr_decay)
            print(_log)
            log.write(_log)
        # loop for an epoch
        random.shuffle(im_train_A)
        random.shuffle(im_train_B)
        total_errDA, total_errDB, total_errG, n_iter = 0, 0, 0, 0
        for idx in range(0, n_A, batch_size):
            step_time = time.time()
            b_images_A = tl.prepro.threading_data(
                    im_train_A[idx : idx + batch_size],
                    fn=prepro)
            if len(b_images_A) != batch_size:
                continue
            # print(b_images_A.shape, b_images_A.min(), b_images_A.max())
            # tl.vis.save_images(b_images_A, [4, 8], image_path='_b_images.png')
            b_images_B = tl.prepro.threading_data(
                    im_train_B[idx : idx + batch_size],
                    fn=prepro)
            if len(b_images_B) != batch_size:
                continue

            # _, _, errDA, errDB = sess.run([d_optim_A, d_optim_B, d_loss_A, d_loss_B],
            #     feed_dict={t_image_A: b_images_A, t_image_B: b_images_B})

            # print(b_images_A.shape, b_images_B.shape)

            _, errDA = sess.run([d_optim_A, d_loss_A],
                feed_dict={t_image_A: b_images_A})

            _, errDB = sess.run([d_optim_B, d_loss_B],
                feed_dict={t_image_A: b_images_A, t_image_B: b_images_B})
            # errDB = 0

            # _, errG = sess.run([g_optim, g_loss],
            #     feed_dict={t_image_A: b_images_A})

            _, errGA = sess.run([g_optim_A, g_loss_A_fake],
                feed_dict={t_image_A: b_images_A})
            _, errGB = sess.run([g_optim_B, g_loss_B_fake],
                feed_dict={t_image_A: b_images_A})
            errG = errGA + errGB

            # print("Epoch: [%2d/%2d] [%4d] time: %4.4fs d_loss_A: %.5f d_loss_B: %.5f g_loss: %.5f " \
            #         % (epoch, n_epoch, n_iter, time.time() - step_time, errDA, errDB, errG))
            print("Epoch: [%2d/%2d] [%4d] time: %4.4fs d_loss_A: %.5f d_loss_B: %.5f g_loss_A: %.5f g_loss_B: %.5f" \
                    % (epoch, n_epoch, n_iter, time.time() - step_time, errDA, errDB, errGA, errGB))

            total_errDA +=errDA ; total_errDB +=errDB; total_errG += errG; n_iter += 1

        _log = "[*] Epoch: [%2d/%2d] d_loss_A: %.5f d_loss_B: %.5f g_loss: %.5f\n" % (epoch, n_epoch, total_errDA/n_iter, total_errDB/n_iter, total_errG/n_iter)
        print(_log)
        log.write(_log)

        # save result and model
        if (epoch != 0) and (epoch % 5 == 0):
        # if epoch % 5 == 0:
            o = sess.run(n_g.outputs, {t_image_A: sample_imgs_A})
            tl.vis.save_images(o, [4, 8], save_dir+'/%d_test.png' % epoch)

        # if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(n_g.all_params, name=checkpoint_dir+'/g.npz', sess=sess)
            tl.files.save_npz(n_d_A.all_params, name=checkpoint_dir+'/dA.npz', sess=sess)
            tl.files.save_npz(n_d_B.all_params, name=checkpoint_dir+'/dB.npz', sess=sess)


# def test():
#     ## data
#     image_A = tl.vis.read_image('2010-09-07 12:23:20.jpg', path='data/cyclegan/summer2winter_yosemite/testA/')
#     image_B = tl.vis.read_image('2006-04-11 11:21:20.jpg', path='data/cyclegan/summer2winter_yosemite/testB/')
#
#     image_B = image_B[:,:,0:3]
#     # print(image_B.shape)
#     # exit()
#     image_A = rescale(image_A)
#     image_B = rescale(image_B)
#
#
#     ## A, B inputs
#     t_image_A = tf.placeholder(tf.float32, [None, 64, 64, 3], 'imageA')
#     t_image_B = tf.placeholder(tf.float32, [None, 64, 64, 3], 'imageB')
#
#     ## model
#     encoder_generator = model.encoder_generator
#     n = encoder_generator(t_image_A, is_train=True, reuse=False)
#     n_emb_A = encoder_generator(t_image_A, is_train=True, reuse=True, debug="return_cnn_embed")
#     n_emb_B = encoder_generator(t_image_B, is_train=True, reuse=True, debug="return_cnn_embed")
#     size = n_emb_B.outputs.get_shape().as_list()
#     t_interp_emb = tf.placeholder('float32', [None, size[1], size[2], size[3]], name='emb')
#     n_g_AB = encoder_generator(t_interp_emb, is_train=True, reuse=True, debug="input_cnn_emb")
#
#     ## load model
#     sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
#     tl.layers.initialize_global_variables(sess)
#     tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g.npz', network=n)
#
#     ## test
#     emb_A = sess.run(n_emb_A.outputs, {t_image_A: [image_A]})
#     emb_B = sess.run(n_emb_B.outputs, {t_image_B: [image_B]})
#
#     images = []
#     for i in np.arange(0, 1.1, 0.1):
#         print(i)
#         emb = i * emb_A + (1 - i) * emb_B
#         o = sess.run(n_g_AB.outputs, {t_interp_emb : emb})
#         images.append(o[0])
#         # tl.vis.save_image(o[0], save_dir +'/_inter_%f.png' % i)
#     tl.vis.save_image(image_A, save_dir +'/_inter_A.png')
#     tl.vis.save_image(image_B, save_dir +'/_inter_B.png')
#     tl.vis.save_images(np.asarray(images), [1, len(images)], save_dir+'/_inter.png')


if __name__ == '__main__':
    train()
    # test()
