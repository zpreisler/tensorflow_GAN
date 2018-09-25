#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def _parse_fce(file):
    from numpy.random import uniform
    img_str=tf.read_file(file)
    img_decoded=tf.image.decode_png(img_str,channels=3)
    img_crop=tf.image.central_crop(img_decoded,0.5)
    img_resized=tf.image.resize_images(img_crop,[48,48])
    return img_resized

def dataset(batch_size=1):
    from glob import glob

    files=glob('/home/zdenek/Projects/tensorflow/patchy_ann/data_4/*.png')
    files_dataset=tf.data.Dataset.from_tensor_slices((files))
    files_dataset=files_dataset.map(_parse_fce)

    dataset=files_dataset.batch(batch_size)
    iterator=tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)

    img_batch=iterator.get_next()
    init_dataset=iterator.make_initializer(dataset)

    return img_batch,init_dataset

def gauss_noise(x,std,shape,name='Noise'):
    with tf.name_scope(name):
        n=tf.truncated_normal(shape=tf.shape(x),mean=0.0,stddev=std,dtype=tf.float32)
        a=tf.add(x,n)
        return tf.reshape(a,shape)

def generator(Z,std):
    with tf.variable_scope("Generator"):
        with tf.variable_scope("Input"):
            print(Z)
            dense=tf.layers.dense(inputs=Z,
                    units=12*12*1*1*3,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=1e-1,dtype=tf.float32),
                    bias_initializer=tf.truncated_normal_initializer(stddev=1e-3,dtype=tf.float32),
                    name='Dense')
            print(dense)

            c=tf.reshape(dense,(-1,12,12,3))
            print(c)

        with tf.variable_scope("Convolution_transpose"):
            convt_1=tf.layers.conv2d_transpose(inputs=c,
                    filters=12,
                    kernel_size=[5,5],
                    strides=[2,2],
                    padding='same',
                    name="convt_1")
            print(convt_1)

        with tf.variable_scope("Convolution_transpose"):
            convt_2=tf.layers.conv2d_transpose(inputs=convt_1,
                    filters=3,
                    kernel_size=[5,5],
                    strides=[2,2],
                    padding='same',
                    name="convt_2")
            print(convt_2)

        with tf.variable_scope("Output"):
            n=gauss_noise(convt_2,shape=(-1,48,48,3),std=std,name="Noise")
            print(n)

        g=tf.tanh(n)

        return g

def discriminator(X,std,reuse=False):
    with tf.variable_scope("Discriminator",reuse=reuse):
        with tf.variable_scope("Input"):
            print(X,X.get_shape())
            X_noise=gauss_noise(X,std=std,shape=(-1,48,48,3),name="Noise")
            print(X_noise,X_noise.get_shape())

        with tf.variable_scope("Convolution"):
            conv_1=tf.layers.conv2d(inputs=X_noise,
                    filters=1,
                    kernel_size=5,
                    strides=[2,2],
                    padding='same',
                    name="conv_1")
            print(conv_1)

            conv_2=tf.layers.conv2d(inputs=conv_1,
                    filters=12,
                    kernel_size=5,
                    strides=[2,2],
                    padding='same',
                    name="conv_2")
            print(conv_2)

        return X_noise,conv_2

def Zbatch(n,m):
    from numpy import random
    return random.uniform(0,1,size=[n,m])

def main(argv):
    print("Generative Adversarial Network")
    from numpy import random

    Z=tf.placeholder(tf.float32,[None,512])
    std=tf.placeholder(tf.float32)
    img_batch,init_dataset=dataset(3)

    """Generator"""
    g=generator(Z,std)

    """Discriminator"""
    d_noise,d=discriminator(img_batch,std)

    """Summaries"""
    all_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for v in all_vars:
        tf.summary.histogram(v.name,v)
        print(v)

    tf.summary.image("Generator",g,max_outputs=24)
    tf.summary.image("Discrminator with noise",d_noise,max_outputs=24)
    tf.summary.image("Discrminator in",img_batch,max_outputs=24)

    summaries=tf.summary.merge_all()

    """Checkpoints"""
    saver=tf.train.Saver()

    with tf.Session() as session:
        print("Session")

        """Init"""
        tf.global_variables_initializer().run(session=session)
        session.run(init_dataset)

        """Summaries"""
        writer=tf.summary.FileWriter("log",session.graph)

        """Learning"""
        for step in range(1):
            print("[%d]"%step)
            a,b,log=session.run([g,d,summaries],feed_dict={Z:Zbatch(2,512),std:3e1})
            writer.add_summary(log,global_step=step)

        saver.save(session,'log/last.ckpt')


if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
