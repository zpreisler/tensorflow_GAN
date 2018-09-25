#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def _parse_fce(file):
    from numpy.random import uniform
    img_str=tf.read_file(file)
    img_decoded=tf.image.decode_png(img_str)
    img_crop=tf.image.central_crop(img_decoded,0.5)
    img_resized=tf.image.resize_images(img_crop,[48,48])
    #img_gray=tf.image.rgb_to_grayscale(img_resized)
    return img_resized

def dataset(batch_size=1):
    from glob import glob

    files=glob('images2/*.png')
    files_dataset=tf.data.Dataset.from_tensor_slices((files))
    files_dataset=files_dataset.map(_parse_fce)

    dataset=files_dataset.repeat().batch(batch_size)
    iterator=tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)

    img_batch=iterator.get_next()
    init_dataset=iterator.make_initializer(dataset)

    return img_batch,init_dataset

def generator(Z):
    with tf.variable_scope("Generator"):
    
        with tf.variable_scope("Input"):
            print(Z)
            dense=tf.layers.dense(inputs=Z,
                    units=8*8*1*4*3,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=1e-1,dtype=tf.float32),
                    bias_initializer=tf.truncated_normal_initializer(stddev=1e-3,dtype=tf.float32),
                    name='Dense')
            print(dense)

            c=tf.reshape(dense,(-1,8,8,3))
            print(c)

            conv_t=tf.layers.conv2d_transpose(inputs=c,
                    filters=3,
                    kernel_size=5,
                    strides=2,
                    padding='same',
                    name="Conv_transpose")
            print(conv_t)

        return conv_t

def gauss_noise(x,std):
    return x+tf.truncated_normal(tf.shape(x),mean=0.0,stddev=std,dtype=tf.float32)

def discriminator(X,reuse=False):
    with tf.variable_scope("Discriminator",reuse=reuse):
        print(X)
        n=gauss_noise(X,std=1e2)
        print(n)
        return n

def Zbatch(n,m):
    from numpy import random
    return random.uniform(0,1,size=[n,m])

def main(argv):
    print("Generative Adversarial Network")
    from numpy import random

    Z=tf.placeholder(tf.float32,[None,512])
    img_batch,init_dataset=dataset()

    """Generator"""
    g=generator(Z)

    """Discriminator"""
    d=discriminator(img_batch)

    dense_kernel,dense_bias=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator/Input/Dense")

    tf.summary.histogram("Dense kernel",dense_kernel)
    tf.summary.histogram("Dense bias",dense_bias)

    tf.summary.image("Generator",g,max_outputs=24)
    tf.summary.image("Discrminator in",img_batch,max_outputs=24)
    tf.summary.image("Discrminator with noise",d,max_outputs=24)

    summaries=tf.summary.merge_all()

    with tf.Session() as session:
        print("Session")

        """Init"""
        tf.global_variables_initializer().run(session=session)
        session.run(init_dataset)

        """Summaries"""
        writer=tf.summary.FileWriter("log",session.graph)

        a,b,log=session.run([g,d,summaries],feed_dict={Z:Zbatch(1,512)})
        writer.add_summary(log)
if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
