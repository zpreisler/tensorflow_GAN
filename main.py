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
    return img_resized/255.0

def dataset(batch_size=1):
    from glob import glob

    files=glob('/home/zdenek/Projects/tensorflow/patchy_ann/data_4/*.png')
    files_dataset=tf.data.Dataset.from_tensor_slices((files))
    files_dataset=files_dataset.map(_parse_fce)

    dataset=files_dataset.repeat().batch(batch_size)
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
                    units=16*16*1*1*12,
                    #kernel_initializer=tf.truncated_normal_initializer(stddev=1e-1,dtype=tf.float32),
                    #bias_initializer=tf.truncated_normal_initializer(stddev=1e-3,dtype=tf.float32),
                    #kernel_initializer=tf.ones_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    use_bias=False,
                    name='Dense')
            print(dense)

            #m=tf.reduce_min(dense)
            #t=dense-m
            #d=tf.reduce_max(t)

            zz=tf.reshape(dense,(-1,16,16,3))

            im=tf.image.resize_images(zz,[52,52],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            #z=tf.reshape(Z,(-1,16,16,12))
            print(im)

            convt=tf.layers.conv2d(inputs=im,
                    filters=3,
                    kernel_size=[1,1],
                    strides=[1,1],
                    padding='valid',
                    use_bias=False,
                    kernel_initializer=tf.ones_initializer(),
                    activation=tf.nn.relu,
                    name="c")

            print("convt",convt)
            c=convt

            cc=tf.reshape(c,(-1,52,52,3))

        return cc

            #bnorm0=tf.layers.batch_normalization(c)

        #with tf.variable_scope("Convolution_transpose"):
        #    convt_1=tf.layers.conv2d_transpose(inputs=bnorm0,
        #            filters=48,
        #            kernel_size=[5,5],
        #            strides=[1,1],
        #            padding='same',
        #            activation=tf.nn.leaky_relu,
        #            use_bias=False,
        #            name="convt_1")
        #    print(convt_1)

        #    bnorm1=tf.layers.batch_normalization(convt_1)

        #    convt_2=tf.layers.conv2d_transpose(inputs=bnorm1,
        #            filters=24,
        #            kernel_size=[5,5],
        #            strides=[2,2],
        #            padding='same',
        #            activation=tf.nn.leaky_relu,
        #            use_bias=False,
        #            name="convt_2")
        #    print(convt_2)

        #    bnorm2=tf.layers.batch_normalization(convt_2)

        #    convt_3=tf.layers.conv2d_transpose(inputs=bnorm2,
        #            filters=3,
        #            kernel_size=[5,5],
        #            strides=[2,2],
        #            padding='same',
        #            use_bias=False,
        #            activation=tf.tanh,
        #            name="convt_3")
        #    print(convt_3)

        #with tf.variable_scope("Output"):
        #    g=convt_3
        #return g

def discriminator(X,std,reuse=False):
    with tf.variable_scope("Discriminator",reuse=reuse):
        with tf.variable_scope("Input"):
            X_noise=gauss_noise(X,std=std,shape=(-1,48,48,3),name="Noise")

        with tf.variable_scope("Convolution"):
            conv_1=tf.layers.conv2d(inputs=X_noise,
                    filters=64,
                    kernel_size=5,
                    strides=[2,2],
                    padding='same',
                    activation=tf.nn.leaky_relu,
                    bias_initializer=tf.truncated_normal_initializer(stddev=1e-2,dtype=tf.float32),
                    name="conv_1")
            print(conv_1)

            conv_2=tf.layers.conv2d(inputs=conv_1,
                    filters=128,
                    kernel_size=5,
                    strides=[2,2],
                    padding='same',
                    activation=tf.nn.leaky_relu,
                    bias_initializer=tf.truncated_normal_initializer(stddev=1e-2,dtype=tf.float32),
                    name="conv_2")
            print(conv_2)

        with tf.variable_scope("Output"):
            t=tf.layers.dropout(conv_2,rate=0.2)
            f=tf.layers.flatten(t)
            d=tf.layers.dense(f,units=1)

        return X_noise,d

def Zbatch(n,m):
    from numpy import random,zeros,ones
    #return random.uniform(0,1,size=[n,m])
    x=ones((n,m))
    print(x)
    x[0][130]=0.0
    x[0][490]=0.0
    x[0][642]=0.0
    x[0][998]=0.0
    x[0][1642]=0.0
    print(x)
    return x

def main(argv):
    print("Generative Adversarial Network")
    from numpy import random

    """Batch"""
    batch_size=1
    zbatch=16*16*12

    Z=tf.placeholder(tf.float32,[None,zbatch])
    std=tf.placeholder(tf.float32)
    img_batch,init_dataset=dataset(batch_size)

    """Generator"""
    g=generator(Z,std)

    """Discriminator"""
    #d_noise,d_logits=discriminator(img_batch,std)
    #d_g_noise,g_logits=discriminator(g,std,reuse=True)

    """logits"""
    #g_loss=tf.reduce_mean(
    #        tf.nn.sigmoid_cross_entropy_with_logits(logits=g_logits,
    #            labels=tf.ones_like(g_logits))
    #        )

    #real_loss=tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(d_logits),logits=d_logits)
    #fake_loss=tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(g_logits),logits=g_logits)

    #d_loss=real_loss+fake_loss
    #g_loss=tf.losses.sigmoid_cross_entropy(tf.ones_like(g_logits),g_logits)

    #d_loss=tf.reduce_mean(
    #        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits,
    #            labels=tf.ones_like(d_logits))+
    #        tf.nn.sigmoid_cross_entropy_with_logits(logits=g_logits,
    #            labels=tf.zeros_like(g_logits))
    #        )

    """Variables"""
    #g_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator")
    #d_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Discriminator")

    """Optimizer"""
    #g_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4,epsilon=1e-2)
    #d_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4,epsilon=1e-2)

    """Train"""
    #g_train=g_optimizer.minimize(g_loss,var_list=g_vars)
    #d_train=g_optimizer.minimize(d_loss,var_list=d_vars)

    """Summaries"""
    #all_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #for v in all_vars:
    #    tf.summary.histogram(v.name,v)
    #    print(v)

    #tf.summary.scalar("Generator_loss",g_loss)
    #tf.summary.scalar("Discriminator_loss",d_loss)

    tf.summary.image("Generator",g,max_outputs=24)
    #tf.summary.image("Discrminator G",d_g_noise,max_outputs=24)
    #tf.summary.image("Discrminator with noise",d_noise,max_outputs=24)
    #tf.summary.image("Discrminator image",img_batch,max_outputs=24)

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
        for step in range(1,2):

            #for d_step in range(6):
            #    _,dd=session.run([d_train,d_loss],feed_dict={Z:Zbatch(batch_size,512),std:5e-1/step})

            #for g_step in range(6):
            #    _,gg=session.run([g_train,g_loss],feed_dict={Z:Zbatch(batch_size,512),std:5e-1/step})

            #print("[%d] d:%lf g:%lf"%(step,dd,gg))

            gg,log=session.run([g,summaries],feed_dict={Z:Zbatch(batch_size,zbatch),std:0})
            writer.add_summary(log,global_step=step)

            print(gg)
            print(gg.shape)

        saver.save(session,'log/last.ckpt')


if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
