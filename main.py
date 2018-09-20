#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

batch_size=64
nsteps=24

def _parse_fce(file):
    img_str=tf.read_file(file)
    img_decoded=tf.image.decode_png(img_str)
    img_crop=tf.image.central_crop(img_decoded,0.5)
    img_resized=tf.image.resize_images(img_crop,[48,48])
    img_gray=tf.image.rgb_to_grayscale(img_resized)
    return img_gray

def dataset():
    from glob import glob

    files=glob('images/*.png')
    files_dataset=tf.data.Dataset.from_tensor_slices(files)
    files_dataset=files_dataset.map(_parse_fce)

    dataset=files_dataset.repeat().batch(batch_size).repeat(nsteps)
    iterator=tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)

    img_batch=iterator.get_next()
    init_dataset=iterator.make_initializer(dataset)

    return img_batch,init_dataset

def generator(Z):
    with tf.variable_scope("Generator"):
    
        dense1=tf.layers.dense(inputs=Z,units=6*6*48)
        c=tf.reshape(dense1,(-1,6,6,48))

        convt1=tf.layers.conv2d_transpose(inputs=c,
                filters=48,
                kernel_size=5,
                strides=2,
                padding='same')
        convt2=tf.layers.conv2d_transpose(inputs=convt1,
                filters=24,
                kernel_size=5,
                strides=2,
                padding='same')
        conv3=tf.layers.conv2d_transpose(convt2,
                filters=1,
                kernel_size=5,
                strides=2,
                padding='same')
        return conv3

def discriminator(X,reuse=False):
    with tf.variable_scope("Discriminator",reuse=reuse):
        conv1=tf.layers.conv2d(inputs=X,
                filters=6,
                kernel_size=5,
                padding='same',
                activation=tf.nn.leaky_relu)

        pool1=tf.layers.average_pooling2d(inputs=conv1,pool_size=2,strides=2)

        conv2=tf.layers.conv2d(inputs=pool1,
                filters=12,
                kernel_size=5,
                padding='same',
                activation=tf.nn.leaky_relu)
        pool2=tf.layers.average_pooling2d(inputs=conv2,pool_size=2,strides=2)

        conv3=tf.layers.conv2d(inputs=pool2,
                filters=24,
                kernel_size=5,
                padding='same',
                activation=tf.nn.leaky_relu)
        pool3=tf.layers.average_pooling2d(inputs=conv3,pool_size=2,strides=2)

        flat=tf.reshape(pool3,(-1,6*6*24))
        d_prob=tf.layers.dense(inputs=flat,units=1)

        return tf.sigmoid(d_prob)

def main(argv):
    print("Generative Adversarial Network")
    from numpy import random

    img_batch,init_dataset=dataset()

    image_x=tf.placeholder(tf.float32,[None,48,48,1])
    Z=tf.placeholder(tf.float32,[None,64])

    #Z_batch=random.uniform(-1,1,size=[batch_size,64])

    generator_sample=generator(Z)
    real_logits=discriminator(img_batch)
    fake_logits=discriminator(generator_sample,reuse=True)

    generator_loss=tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                labels=tf.ones_like(fake_logits))
            )

    discriminator_loss=tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                labels=tf.ones_like(real_logits))+
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                labels=tf.zeros_like(fake_logits))
            )

    generator_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator")
    discriminator_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Discriminator")

    generator_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)
    discriminator_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)

    train_generator=generator_optimizer.minimize(generator_loss,var_list=generator_vars)
    train_discriminator=discriminator_optimizer.minimize(discriminator_loss,var_list=discriminator_vars)

    img_summary=tf.summary.image('image',img_batch,max_outputs=4)
    gen_img_summary=tf.summary.image('generated_image',image_x,max_outputs=4)

    with tf.Session() as session:

        session.run(init_dataset)
        tf.global_variables_initializer().run(session=session)

        writer=tf.summary.FileWriter("log",session.graph)

        for i in range(10000):

            Z_batch=random.uniform(-1,1,size=[batch_size,64])

            for _ in range(nsteps):
                _,d_loss=session.run([train_discriminator,discriminator_loss],feed_dict={Z:Z_batch})

            for _ in range(nsteps):
                _,g_loss=session.run([train_generator,generator_loss],feed_dict={Z:Z_batch})

            if i%1 == 0:
                summary=session.run(img_summary)
                writer.add_summary(summary,i)

                gen_img=session.run(generator_sample,feed_dict={Z:Z_batch})
                summary=session.run(gen_img_summary,feed_dict={image_x:gen_img})
                writer.add_summary(summary,i)

                print("[%d]"%i,"[d_loss]",d_loss,"[g_loss]",g_loss)


if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
