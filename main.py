#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

batch_size=8
nsteps=5
zbatch_size=99

def _parse_fce(file,s=0.0):
    from numpy.random import uniform
    img_str=tf.read_file(file)
    img_decoded=tf.image.decode_png(img_str)
    img_crop=tf.image.central_crop(img_decoded,0.5)
    img_resized=tf.image.resize_images(img_crop,[48,48])
    img_gray=tf.image.rgb_to_grayscale(img_resized)
    noise=tf.random_normal(shape=tf.shape(img_gray),mean=0.0,stddev=s)
    return img_gray/128.0-1+noise

def dataset(s=0.0):
    from glob import glob

    files=glob('images2/*.png')
    files_dataset=tf.data.Dataset.from_tensor_slices((files,len(files)*[s]))
    files_dataset=files_dataset.map(_parse_fce)

    dataset=files_dataset.repeat().batch(batch_size).repeat(nsteps)
    iterator=tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)

    img_batch=iterator.get_next()
    init_dataset=iterator.make_initializer(dataset)

    return img_batch,init_dataset

def generator(Z):
    with tf.variable_scope("Generator"):
    
        dense1=tf.layers.dense(inputs=Z,units=3*3*96*2)
        c=tf.reshape(dense1,(-1,3,3,96*2))

        #nc=tf.layers.batch_normalization(c)

        convt0=tf.layers.conv2d_transpose(inputs=c,
                filters=96,
                kernel_size=2,
                strides=2,
                padding='same')
                #activation=tf.nn.relu)

        #nconvt0=tf.layers.batch_normalization(convt0)

        #dropout0=tf.layers.dropout(nconvt0,rate=0.0)

        convt1=tf.layers.conv2d_transpose(inputs=convt0,
                filters=48,
                kernel_size=2,
                strides=2,
                padding='same')
                #activation=tf.nn.relu)

        #nconvt1=tf.layers.batch_normalization(convt1)

        #dropout1=tf.layers.dropout(nconvt1,rate=0.0)

        convt2=tf.layers.conv2d_transpose(inputs=convt1,
                filters=24,
                kernel_size=2,
                strides=2,
                padding='same')
                #activation=tf.nn.relu)

        #nconvt2=tf.layers.batch_normalization(convt2)

        convt3=tf.layers.conv2d_transpose(inputs=convt2,
                filters=1,
                kernel_size=2,
                strides=2,
                padding='same')
                #activation=tf.nn.relu)

        #nconvt3=tf.layers.batch_normalization(convt3)
        t=tf.nn.tanh(convt3)

        print(c)
        print(convt0)
        print(convt1)
        print(convt2)
        print(convt3)
        print(t)

        return t

def discriminator(X,reuse=False):
    with tf.variable_scope("Discriminator",reuse=reuse):

        conv0=tf.layers.conv2d(inputs=X,
                filters=12,
                strides=2,
                kernel_size=2,
                padding='same')
                #activation=tf.nn.leaky_relu)

        nconv0=tf.layers.batch_normalization(conv0)
        dropout0=tf.layers.dropout(nconv0,rate=0.0)

        conv1=tf.layers.conv2d(inputs=dropout0,
                filters=24,
                strides=2,
                kernel_size=2,
                padding='same')
                #activation=tf.nn.leaky_relu)

        nconv1=tf.layers.batch_normalization(conv1)
        dropout1=tf.layers.dropout(nconv1,rate=0.0)

        conv2=tf.layers.conv2d(inputs=dropout1,
                filters=48,
                strides=2,
                kernel_size=2,
                padding='same')
                #activation=tf.nn.leaky_relu)

        nconv2=tf.layers.batch_normalization(conv2)
        dropout2=tf.layers.dropout(nconv2,rate=0.0)

        conv3=tf.layers.conv2d(inputs=dropout2,
                filters=96,
                strides=2,
                kernel_size=2,
                padding='same')
                #activation=tf.nn.leaky_relu)

        nconv3=tf.layers.batch_normalization(conv3)

        dropout3=tf.layers.dropout(nconv3,rate=0.0)

        flat=tf.layers.flatten(dropout3)
        d_prob=tf.layers.dense(inputs=flat,units=1)

        print(conv0)
        print(conv1)
        print(conv2)
        print(conv3)
        print(flat)
        print(d_prob)

        #return tf.sigmoid(d_prob)
        return d_prob

def main(argv):
    print("Generative Adversarial Network")
    from numpy import random

    img_batch,init_dataset=dataset()
    img_noise_batch,init_noise_dataset=dataset(s=0.33)

    image_x=tf.placeholder(tf.float32,[None,48,48,1])
    Z=tf.placeholder(tf.float32,[None,zbatch_size])

    generator_sample=generator(Z)
    real_logits=discriminator(img_batch)
    fake_logits=discriminator(generator_sample,reuse=True)
    attack_logits=discriminator(img_noise_batch,reuse=True)

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
    attack_loss=tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                labels=tf.ones_like(real_logits))+
            tf.nn.sigmoid_cross_entropy_with_logits(logits=attack_logits,
                labels=tf.zeros_like(attack_logits))
            )



    generator_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator")
    discriminator_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Discriminator")

    generator_optimizer=tf.train.AdamOptimizer(learning_rate=5e-6)
    discriminator_optimizer=tf.train.AdamOptimizer(learning_rate=5e-6)
    attack_optimizer=tf.train.AdamOptimizer(learning_rate=5e-6)

    train_generator=generator_optimizer.minimize(generator_loss,var_list=generator_vars)
    train_discriminator=discriminator_optimizer.minimize(discriminator_loss,var_list=discriminator_vars)
    train_attack=attack_optimizer.minimize(attack_loss,var_list=discriminator_vars)

    img_summary=tf.summary.image('image',img_batch,max_outputs=10)
    attack_img_summary=tf.summary.image('attack_image',img_noise_batch,max_outputs=10)

    gen_img_summary=tf.summary.image('generated_image',image_x,max_outputs=20)


    with tf.Session() as session:


        session.run(init_dataset)
        session.run(init_noise_dataset)
        tf.global_variables_initializer().run(session=session)

        writer=tf.summary.FileWriter("log0",session.graph)

        Z_batch=random.uniform(-1,1,size=[batch_size,zbatch_size])

        d_loss=0.0

        for i in range(20000):

            Z_batch=random.uniform(-1,1,size=[batch_size,zbatch_size])

            #for _ in range(nsteps):
            #    _,d_loss=session.run([train_discriminator,discriminator_loss],feed_dict={Z:Z_batch})

            for _ in range(nsteps):
                _,a_loss=session.run([train_attack,attack_loss],feed_dict={Z:Z_batch})

            for _ in range(nsteps):
                _,g_loss=session.run([train_generator,generator_loss],feed_dict={Z:Z_batch})


            #for i,v in enumerate(discriminator_vars):
            #    print(i,v)
            #print(session.run(discriminator_vars[0]))

            if i%1 == 0:

                summary=session.run(img_summary)
                writer.add_summary(summary,i)

                summary=session.run(attack_img_summary)
                writer.add_summary(summary,i)

                gen_img=session.run(generator_sample,feed_dict={Z:Z_batch})

                summary=session.run(gen_img_summary,feed_dict={image_x:gen_img})
                writer.add_summary(summary,i)

                print("[%d]"%i,"[d_loss]",d_loss,"[g_loss]",g_loss,"[a_loss]",a_loss)

                #img=session.run(img_batch)
                #print(gen_img[0])
                #print(img[0])

            #if i%1 == 0:
            #    saver=save(session,'check')


if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
