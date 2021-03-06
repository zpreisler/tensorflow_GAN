#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def zbatch(n,m):
    from numpy import random,zeros,ones
    #return random.uniform(0,1,size=[n,m])
    return random.normal(0,1,size=[n,m])

def img_save(img,prefix="",count=0):
    from matplotlib.pyplot import imshow,show,figure,subplots_adjust,savefig,imsave,close
    from numpy import stack,concatenate,swapaxes

    print("img.shape",img.shape)
    a=concatenate(img,0)
    print("a.shape",a.shape)
    b=swapaxes(a,0,2)
    print("b.shape",b.shape)
    c=concatenate(b,0)
    print("c.shape",c.shape)

    fig=figure(figsize=(16,16))
    subplots_adjust(left=0,bottom=0,right=1,top=1)

    name="conv/%s_%d.png"%(prefix,count)
    print(name)
    #imshow(c)
    imsave(name,c,dpi=1200)
    close(fig)

def img_plot(img):
    from matplotlib.pyplot import imshow,show,figure,subplots_adjust,savefig,imsave,close
    from numpy import stack,concatenate,swapaxes

    print("img.shape",img.shape)
    a=concatenate(img,0)
    print("a.shape",a.shape)
    b=swapaxes(a,0,2)
    print("b.shape",b.shape)
    c=concatenate(b,0)
    print("c.shape",c.shape)

    fig=figure(figsize=(16,16))
    subplots_adjust(left=0,bottom=0,right=1,top=1)

    imshow(c)

def main(argv):
    print("Generative Adversarial Network")
    from numpy import random
    from model.data_pipeline import dataset
    from model.model import generator,discriminator,GAN

    """Batch"""
    batch_size=16
    zbatch_size=4096

    img_batch,init_dataset=dataset("images_made/*.png",batch_size=batch_size)

    """GAN"""

    z=tf.placeholder(tf.float32,[None,zbatch_size])
    gan=GAN(img_batch,z=z,learning_rate=1e-3)

    """Checkpoints"""
    saver=tf.train.Saver()

    with tf.Session() as session:
        print("Session")

        """Init"""
        tf.global_variables_initializer().run(session=session)
        session.run(init_dataset)

        """Summaries"""
        writer=tf.summary.FileWriter("log/run",session.graph)

        """Learning"""


        from matplotlib.pyplot import imshow,show,figure

        for step in range(1,20000):

            img=session.run(gan.g.output_image,feed_dict={z:zbatch(batch_size,zbatch_size)})

            for d_step in range(6):
                _,dd=session.run([gan.d_train,gan.d_loss],feed_dict={z:zbatch(batch_size,zbatch_size)})

            for g_step in range(6):
                _,gg=session.run([gan.g_train,gan.g_loss],feed_dict={z:zbatch(batch_size,zbatch_size)})

            if step%5 is 0:
                print("[%d] d:%lf g:%lf"%(step,dd,gg))

                log=session.run(gan.summaries,feed_dict={z:zbatch(batch_size,zbatch_size)})
                writer.add_summary(log,global_step=step)

            if step%1000 is 0:
                saver.save(session,'log/run/last.ckpt')


if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
