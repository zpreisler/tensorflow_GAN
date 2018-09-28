#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def zbatch(n,m):
    from numpy import random,zeros,ones
    return random.uniform(0,1,size=[n,m])

def main(argv):
    print("Generative Adversarial Network")
    from numpy import random
    from model.data_pipeline import dataset
    from model.model import generator,discriminator,GAN

    """Batch"""
    batch_size=32
    zbatch_size=1024

    img_batch,init_dataset=dataset(batch_size=batch_size)

    """GAN"""

    z=tf.placeholder(tf.float32,[None,zbatch_size])
    gan=GAN(img_batch,z=z)

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
        for step in range(1,200):

            img=session.run(gan.g.output_image,feed_dict={z:zbatch(batch_size,zbatch_size)})

            for d_step in range(6):
                _,dd=session.run([gan.d_train,gan.d_loss],feed_dict={z:zbatch(batch_size,zbatch_size)})

            for g_step in range(6):
                _,gg=session.run([gan.g_train,gan.g_loss],feed_dict={z:zbatch(batch_size,zbatch_size)})

            if step%20 is 0:
                print("[%d] d:%lf g:%lf"%(step,dd,gg))

                log=session.run(gan.summaries,feed_dict={z:zbatch(batch_size,zbatch_size)})
                writer.add_summary(log,global_step=step)

        saver.save(session,'log/run/last.ckpt')


if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
