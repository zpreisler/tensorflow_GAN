#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def zbatch(n,m):
    from numpy import random,zeros,ones
    return random.uniform(0,1,size=[n,m])

def one_peak(n,m,t=4096):
    from numpy import random,zeros,ones,histogram,array,add,concatenate
    rnd=random.normal(0,1,(n,t))
    p=[]
    for r in rnd:
        h,x=histogram(r,bins=m)
        p+=[[*h]]
    p=array(p)
    return p/p.max()

def two_peak(n,m,t=2048):
    from numpy import random,zeros,ones,histogram,array,add,concatenate
    rnd1=random.normal(0,1,(n,t))
    rnd2=random.normal(5,1,(n,t))
    rnd=concatenate((rnd1,rnd2),1)
    p=[]
    for r in rnd:
        h,x=histogram(r,bins=m)
        p+=[[*h]]
    p=array(p)
    return p/p.max()

def mix_peak(n,m,t=2048):
    from numpy import random,zeros,ones,histogram,array,add,concatenate
    if random.randint(2) is 0:
        rnd1=random.normal(0,1,(n,t))
        rnd2=random.normal(5,1,(n,t))
        rnd=concatenate((rnd1,rnd2),1)
    else:
        rnd=random.normal(0,1,(n,t*2))
    p=[]
    for r in rnd:
        h,x=histogram(r,bins=m)
        p+=[[*h]]
    p=array(p)
    return p/p.max()

def xbatch(n,m,mode=0):
    from numpy import random,zeros,ones,histogram,array,add,concatenate
    fce=[one_peak(n,m),two_peak(n,m),mix_peak(n,m)]
    return fce[2]
    #return fce[mode]

def main(argv):
    print("Generative Adversarial Network")
    from numpy import random,arange
    from model.data_pipeline import dataset
    from model.model import generator,discriminator,GAN

    """Batch"""
    batch_size=128
    xbatch_size=128
    zbatch_size=256

    steps=50000
    d_steps=8
    g_steps=8

    run="mix"

    """GAN"""

    x=tf.placeholder(tf.float32,[None,xbatch_size])
    z=tf.placeholder(tf.float32,[None,zbatch_size])
    gan=GAN(x=x,z=z,learning_rate=1e-2)

    """Checkpoints"""
    saver=tf.train.Saver()

    xx=xbatch(batch_size,xbatch_size)
    print(x)
    print(xx.shape)

    from matplotlib.pyplot import plot,show,figure,close,savefig,xlim,ylim,legend

    with tf.Session() as session:
        print("Session")

        """Init"""
        tf.global_variables_initializer().run(session=session)
        #saver.restore(session,"log/mix/last.ckpt")

        """Summaries"""
        writer=tf.summary.FileWriter("log/"+run,session.graph)

        count=0
        """Learning"""
        for step in range(1,steps):
            for d_step in range(d_steps):
                _,dd=session.run([gan.d_train,gan.d_loss],feed_dict={x: xbatch(batch_size,xbatch_size), z: zbatch(batch_size,zbatch_size)})
            for g_step in range(g_steps):
                _,gg=session.run([gan.g_train,gan.g_loss],feed_dict={x: xbatch(batch_size,xbatch_size), z: zbatch(batch_size,zbatch_size)})

            if step%5 is 0:
                print("[%d] d:%lf g:%lf"%(step,dd,gg))

                log=session.run(gan.summaries,feed_dict={x: xbatch(batch_size,xbatch_size), z:zbatch(batch_size,zbatch_size)})
                writer.add_summary(log,global_step=step)
                
                gg,dd=session.run([gan.g.output_image,gan.x],feed_dict={x: xbatch(batch_size,xbatch_size), z:zbatch(batch_size,zbatch_size)})

                figure()
                plot(gg[0],label="generated %d"%count)
                plot(gg[1],label="generated %d"%count)
                plot(dd[0],label="true")
                xlim(0,xbatch_size)
                ylim(-0.2,1.2)
                legend(frameon=False,loc=1)
                savefig("figures/"+run+"/f_%04d.png"%count)
                count+=1
                close()

            if step%100 is 0:
                saver.save(session,'log/'+run+'/last.ckpt')


if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
