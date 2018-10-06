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

def mixed_peak(n,m,t=2048):
    from numpy import random,zeros,ones,histogram,array,add,concatenate
    #if random.randint(2) is 0:
    #    rnd1=random.normal(0,1,(n,t))
    #    rnd2=random.normal(5,1,(n,t))
    #    rnd=concatenate((rnd1,rnd2),1)
    #else:
    #    rnd=random.normal(0,1,(n,t*2))

    r=[]
    for n in range(n):
        if random.randint(2) is 0:
            rnd=random.normal(0,1,(t*2))
        else:
            rnd1=random.normal(0,1,(t))
            rnd2=random.normal(5,1,(t))
            rnd=concatenate((rnd1,rnd2),0)

        r+=[rnd]
    rnd=r

    p=[]
    for r in rnd:
        h,x=histogram(r,bins=m)
        p+=[[*h]]
    p=array(p)
    return p/p.max()



def xbatch(n,m,mode=0):
    from numpy import random,zeros,ones,histogram,array,add,concatenate
    fce=[one_peak(n,m),two_peak(n,m),mix_peak(n,m),mixed_peak(n,m)]
    return fce[3]
    #return fce[mode]

def main(argv):
    print("Generative Adversarial Network")
    from numpy import random,arange,ones
    from model.data_pipeline import dataset
    from model.model import generator,discriminator,GAN

    """Batch"""
    batch_size=32
    xbatch_size=128
    zbatch_size=512

    steps=20000
    d_steps=8
    g_steps=8

    run="wan"

    """GAN"""

    x=tf.placeholder(tf.float32,[None,xbatch_size])
    z=tf.placeholder(tf.float32,[None,zbatch_size])

    gan_dropout=tf.placeholder(tf.int32)

    gan=GAN(x=x,z=z,gan_dropout=gan_dropout,learning_rate=1e-2)

    """Checkpoints"""
    saver=tf.train.Saver()

    #xx=xbatch(batch_size,xbatch_size)
    #print(x)
    #print(xx.shape)
    nx=4
    ny=4

    from matplotlib.pyplot import plot,show,figure,close,savefig,xlim,ylim,legend,subplots
    from numpy import array,arange

    with tf.Session() as session:
        print("Session")

        """Init"""
        tf.global_variables_initializer().run(session=session)
        #saver.restore(session,"log/mix/last.ckpt")

        """Summaries"""
        writer=tf.summary.FileWriter("log/"+run,session.graph)

        count=0
        """Learning"""

        d_dropout=random.randint(2,size=4)
        d_dropout=ones(4)

        for step in range(1,steps):
            for d_step in range(d_steps):
                _,dd=session.run([gan.d_train,gan.d_loss],
                        feed_dict={x: xbatch(batch_size,xbatch_size),
                            z: zbatch(batch_size,zbatch_size),
                            gan_dropout: d_dropout})

            for g_step in range(g_steps):
                _,gg=session.run([gan.g_train,gan.g_loss],
                        feed_dict={x: xbatch(batch_size,xbatch_size),
                            z: zbatch(batch_size,zbatch_size),
                            gan_dropout: d_dropout})

            if step%5 is 0:
                print("[%d] d:%lf g:%lf"%(step,dd,gg))

                log=session.run(gan.summaries,
                        feed_dict={x: xbatch(batch_size,xbatch_size),
                            z:zbatch(batch_size,zbatch_size),
                            gan_dropout: d_dropout})
                writer.add_summary(log,global_step=step)
                
            if step%50 is 0:
                gg,dd=session.run([gan.g.output,gan.x],
                        feed_dict={x: xbatch(batch_size,xbatch_size),
                            z:zbatch(batch_size,zbatch_size),
                            gan_dropout: d_dropout})

                print(count,gg.shape,dd.shape)

                #fig,axes=subplots(nx,ny,constrained_layout=True,figsize=(12,12))
                fig,axes=subplots(nx,ny,figsize=(12,12))

                for nn in arange(nx):
                    for mm in arange(ny):
                        print(nn,mm)
                        axes[nn,mm].plot(gg[nn*4+mm],label="generated %d"%count)
                        axes[nn,mm].plot(dd[nn*4+mm],label="true")
                        axes[nn,mm].set_xlim(0,xbatch_size)
                        axes[nn,mm].set_ylim(-0.2,1.2)
                        axes[nn,mm].legend(frameon=False,loc=1)

                savefig("figures/"+run+"/f_%04d.png"%count)
                close()

                count+=1

            if step%1 is 0:
                d_dropout=random.randint(2,size=4)
                s=sum(d_dropout)
                while (s==0):
                    d_dropout=random.randint(2,size=4)
                    s=sum(d_dropout)

                print(s)

            if step%1000 is 0:
                saver.save(session,'log/'+run+'/last.ckpt')


if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
