class generator:
    def __init__(self,z):
        self.z=z
        self.build_graph()

    def build_graph(self):
        import tensorflow as tf
        with tf.variable_scope("Generator"):

            self.dense_1=tf.layers.dense(inputs=self.z,
                    units=300,
                    activation=tf.nn.leaky_relu,
                    #kernel_initializer=tf.orthogonal_initializer,
                    use_bias=False,
                    name='dense_1')
            
            self.dense_2=tf.layers.dense(inputs=self.dense_1,
                    units=600,
                    activation=tf.nn.leaky_relu,
                    #kernel_initializer=tf.orthogonal_initializer,
                    use_bias=False,
                    name='dense_2')

            self.dense_3=tf.layers.dense(inputs=self.dense_2,
                    units=500,
                    activation=tf.tanh,
                    #kernel_initializer=tf.orthogonal_initializer,
                    use_bias=False,
                    name='dense_3')

            self.dense_4=tf.layers.dense(inputs=self.dense_3,
                    units=128,
                    activation=tf.tanh,
                    #kernel_initializer=tf.orthogonal_initializer,
                    use_bias=False,
                    name='dense_4')

            self.output=self.dense_4

class discriminator:
    def __init__(self,x,reuse=False,name='Discriminator'):
        self.x=x
        self.name=name
        self.build_graph(reuse=reuse)

    def build_graph(self,reuse):
        import tensorflow as tf
        with tf.variable_scope(self.name,reuse=reuse):

            self.dense_1=tf.layers.dense(inputs=self.x,
                    units=256,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_1')
            
            self.dense_2=tf.layers.dense(inputs=self.dense_1,
                    units=300,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_2')

            self.dense_3=tf.layers.dense(inputs=self.dense_2,
                    units=300,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_3')

            self.dense_4=tf.layers.dense(inputs=self.dense_3,
                    units=300,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_4')

            self.dense_5=tf.layers.dense(inputs=self.dense_4,
                    units=300,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_5')

            self.dense_out=tf.layers.dense(inputs=self.dense_5,
                    units=1,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_out')

            self.logits=tf.sigmoid(self.dense_out)

class GAN:
    def __init__(self,x,z,gan_dropout,learning_rate=1e-2):
        import tensorflow as tf
        nd=4

        self.x=x
        self.z=z

        self.gan_dropout=gan_dropout

        self.g=generator(self.z)
        self.discriminator_ensemble(nd=nd)

        self.define_real_loss(nd=nd)
        self.define_fake_loss(nd=nd)
        self.define_discriminator_loss(nd=nd)
        self.define_generator_loss(nd=nd)

        self.get_collections()

        """Optimizer"""
        with tf.variable_scope('generator_optimizer'):
            self.g_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        with tf.variable_scope('discriminator_optimizer'):
            self.d_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        """Train"""
        with tf.variable_scope('generator_training'):
            self.g_train=self.g_optimizer.minimize(self.g_loss,var_list=self.g_vars)

        with tf.variable_scope('discriminator_training'):
            self.d_train=self.g_optimizer.minimize(self.d_loss,var_list=self.d_vars)

        self.build_summary()

    def discriminator_ensemble(self,nd=8):
        import tensorflow as tf
        self.real=[]
        self.fake=[]
        for n in range(nd):
            self.real+=[discriminator(self.x,name="D%d"%n)]
            self.fake+=[discriminator(self.g.output,reuse=True,name="D%d"%n)]

    def define_real_loss(self,nd=8):
        import tensorflow as tf
        """logits"""
        with tf.name_scope('real_loss'):
            self.real_loss=[]
            for n in range(nd):
                self.real_loss+=[tf.losses.sigmoid_cross_entropy(
                        multi_class_labels=tf.ones_like(self.real[n].logits),
                        logits=self.real[n].logits)]

    def define_fake_loss(self,nd=8):
        import tensorflow as tf
        with tf.name_scope('fake_loss'):
            self.fake_loss=[]
            for n in range(nd):
                self.fake_loss+=[tf.losses.sigmoid_cross_entropy(
                        multi_class_labels=tf.zeros_like(self.fake[n].logits),
                        logits=self.fake[n].logits)]

    def define_discriminator_loss(self,nd=8):
        import tensorflow as tf
        with tf.name_scope('discriminator_loss'):
            self.d_loss=0
            count=1.0
            for n in range(nd):
                self.d_loss+=self.real_loss[n]+self.fake_loss[n]
                count+=1.0
            self.d_loss/=count

    def f1(self):
        import tensorflow as tf
        return tf.constant(0.0)

    def f2(self):
        import tensorflow as tf
        return tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.fake[self.n].logits),logits=self.fake[self.n].logits)

    def define_generator_loss(self,nd=8):
        import tensorflow as tf
        with tf.name_scope('generator_loss'):
            self.g_loss=0

            s=tf.reduce_sum(self.gan_dropout)
            s=tf.to_float(s)

            for n in range(nd):
                x=self.gan_dropout[n]
                self.n=n
                self.g_loss+=tf.cond(x>0,
                        self.f2,
                        self.f1)
            self.g_loss/=s

    def get_collections(self):
        import tensorflow as tf
        """Variables"""
        self.g_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator")
        self.d_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="D")

    def build_summary(self):
        import tensorflow as tf
        tf.summary.scalar("Generator_loss",self.g_loss)
        tf.summary.scalar("Discriminator_loss",self.d_loss)

        self.summaries=tf.summary.merge_all()
