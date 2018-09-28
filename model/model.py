class generator:
    def __init__(self,z):
        self.z=z
        self.build_graph()

    def build_graph(self):
        import tensorflow as tf
        with tf.variable_scope("Generator"):

            self.dense=tf.layers.dense(inputs=self.z,
                    units=49*49*3,
                    use_bias=False,
                    activation=tf.nn.relu,
                    name='dense')

            self.dense_reshape=tf.reshape(self.dense,(-1,49,49,3),name='reshape')

            self.conv2d=tf.layers.conv2d(inputs=self.dense_reshape,
                    filters=3,
                    kernel_size=[2,2],
                    strides=[1,1],
                    padding='valid',
                    use_bias=False,
                    kernel_initializer=tf.ones_initializer(),
                    activation=tf.nn.relu,
                    name='conv2d')
            
            self.output_image=tf.tanh(self.conv2d)

class discriminator:
    def __init__(self,x,reuse=False):
        self.x=x
        self.build_graph(reuse=reuse)

    def build_graph(self,reuse):
        import tensorflow as tf
        with tf.variable_scope("Discriminator",reuse=reuse):

            self.conv2d=tf.layers.conv2d(inputs=self.x,
                    filters=32,
                    kernel_size=[2,2],
                    strides=[1,1],
                    padding='valid',
                    activation=tf.nn.leaky_relu,
                    name="conv2d")

            self.flatten=tf.layers.flatten(self.conv2d)
            p=tf.layers.dense(self.flatten,units=1)
            self.logits=tf.sigmoid(p)

    def gauss_noise(x,std,shape,name='Noise'):
        with tf.name_scope(name):
            n=tf.truncated_normal(shape=tf.shape(x),mean=0.0,stddev=std,dtype=tf.float32)
            a=tf.add(x,n)
        return tf.reshape(a,shape)

class GAN:
    def __init__(self,x,z):
        import tensorflow as tf
        self.x=x
        self.z=z
        self.g=generator(self.z)

        self.real_d=discriminator(self.x)
        self.fake_d=discriminator(self.g.output_image,reuse=True)

        """logits"""
        with tf.name_scope('real_loss'):
            self.real_loss=tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.real_d.logits),logits=self.real_d.logits)

        with tf.name_scope('fake_loss'):
            self.fake_loss=tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(self.fake_d.logits),logits=self.fake_d.logits)

        with tf.name_scope('discriminator_loss'):
            self.d_loss=self.real_loss+self.fake_loss

        with tf.name_scope('generator_loss'):
            self.g_loss=tf.losses.sigmoid_cross_entropy(tf.ones_like(self.fake_d.logits),self.fake_d.logits)

        """Variables"""
        self.g_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Generator")
        self.d_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="Discriminator")

        """Optimizer"""
        with tf.variable_scope('generator_optimizer'):
            self.g_optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)

        with tf.variable_scope('discriminator_optimizer'):
            self.d_optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)

        """Train"""
        with tf.variable_scope('generator_training'):
            self.g_train=self.g_optimizer.minimize(self.g_loss,var_list=self.g_vars)

        with tf.variable_scope('discriminator_training'):
            self.d_train=self.g_optimizer.minimize(self.d_loss,var_list=self.d_vars)

        tf.summary.scalar("Generator_loss",self.g_loss)
        tf.summary.scalar("Discriminator_loss",self.d_loss)

        tf.summary.image("Generator_image",self.g.output_image,max_outputs=24)
        tf.summary.image("Discrminator_image",self.x,max_outputs=24)

        self.summaries=tf.summary.merge_all()