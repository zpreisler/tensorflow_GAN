class generator:
    def __init__(self,z):
        self.z=z
        self.build_graph()

    def build_graph(self):
        import tensorflow as tf
        with tf.variable_scope("Generator"):

            self.dense_in=tf.layers.dense(inputs=self.z,
                    units=64*4*4,
                    #use_bias=False,
                    activation=tf.nn.relu,
                    name='dense_1')

            self.dense_reshape=tf.reshape(self.dense_in,(-1,4,4,64),name='reshape')

            print(self.dense_reshape)

            self.ct1=tf.layers.conv2d_transpose(inputs=self.dense_reshape,
                    filters=32,
                    kernel_size=[4,4],
                    strides=[2,2],
                    padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name="ct1")

            print(self.ct1)

            self.ct2=tf.layers.conv2d_transpose(inputs=self.ct1,
                    filters=16,
                    kernel_size=[4,4],
                    strides=[2,2],
                    padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name="ct2")

            print(self.ct2)

            self.ct3=tf.layers.conv2d_transpose(inputs=self.ct2,
                    filters=8,
                    kernel_size=[4,4],
                    strides=[2,2],
                    padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name="ct3")

            print(self.ct3)

            self.ct4=tf.layers.conv2d_transpose(inputs=self.ct3,
                    filters=1,
                    kernel_size=[4,4],
                    strides=[2,2],
                    padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name="ct4")

            print(self.ct4)

            self.output_image=self.ct4

            print(self.output_image)

class discriminator:
    def __init__(self,x,reuse=False):
        self.x=x
        self.build_graph(reuse=reuse)

    def build_graph(self,reuse):
        import tensorflow as tf
        with tf.variable_scope("Discriminator",reuse=reuse):

            self.conv2d=tf.layers.conv2d(inputs=self.x,
                    filters=8,
                    kernel_size=[2,2],
                    strides=[1,1],
                    padding='valid',
                    #use_bias=False,
                    #bias_initializer=tf.zeros_initializer(),
                    #kernel_initializer=tf.ones_initializer(),
                    #kernel_initializer=tf.orthogonal_initializer,
                    activation=tf.nn.leaky_relu,
                    name="conv2d")
            
            self.pool=tf.layers.average_pooling2d(inputs=self.conv2d,
                    pool_size=[2,2],
                    strides=[2,2],
                    padding='valid',
                    name="pool2d")

            self.conv2d_2=tf.layers.conv2d(inputs=self.pool,
                    filters=16,
                    kernel_size=[2,2],
                    strides=[1,1],
                    padding='valid',
                    #use_bias=False,
                    #bias_initializer=tf.zeros_initializer(),
                    #kernel_initializer=tf.ones_initializer(),
                    activation=tf.nn.leaky_relu,
                    name="conv2d_2")

            self.pool_2=tf.layers.average_pooling2d(inputs=self.conv2d_2,
                    pool_size=[2,2],
                    strides=[2,2],
                    padding='valid',
                    name="pool2d_2")

            self.conv2d_3=tf.layers.conv2d(inputs=self.pool_2,
                    filters=32,
                    kernel_size=[2,2],
                    strides=[1,1],
                    padding='valid',
                    #use_bias=False,
                    #bias_initializer=tf.zeros_initializer(),
                    #kernel_initializer=tf.ones_initializer(),
                    activation=tf.nn.leaky_relu,
                    name="conv2d_3")

            self.flatten=tf.layers.flatten(self.conv2d_3)
            p=tf.layers.dense(self.flatten,units=1)
            self.logits=tf.sigmoid(p)

    def gauss_noise(x,std,shape,name='Noise'):
        with tf.name_scope(name):
            n=tf.truncated_normal(shape=tf.shape(x),mean=0.0,stddev=std,dtype=tf.float32)
            a=tf.add(x,n)
        return tf.reshape(a,shape)

class GAN:
    def __init__(self,x,z,learning_rate=1e-3):
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
            self.g_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        with tf.variable_scope('discriminator_optimizer'):
            self.d_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

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
