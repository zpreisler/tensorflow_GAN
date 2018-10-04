class generator:
    def __init__(self,z):
        self.z=z
        self.build_graph()

    def build_graph(self):
        import tensorflow as tf
        with tf.variable_scope("Generator"):

            self.dense_1=tf.layers.dense(inputs=self.z,
                    units=128,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_1')
            
            self.dense_2=tf.layers.dense(inputs=self.dense_1,
                    units=128,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_2')

            self.output_image=self.dense_2

class discriminator:
    def __init__(self,x,reuse=False):
        self.x=x
        self.build_graph(reuse=reuse)

    def build_graph(self,reuse):
        import tensorflow as tf
        with tf.variable_scope("Discriminator",reuse=reuse):

            self.dense_1=tf.layers.dense(inputs=self.x,
                    units=128,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_1')
            
            self.dense_2=tf.layers.dense(inputs=self.dense_1,
                    units=128,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_2')

            self.dense_3=tf.layers.dense(inputs=self.dense_2,
                    units=2,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_3')

            self.dense_out=tf.layers.dense(inputs=self.dense_3,
                    units=1,
                    activation=tf.nn.leaky_relu,
                    kernel_initializer=tf.orthogonal_initializer,
                    name='dense_out')

            self.logits=tf.sigmoid(self.dense_out)

class GAN:
    def __init__(self,x,z,learning_rate=1e-2):
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

        #tf.summary.image("Generator_image",self.g.output_image,max_outputs=24)
        #tf.summary.image("Discrminator_image",self.x,max_outputs=24)

        #tf.summary.histogram("Generator_image",self.g.output_image)
        #tf.summary.histogram("Discrminator_image",self.x)

        self.summaries=tf.summary.merge_all()
