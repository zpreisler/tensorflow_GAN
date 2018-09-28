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

            self.flatten=tf.layers.flatten(conv2d)
            p=tf.layers.dense(f,units=1)
            self.logits=tf.sigmoid(p)
