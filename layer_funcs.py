# Basic layer functions for our model
import tensorflow as tf

# weight
def weight(name,shape,stddev=0.02,trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name,shape,tf.float32,trainable=trainable,initializer=tf.random_normal_initializer(stddev=stddev,dtype=dtype))

    return var

# bias
def bias(name,shape,bias_start=0.0,trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name,shape,tf.float32,trainable=trainable,initializer=tf.constant_initializer(bias_start,dtype=dtype))

    return var

# leaky-relu:
def lrelu(x,leak=0.2,name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x,leak*x,name=name)

# relu
def relu(value,name='relu'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)

# conv2d
def conv2d(value,output_dim,k_h=5,k_w=5,strides=[1,2,2,1],name='conv2d'):
    with tf.variable_scope(name):
        weights = weight('weights',[k_h,k_w,value.get_shape()[-1],output_dim])
        conv = tf.nn.conv2d(value,weights,strides=strides,padding='SAME')
        biases = bias('biases',[output_dim])
        conv = tf.nn.bias_add(conv,biases)

        return conv

# deconv2d
def deconv2d(value,shape,d_shape,k_h=5,k_w=5,strides=[1,2,2,1],name='deconv2d',with_w=False):
    with tf.variable_scope(name):        
        weights = weight('weights',[k_h,k_w,shape[-1],value.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(value,weights,d_shape,strides=strides)
        biases = bias('biases',[shape[-1]])
        deconv = tf.nn.bias_add(deconv,biases)
        if with_w:
            return deconv,weights,biases
        else:
            return deconv

# batch normalize
def batch_norm(value,mode='train',name='batch_norm',epsilon=1e-5,momentum=0.9):
    with tf.variable_scope(name):
        if mode=='train':
            return tf.layers.batch_normalization(value,momentum=momentum,epsilon=epsilon,training=True)
        else:
            return tf.layers.batch_normalization(value,momentum=momentum,epsilon=epsilon,training=False)
        
#        
