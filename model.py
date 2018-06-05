import tensorflow as tf
from layer_funcs import *
from utils import *
import os
import time

# basic initialization
HEIGHT = 175
WIDTH = 250
BATCH_SIZE = 25
LR = 0.0002

# sampler: for sampling result generated during training
def sampler(image,mode='validation',name='sampler'):
    with tf.name_scope(name):
        tf.get_variable_scope().reuse_variables()
        return generator(image,mode=mode)

# generator: it has 12 layers, each layer includes convolution, batch-normalization, relu activation functions 
def generator(image,mode='train',name='generator'):
    shape0 = image.get_shape().as_list()
    shape0[-1] = 3
    useless_node = conv2d(image,output_dim=3,strides=[1,1,1,1],name='useless_node')
    d_shape0 = tf.shape(useless_node)
    
    with tf.name_scope(name):
        g_conv1 = conv2d(image,output_dim=64,name='g_conv1')
        g_bn1 = batch_norm(g_conv1,mode=mode,name='g_bn1')
        g_lrelu1 = lrelu(g_bn1,name='g_lrelu1')
        shape1 = g_lrelu1.get_shape().as_list()
        d_shape1 = tf.shape(g_lrelu1)
        
        g_conv2 = conv2d(g_lrelu1,output_dim=128,name='g_conv2')
        g_bn2 = batch_norm(g_conv2,mode=mode,name='g_bn2')
        g_lrelu2 = lrelu(g_bn2,name='g_lrelu2')
        shape2 = g_lrelu2.get_shape().as_list()
        d_shape2 = tf.shape(g_lrelu2)
        
        g_conv3 = conv2d(g_lrelu2,output_dim=256,name='g_conv3')
        g_bn3 = batch_norm(g_conv3,mode=mode,name='g_bn3')
        g_lrelu3 = lrelu(g_bn3,name='g_lrelu3')
        shape3 = g_lrelu3.get_shape().as_list()
        d_shape3 = tf.shape(g_lrelu3)
        
        g_conv4 = conv2d(g_lrelu3,output_dim=512,name='g_conv4')
        g_bn4 = batch_norm(g_conv4,mode=mode,name='g_bn4')
        g_lrelu4 = lrelu(g_bn4,name='g_lrelu4')
        shape4 = g_lrelu4.get_shape().as_list()
        d_shape4 = tf.shape(g_lrelu4)
        
        g_conv5 = conv2d(g_lrelu4,output_dim=512,name='g_conv5')
        g_bn5 = batch_norm(g_conv5,mode=mode,name='g_bn5')
        g_lrelu5 = lrelu(g_bn5,name='g_lrelu5')
        shape5 = g_lrelu5.get_shape().as_list()
        d_shape5 = tf.shape(g_lrelu5)
        
        g_encode = conv2d(g_lrelu5,output_dim=512,name='g_encode')
        g_encode_bn = batch_norm(g_encode,mode=mode,name='g_encode_bn')
        g_encode_lrelu = lrelu(g_encode_bn,name='g_encode_lrelu')

        # use U-Net decoder
        g_de_conv5 = deconv2d(g_encode_lrelu,shape=shape5,d_shape=d_shape5,name='g_de_conv5')
        g_de_bn5 = batch_norm(g_de_conv5,mode=mode,name='g_de_bn5')
        g_de_relu5 = relu(g_de_bn5,name='g_de_relu5')
        g_de5 = tf.concat([g_de_relu5,g_lrelu5],3)

        g_de_conv4 = deconv2d(g_de5,shape=shape4,d_shape=d_shape4,name='g_de_conv4')
        g_de_bn4 = batch_norm(g_de_conv4,mode=mode,name='g_de_bn4')
        g_de_relu4 = relu(g_de_bn4,name='g_de_relu4')
        g_de4 = tf.concat([g_de_relu4,g_lrelu4],3)

        g_de_conv3 = deconv2d(g_de4,shape=shape3,d_shape=d_shape3,name='g_de_conv3')
        g_de_bn3 = batch_norm(g_de_conv3,mode=mode,name='g_de_bn3')
        g_de_relu3 = relu(g_de_bn3,name='g_de_relu3')
        g_de3 = tf.concat([g_de_relu3,g_lrelu3],3)

        g_de_conv2 = deconv2d(g_de3,shape=shape2,d_shape=d_shape2,name='g_de_conv2')
        g_de_bn2 = batch_norm(g_de_conv2,mode=mode,name='g_de_bn2')
        g_de_relu2 = relu(g_de_bn2,name='g_de_relu2')
        g_de2 = tf.concat([g_de_relu2,g_lrelu2],3)

        g_de_conv1 = deconv2d(g_de2,shape=shape1,d_shape=d_shape1,name='g_de_conv1')
        g_de_bn1 = batch_norm(g_de_conv1,mode=mode,name='g_de_bn1')
        g_de_relu1 = relu(g_de_bn1,name='g_de_relu1')
        g_de1 = tf.concat([g_de_relu1,g_lrelu1],3)

        g_decode = deconv2d(g_de1,shape=shape0,d_shape=d_shape0,name='g_decode')
        g_decode_bn = batch_norm(g_decode,mode=mode,name='g_decode_bn')
        g_decode_relu = relu(g_decode_bn,name='g_decode_relu')
        
        g_out = conv2d(g_decode_relu,output_dim=3,strides=[1,1,1,1],name='g_out')
        
        g_activate = tf.nn.tanh(g_out,name='g_tanh')
        return g_activate

# discriminator 
def discriminator(image,reuse=False,mode='train',name='discriminator'):
    with tf.name_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        d_conv1 = conv2d(image,output_dim=64,name='d_conv1')
        d_bn1 = batch_norm(d_conv1,mode=mode,name='d_bn1')
        d_lrelu1 = lrelu(d_bn1,name='d_lrelu1')

        d_conv2 = conv2d(d_lrelu1,output_dim=128,name='d_conv2')
        d_bn2 = batch_norm(d_conv2,mode=mode,name='d_bn2')
        d_lrelu2 = lrelu(d_bn2,name='d_lrelu2')

        d_conv3 = conv2d(d_lrelu2,output_dim=256,name='d_conv3')
        d_bn3 = batch_norm(d_conv3,mode=mode,name='d_bn3')
        d_lrelu3 = lrelu(d_bn3,name='d_lrelu3')

        d_conv4 = conv2d(d_lrelu3,output_dim=512,name='d_conv4')
        d_bn4 = batch_norm(d_conv4,mode=mode,name='d_bn4')
        d_lrelu4 = lrelu(d_bn4,name='d_lrelu4')
        
        d_out = conv2d(d_lrelu4,output_dim=1,strides=[1,1,1,1],name='d_out')
       
        return tf.nn.sigmoid(d_out),d_out

# training process for our model
def train(epoch=20,lba=100,load_model=False):
    with tf.variable_scope('graph'):
        global_step = tf.Variable(0,name='global_step',trainable=False)

        data_dir = os.path.abspath('..')+'/data/'
        log_dir = os.path.abspath('..')+'/log/'
        save_dir = os.path.abspath('..')+'/outcome/'
        
        train_dir = data_dir + 'train'
        test_dir = data_dir + 'test'
        # training data
        gray_img,color_img = inputs(train_dir,BATCH_SIZE,6)   
        # testing data
        test_gray,test_color = inputs(test_dir,BATCH_SIZE,2)
        
        in_holder = tf.placeholder(tf.float32,shape=[None,None,None,1],name='in_holder')
        
        d_input_R = tf.concat([gray_img,color_img],3)
        D_R,D_logits_R = discriminator(d_input_R)

        G = generator(gray_img)
        d_input_F = tf.concat([gray_img,G],3)
        D_F,D_logits_F = discriminator(d_input_F,reuse=True)
        
        # sample test data
        samples = sampler(in_holder)
        test_l2_error = tf.reduce_mean(tf.abs(samples-color_img)**2)
        
        # loss for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_R,labels=tf.ones_like(D_R)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_F,labels=tf.zeros_like(D_F)))

        d_loss = d_loss_real+d_loss_fake
        
        # loss for generator
        g_loss_prim = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_F,labels=tf.ones_like(D_F)))
        g_loss_norm = lba*tf.reduce_mean(tf.abs(G-color_img))
        
        g_loss = g_loss_prim + g_loss_norm
        
        # tensorflow summary operations
        DR_sum = tf.summary.histogram('D_real',D_R)
        DF_sum = tf.summary.histogram('D_fake',D_F)
        g_sum = tf.summary.image('gray_input',gray_img)
        G_sum = tf.summary.image('G',G)

        d_loss_sum = tf.summary.scalar('d_loss',d_loss)
        g_loss_sum = tf.summary.scalar('g_loss',g_loss)
        test_sum = tf.summary.scalar('validation_l2_error',test_l2_error)
    
        g_sum = tf.summary.merge([g_sum,DF_sum,G_sum,g_loss_sum,test_sum])
        d_sum = tf.summary.merge([DR_sum,d_loss_sum])

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        saver = tf.train.Saver()
        
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        d_optim = tf.train.AdamOptimizer(LR,beta1=0.5).minimize(d_loss,var_list=d_vars,global_step=global_step)
        g_optim = tf.train.AdamOptimizer(LR,beta1=0.5).minimize(g_loss,var_list=g_vars,global_step=global_step)
    
    with tf.variable_scope('graph'):
        model_name = 'model_{}'.format(int(time.time()))

        sess = tf.InteractiveSession()
        writer = tf.summary.FileWriter(log_dir+model_name,sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init)

        if load_model:
            print('Reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(log_dir+model_name)

            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess,os.path.join(log_dir+model_name,ckpt_name))
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]


        for epo in range(epoch):
            batch_idx = 300

            for idx in range(batch_idx):
                _,summary_d = sess.run([d_optim,d_sum])
                writer.add_summary(summary_d,epo*batch_idx+idx)

                sess.run([g_optim])
                #writer.add_summary(summary_g,epo*batch_idx+idx)
                test_gray_img = test_gray.eval(session=sess)
                _,summary_g = sess.run([g_optim,g_sum],feed_dict={in_holder:test_gray_img})
                writer.add_summary(summary_g,epo*batch_idx+idx)

                errD_fake = d_loss_fake.eval()
                errD_real = d_loss_real.eval()
                errG = g_loss.eval()

                if idx % 20 == 0:
                    print('epoch:{} [{}/{}] d_loss:{}, g_loss:{}'.format(epo,idx,batch_idx,errD_fake+errD_real,errG))

                if idx % 100  == 0:
                    sample = sess.run(samples,feed_dict={in_holder:test_gray_img})
                    sample_path = save_dir+'/samples/'
                    save_images(sample,[5,5],sample_path + 'sample_{}_epoch_{}.jpg'.format(epo,idx))

                    print('===========    {}_epoch_{}.jpg save down    ==========='.format(epo, idx))


                if (idx % 100 == 0) or (idx+1 == batch_idx):
                    checkpoint_path = os.path.join(log_dir+model_name,'my_GAN.ckpt')
                    saver.save(sess,checkpoint_path,global_step=global_step)

                    print('===========    model saved    ===========' )
                    

            
        coord.request_stop()
        coord.join(threads)
        sess.close()

def main():
    train()

if __name__ == '__main__':
    main()
