import tensorflow as tf
import os 
import scipy.misc
import numpy as np

HEIGHT = 175
WIDTH = 250

# read image pairs from tfrecords file
def read_and_decode(filename_queue,batch_size):
	reader = tf.TFRecordReader()
	_,serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(
		serialized_example,
		features={
			'image_raw':tf.FixedLenFeature([], tf.string),
			'label_raw':tf.FixedLenFeature([], tf.string)
		})

	image = tf.decode_raw(features['image_raw'],tf.uint8)
	label = tf.decode_raw(features['label_raw'],tf.uint8)

	image = tf.cast(image,tf.float32)/255.0
	label = tf.cast(label,tf.float32)/255.0

	image = tf.reshape(image,[HEIGHT,WIDTH,1])
	label = tf.reshape(label,[HEIGHT,WIDTH,3])

	images,labels = tf.train.shuffle_batch([image,label],batch_size=batch_size,
    										capacity=1000 + 3 * batch_size,min_after_dequeue = 1000)

	return images,labels

# generate image batches for training and testing
def inputs(data_dir,batch_size,num=1,name='input'):
	with tf.name_scope(name):
		filenames=[os.path.join(data_dir,'g_c_pairs{}.tfrecords'.format(i+1)) for i in range(num)]
		filename_queue = tf.train.string_input_producer(filenames)

		gray_img,color_img = read_and_decode(filename_queue,batch_size)

		return gray_img,color_img

# save testing result 
def save_images(images, size, path):
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w, :] = image
        
    return scipy.misc.imsave(path, merge_img)