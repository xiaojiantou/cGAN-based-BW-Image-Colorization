import os 
import time
from PIL import Image
import tensorflow as tf
import numpy as np

HEIGHT = 175
WEIDTH = 250

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecords(g_path,c_path,name):
	'''
	Convert a dataset to tfrecords
	'''
	for i in range(8):
		writer = tf.python_io.TFRecordWriter(name+str(i+1)+'.tfrecords')
		for img_name in os.listdir(g_path)[i*1000:(i+1)*1000]:
			img_path = g_path + img_name
			label_path = c_path + img_name

			img = Image.open(img_path)
			img_raw = img.tobytes()

			label = Image.open(label_path)
			if np.array(label).shape[-1] is not 3:
				continue
			label_raw = label.tobytes()

			example = tf.train.Example(features=tf.train.Features(feature=
										{'image_raw':_bytes_feature(img_raw),
										 'label_raw':_bytes_feature(label_raw)}))
			writer.write(example.SerializeToString())
		writer.close()

if __name__ == '__main__':
	c_image_folder = 'colorful\\'
	g_image_folder = 'grayscale\\'

	name = 'g_c_pairs'

	# convert grayscale and colorful image pairs to tfrecords
	convert_to_tfrecords(g_image_folder,c_image_folder,name)

	print('covert done')