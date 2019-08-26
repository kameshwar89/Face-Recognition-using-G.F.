from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import argparse
import importlib
import time

sys.path.insert(1, "../src")
import facenet
import numpy as np
from sklearn.datasets import load_files
import tensorflow as tf
from six.moves import xrange

def main(args):

	with tf.Graph().as_default():

		with tf.Session() as sess:

			output_dir = os.path.expanduser(args.output_dir)
			if not os.path.isdir(output_dir):
				os.makedirs(output_dir)

			print("Loading trained model...\n")
			meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.trained_model_dir))
			facenet.load_model(args.trained_model_dir, meta_file, ckpt_file)

			print("Finding image paths and targets...\n")
			data = load_files(args.data_dir, load_content=False, shuffle=False)
			labels_array = data['target']
			paths = data['filenames']

			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

			image_size = images_placeholder.get_shape()[1]
			embedding_size = embeddings.get_shape()[1]

			print('Generating embeddings from images...\n')
			start_time = time.time()
			batch_size = args.batch_size
			nrof_images = len(paths)
			nrof_batches = int(np.ceil(1.0*nrof_images / batch_size))
			emb_array = np.zeros((nrof_images, embedding_size))
			for i in xrange(nrof_batches):
				start_index = i*batch_size
				end_index = min((i+1)*batch_size, nrof_images)
				paths_batch = paths[start_index:end_index]
				images = facenet.load_data(paths_batch, do_random_crop=False, do_random_flip=False, image_size=image_size, do_prewhiten=True)
				feed_dict = { images_placeholder:images, phase_train_placeholder:False}
				emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

			time_avg_forward_pass = (time.time() - start_time) / float(nrof_images)
			print("Forward pass took avg of %.3f[seconds/image] for %d images\n" % (time_avg_forward_pass, nrof_images))

			print("Finally saving embeddings and gallery to: %s" % (output_dir))
			np.save(os.path.join(output_dir, "gallery.npy"), labels_array)
			np.save(os.path.join(output_dir, "signatures.npy"), emb_array)

def parse_arguments(argv):
	parser = argparse.ArgumentParser(description="Batch-represent face embeddings from a given data directory")
	parser.add_argument('-d', '--data_dir', type=str,
		help='directory of images with structure as seen at the top of this file.')
	parser.add_argument('-o', '--output_dir', type=str,
		help='directory containing aligned face patches with file structure as seen at the top of this file.')
	parser.add_argument('--trained_model_dir', type=str,
        help='Load a trained model before training starts.')
	parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.', default=50)

	return parser.parse_args(argv)


if __name__ == "__main__":
	main(parse_arguments(sys.argv[1:]))
