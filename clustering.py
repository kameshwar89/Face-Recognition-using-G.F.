import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import os
import math
def face_distance(face_encodings, face_to_compare):
    import numpy as np
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.sum(face_encodings*face_to_compare,axis=1)

def load_model(model_dir, meta_file, ckpt_file):
    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))

def _chinese_whispers(encoding_list, threshold=0.55, iterations=20):
    
    from random import shuffle
    import networkx as nx
    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        print ("No enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        node_id = idx+1

        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
        nodes.append(node)

        if (idx+1) >= len(encodings):
            break

        compare_encodings = encodings[idx+1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                edge_id = idx+i+2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    for _ in range(0, iterations):
        cluster_nodes = G.nodes()
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if G.node[ne]['cluster'] in clusters:
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            edge_weight_sum = 0
            max_cluster = 0
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            G.node[node]['cluster'] = max_cluster

    clusters = {}

    for (_, data) in G.node.items():
        cluster = data['cluster']
        path = data['path']

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters

def cluster_facial_encodings(facial_encodings):

    if len(facial_encodings) <= 1:
        print ("Number of facial encodings must be greater than one, can't cluster")
        return []

    sorted_clusters = _chinese_whispers(facial_encodings.items())
    return sorted_clusters

def compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
                    embedding_size,nrof_images,nrof_batches,emb_array,batch_size,paths):

    for i in range(nrof_batches):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)
        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

    facial_encodings = {}
    for x in range(nrof_images):
        facial_encodings[paths[x]] = emb_array[x,:]


    return facial_encodings

def get_onedir(paths):
    dataset = []
    path_exp = os.path.expanduser(paths)
    if os.path.isdir(path_exp):
        images = os.listdir(path_exp)
        image_paths = [os.path.join(path_exp,img) for img in images]

        for x in image_paths:
            if os.path.getsize(x)>0:
                dataset.append(x)
        
    return dataset 


def main(args):

    from os.path import join, basename, exists
    from os import makedirs
    import numpy as np
    import shutil
    import sys

    if not exists(args.output):
        makedirs(args.output)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            image_paths = get_onedir(args.input)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            load_model(args.model_dir, meta_file, ckpt_file)
        
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            image_size = images_placeholder.get_shape()[1]
            print("image_size:",image_size)
            embedding_size = embeddings.get_shape()[1]
            print('Runnning forward pass on images') 

            nrof_images = len(image_paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            facial_encodings = compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
                embedding_size,nrof_images,nrof_batches,emb_array,args.batch_size,image_paths)
            sorted_clusters = cluster_facial_encodings(facial_encodings)
            num_cluster = len(sorted_clusters)
                
            for idx, cluster in enumerate(sorted_clusters):
                cluster_dir = join(args.output, str(idx))
                if not exists(cluster_dir):
                    makedirs(cluster_dir)
                for path in cluster:
                    shutil.copy(path, join(cluster_dir, basename(path)))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
    parser.add_argument('--model_dir', type=str, help='model dir', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size', required=30)
    parser.add_argument('--input', type=str, help='Input dir of images', required=True)
    parser.add_argument('--output', type=str, help='Output dir of clusters', required=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(parse_args())
