import tensorflow as tf
import numpy as np
import os
import time
import datetime
from cloud_cnnbag.trainer import input_generator as i_gen
from cloud_cnnbag.trainer import simple_cnnbag_cloud as cnn
from cloud_cnnbag.trainer.train_cnnbag_local import Model_Trainer
import argparse
#tf.contrib.data.Dataset



def predict(checkpoint_file, input, outdir, word_embd, other_embd):
    #checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    all_predictions = []
    trainer = Model_Trainer()
    num_ex = trainer.get_num_examples([input])[0]
    num_batch = 64
    max_steps = int(np.ceil(num_ex/num_batch))
    print('Max steps:', max_steps)
    sess1 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    print('Init embeddings')
    with tf.device('/cpu:0'):
        embedding_weights = tf.get_variable('embedding_weights', (71608, 500), trainable=False,
                                                initializer=tf.zeros_initializer)
        random_label = tf.get_variable('random_label', (9, 100), trainable=False, initializer=tf.zeros_initializer)
        random_distance = tf.get_variable('random_distance', (201, 100), trainable=False,initializer=tf.zeros_initializer)
    with tf.device('/cpu:0'):
        saver = tf.train.Saver([embedding_weights])
        saver.restore(sess1, os.path.join(word_embd, 'embedding_weights.ckpt'))
        saver = tf.train.Saver([random_distance, random_label])
        saver.restore(sess1, os.path.join(other_embd, 'random_distance_labels.ckpt'))
    embeddings = (embedding_weights, random_label, random_distance)
    iterator, features, labels, names = i_gen.create_input(embeddings, input, num_batch, 0)
    sess1.run(iterator.initializer)
    print('Running predictions')
    graph = tf.Graph()

    with graph.as_default():
        #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            count = 0
            #tf.reset_default_graph()
            print('Loading graph')
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            print('Loading vars')
            input = graph.get_operation_by_name('cnn/input_x').outputs[0]
            label = graph.get_operation_by_name('cnn/input_y').outputs[0]
            dropout_keep = graph.get_operation_by_name('cnn/Placeholder').outputs[0]
            batch_size = graph.get_operation_by_name('cnn/Placeholder_1').outputs[0]
            prediction_1 = graph.get_operation_by_name('cnn/output/predicted_score_class1').outputs[0]
            prediction_class = graph.get_operation_by_name('cnn/output/predicted_class').outputs[0]
            while True:
            #for i in range(20):
                try:
                    print('Step %d of %d'%(count, max_steps))
                    input_vals, label_vals, names_vals = sess1.run([features['input'], labels, names])
                    feed_dict = {input: input_vals,
                                 label: label_vals,
                                 batch_size: num_batch,
                                 dropout_keep: 1.0}
                    predictions, curlabels = sess.run([prediction_1,prediction_class], feed_dict)
                    all_predictions.extend(zip(names_vals, predictions, curlabels))
                    count+=1
                except tf.errors.OutOfRangeError:
                    break

    pred_strs = ['\t'.join([str(p[0]), str(p[1][0]), str(p[2])]) for p in all_predictions]
    out_str = '\n'.join(pred_strs)
    with open(os.path.join(outdir, 'results.txt'), 'w') as f:
        f.write(out_str)
    sess1.close()


def main():
    predict(args.checkpoint, args.input, args.outdir, args.word_embd, args.other_embd)

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
    default='C:/Users/ttasn/Documents/ASU/Research/PhD/deeplearning/phd/output/cnn_bag_large_10_28/cnnbag_large_1/cnnmodel.ckpt-3262')
    parser.add_argument('--input', default='C:/Users/ttasn/Documents/ASU/Research/PhD/deeplearning/phd/eval/textprocessor/examples/tfrecords.tfrecords')
    parser.add_argument('--outdir', default='eval')
    parser.add_argument('--word_embd', default='C:/Users/ttasn/Documents/ASU/Research/PhD/deeplearning/dataset_creation/data/embd_all2/')
    parser.add_argument('--other_embd', default = 'C:/Users/ttasn/Documents/ASU/Research/PhD/deeplearning/dataset_creation/data/embd_pos_label/')
    args = parser.parse_args()
    main()