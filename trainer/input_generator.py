from __future__ import division, print_function, absolute_import
import tensorflow as tf



def parser(example_proto):
    # Define how to parse the example
    cont_features = {
        "numsent": tf.FixedLenFeature([], dtype=tf.int64),
        "class": tf.FixedLenFeature([], dtype=tf.int64),
        "name": tf.FixedLenFeature([], dtype=tf.string)
    }
    seq_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "distances":tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "lengths":tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "sections":tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    # Parse the example
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        example_proto,
        context_features=cont_features,
        sequence_features=seq_features
    )
    classname =tf.cast(context_parsed["class"], tf.int32)
    #tokens = tf.nn.embedding_lookup(embedding_weights, sequence_parsed['tokens'])
    #labels = tf.nn.embedding_lookup(random_label, sequence_parsed['labels'])
    #distances = tf.nn.embedding_lookup(random_distance, sequence_parsed['distances'])
    #input_tensor = tf.concat([tokens, labels, distances],1)
    tokens = sequence_parsed['tokens']
    labels = sequence_parsed['labels']
    distances = sequence_parsed['distances']
    length_tensor = sequence_parsed['lengths']
    section_tensor = tf.one_hot(sequence_parsed['sections'],3)
    #print(input_tensor.shape, length_tensor.shape, section_tensor.shape, classname.shape)
    return tokens, labels, distances, length_tensor, section_tensor, classname, context_parsed["name"]
    #return input_tensor, length_tensor, section_tensor, classname
    #return {"input": input_tensor, "length": length_tensor, "section":section_tensor}, classname


def get_len(path):
    record_iterator = tf.python_io.tf_record_iterator(path=path)
    curlen = 0
    for serialized_example in record_iterator:
        curlen += 1
    return tf.cast(curlen,tf.int32)

def create_input(weights, path, batch_size, shuffle_num):
    #with tf.device('/gpu:0'):
    dataset =  tf.contrib.data.TFRecordDataset(path)
    dataset = dataset.map(parser)
    #dataset = dataset.padded_batch(batch_size, padded_shapes=([None,700],[None],[None, 3],[]))
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None],[None],[None], [None],[None, 3],[], []))
    if shuffle_num>0:
        dataset = dataset.shuffle(shuffle_num)
    iterator = dataset.make_initializable_iterator()
    #input_tensors, length_tensors, section_tensors, classes =  iterator.get_next()
    tokens, labels, distances, length_tensors, section_tensors, classes, name = iterator.get_next()
    with tf.device('/cpu:0'):
        tokens = tf.nn.embedding_lookup(weights[0], tokens)
        labels = tf.nn.embedding_lookup(weights[1], labels)
        distances = tf.nn.embedding_lookup(weights[2], distances)
        input_tensors = tf.concat([tokens, labels, distances],2)
        input_tensors = tf.expand_dims(input_tensors, -1)
        one_hot_classes = tf.one_hot(classes, 2)
    return iterator, {"input": input_tensors, "length": length_tensors, "section":section_tensors}, one_hot_classes, name

def test():
    trainpath= 'C:/Users/ttasn/Documents/ASU/Research/PhD/deeplearning/dataset_creation/data/model_type_1/sample/minisample/'
    validpath= 'C:/Users/ttasn/Documents/ASU/Research/PhD/deeplearning/dataset_creation/data/model_type_1/sample/minisample/'
    with tf.Session() as sess:
        train_size = sess.run(get_len(trainpath))
        print('train_size', train_size)
        valid_size = sess.run(get_len(validpath))
        print('valid_size', valid_size)