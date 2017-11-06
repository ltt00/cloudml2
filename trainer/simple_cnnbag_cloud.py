from __future__ import division, print_function, absolute_import
import tensorflow as tf
from enum import Enum




def _variable_on_gpu(name, shape, initializer):
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, initializer, wd_factor):
    var = _variable_on_gpu(name, shape, initializer)
    if wd_factor is not None and wd_factor != 0.:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd_factor, name='weight_loss')
    else:
        weight_decay = tf.constant(0.0, dtype=tf.float32)
    return var, weight_decay

class Model(object):
    def __init__(self, config, numtrainbatch):
        self.emb_size = config['embd_size']
        self.num_kernel = config['num_kernel']
        self.min_window = config['min_window']
        self.max_window = config['max_window']
        self.num_classes = config['num_classes']
        self.l2_reg = config['l2_reg']
        self._initlr = config['init_lr']
        self.optimizer = config['optimizer']
        self.decay_step = numtrainbatch
        self.build_graph()

    def build_graph(self):
        losses = []
        self.dropout_keep = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32)
        self._inputs = tf.placeholder(tf.float32, [None, None, self.emb_size, 1], name="input_x")
        self._labels = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        # conv + pooling layer
        pool_tensors = []
        for k_size in range(self.min_window, self.max_window + 1):
            with tf.variable_scope('conv-%d' % k_size) as scope:
                kernel, wd = _variable_with_weight_decay(
                    name='kernel-%d' % k_size,
                    shape=[k_size, self.emb_size, 1, self.num_kernel],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    wd_factor=self.l2_reg)
                losses.append(wd)
                conv = tf.nn.conv2d(input=self._inputs, filter=kernel, strides=[1, 1, 1, 1], padding='VALID')
                biases = _variable_on_gpu(name='bias-%d' % k_size,
                                          shape=[self.num_kernel],
                                          initializer=tf.constant_initializer(0.01))
                conv_plus_b = tf.nn.bias_add(conv, biases)
                activation = tf.nn.relu(conv_plus_b, name=scope.name)
                pool = tf.reduce_max(activation, axis=1, keep_dims=True)
                pool = tf.squeeze(pool, squeeze_dims=[1, 2])
                pool_tensors.append(pool)

        self.pool_flat = tf.concat(values=pool_tensors, axis=1, name='pool')
        num_filters = self.max_window - self.min_window + 1
        pool_size = num_filters * self.num_kernel
        pool_dropout = tf.nn.dropout(self.pool_flat,self.dropout_keep)
        #pool_dropout = tf.cond(tf.equal(self.mode,tf.constant(0)),lambda: tf.nn.dropout(self.pool_flat,self.dropout_keep), lambda: self.pool_flat)
        """
        if self.mode==runmodes.train and self.dropout_keep > 0:
            pool_dropout = tf.nn.dropout(self.pool_flat, self.dropout_keep)
        else:
            pool_dropout = self.pool_flat
        """
        # fully-connected layer
        with tf.variable_scope('output') as scope:
            W, wd = _variable_with_weight_decay('W', shape=[pool_size, self.num_classes],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                wd_factor=self.l2_reg)
            losses.append(wd)
            biases = _variable_on_gpu('bias', shape=[self.num_classes],
                                      initializer=tf.constant_initializer(0.01))
            self.logits = tf.nn.bias_add(tf.matmul(pool_dropout, W), biases, name='logits')
            scores = tf.nn.softmax(self.logits, name = "predicted_softmax_score")
            scores_class1 = tf.slice(scores, [0,1], [-1,1], name="predicted_score_class1")
            classes = tf.argmax(input=self.logits, axis=1, name = "predicted_class")
            self._predictions = { "classes": classes,
                "scores": scores,
                "scores_class1": scores_class1
            }

        with tf.variable_scope('evaluation') as scope:
            self._labels_class1 = tf.argmax(self._labels, 1)
            # thresholds = [0.1*i for i in range(0,11,1)]
            thresholds = [0.02 * i for i in range(50, -1, -1)]
            precisions_tf, precisions_op_tf = tf.metrics.precision_at_thresholds(self._labels_class1, scores_class1,
                                                                                 thresholds)
            recalls_tf, recalls_op_tf = tf.metrics.recall_at_thresholds(self._labels_class1, scores_class1, thresholds)
            auc_tf, auc_op_tf = tf.metrics.auc(labels=self._labels_class1, predictions=scores_class1, curve='PR')
            self._eval_op = {'precisions_op': precisions_op_tf, 'recalls_op': recalls_op_tf, 'auc_op': auc_op_tf}
            self._eval_val = {'precisions': precisions_tf, 'recalls_': recalls_tf, 'aucs': auc_tf}

        with tf.variable_scope('loss') as scope:
            self.global_step_lrdecay = tf.Variable(0, trainable=False)
            self._learning_rate = tf.train.exponential_decay(self._initlr, self.global_step_lrdecay, self.decay_step, 0.96,
                                                             staircase=True)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self._labels,
                                                                    name='cross_entropy_per_example')
            self.cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
            losses.append(self.cross_entropy_loss)
            self._total_loss = tf.add_n(losses, name='total_loss')
            opt = self.get_optimizer(self._learning_rate)
            self._train_op = opt.minimize(loss=self._total_loss, global_step=self.global_step_lrdecay)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

        #isPredict = tf.cond(tf.equal(self.mode,tf.constant(2)),lambda:tf.no_op(), lambda:self.not_predict(losses, scores_class1))
        return

    """
    def not_predict(self, losses, scores_class1):
        # loss
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self._labels,
                                                                    name='cross_entropy_per_example')
            self.cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
            losses.append(self.cross_entropy_loss)
            self._total_loss = tf.add_n(losses, name='total_loss')
            y=tf.cond(tf.equal(self.mode, tf.constant(0)),lambda:self.optimize(), lambda:self.not_optimize())

        with tf.variable_scope('evaluation') as scope:
                self._labels_class1 = tf.argmax(self._labels, 1)
                #thresholds = [0.1*i for i in range(0,11,1)]
                thresholds = [0.02*i for i in range(50,-1,-1)]
                precisions_tf, precisions_op_tf= tf.metrics.precision_at_thresholds(self._labels_class1, scores_class1, thresholds)
                recalls_tf, recalls_op_tf = tf.metrics.recall_at_thresholds(self._labels_class1, scores_class1, thresholds)
                auc_tf, auc_op_tf = tf.metrics.auc(labels=self._labels_class1,predictions=scores_class1,curve='PR')
                self._eval_op = {'precisions_op':precisions_op_tf,'recalls_op':recalls_op_tf, 'auc_op':auc_op_tf}
                self._eval_val = {'precisions': precisions_tf, 'recalls_': recalls_tf, 'aucs': auc_tf}

        return tf.no_op()

    def optimize(self):
        opt = self.get_optimizer(self._learning_rate)
        self._train_op = opt.minimize(loss=self._total_loss, global_step=self.global_step_lrdecay)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        return tf.no_op()

    def not_optimize(self):
        self._train_op = tf.no_op()
        return tf.no_op()
    """

    def get_optimizer(self, learning_rate):
        if self.optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate)
        elif self.optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif self.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif self.optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError("Optimizer not supported.")
        return opt


    # @property
    def inputs(self):
        return self._inputs

    # @property
    def labels(self):
        return self._labels

    @property
    def initlr(self):
        return self._initlr

    @property
    def train_op(self):
        return self._train_op

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def eval_op(self):
        return self._eval_op

    @property
    def eval_val(self):
        return self._eval_val

    @property
    def predictions(self):
        return self._predictions

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def labels_class1(self):
        return self._labels_class1


