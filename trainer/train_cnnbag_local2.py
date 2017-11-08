from __future__ import division, print_function, absolute_import
from datetime import datetime
import time
import os
#import logging
#logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import numpy as np
from  cloud_cnnbag.trainer import input_generator as i_gen
from cloud_cnnbag.trainer import simple_cnnbag_cloud as scnn
from sklearn import metrics as skmetrics
import matplotlib
import logging
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
tf.logging.set_verbosity(tf.logging.INFO)


class Model_Trainer:


    def init_embeddings(self, sess, embedding_weights, random_label, random_distance):
        saver = tf.train.Saver([embedding_weights])
        vizpath = self.config['word_embd_dir']
        saver.restore(sess, os.path.join(vizpath, 'embedding_weights.ckpt'))
        saver = tf.train.Saver([random_distance, random_label])
        vizpath2 = self.config['pos_label_embd_dir']
        saver.restore(sess, os.path.join(vizpath2, 'random_distance_labels.ckpt'))

    def get_outdir(self, outdir_path):
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(outdir_path, timestamp))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        return out_dir

    def set_config(self, arguments):
        self.config=dict()
        self.config['trainpath'] = arguments.trainpath
        self.config['evalpath'] = arguments.evalpath
        self.config['jobdir'] = self.get_outdir(arguments.job_dir)
        #self.config['jobdir'] = arguments.job_dir
        self.config['word_embd_dir'] = arguments.embdworddir
        self.config['pos_label_embd_dir'] = arguments.embdposlabeldir
        self.config['embd_size'] = arguments.embdsize
        self.config['num_classes'] = arguments.numclasses
        self.config['train_batch_size'] = arguments.trainbatchsize
        self.config['eval_batch_size'] = arguments.evalbatchsize
        self.config['num_shuffle'] = arguments.numshuffle
        self.config['numepochs'] = arguments.numepochs
        self.config['optimizer'] = arguments.optimizer
        self.config['init_lr'] = arguments.lr
        self.config['l2_reg'] = arguments.l2reg
        self.config['dropout'] = arguments.dropout
        self.config['num_kernel'] = arguments.numfilters
        self.config['min_window'] = arguments.minwin
        self.config['max_window'] = arguments.maxwin
        self.config['summary_step'] = arguments.summaryfreq
        self.config['checkpoint_step'] = arguments.checkpointfreq
        self.config['log_step'] = 5


    def dev_step(self, mtest, iterator, num_valid_batch, valid_features, valid_labels, global_step, sess):
        dev_loss = []
        #dev_auc = []
        #dev_f1_score = []
        all_scores = []
        all_labels = []
        all_predicted_classes = []
        sess.run(iterator.initializer)
        for i in range(num_valid_batch):
            try:
                valid_input_vals, valid_label_vals = sess.run([valid_features['input'], valid_labels])
                feed_dict = {mtest._inputs: valid_input_vals,
                             mtest._labels: valid_label_vals,
                             mtest.batch_size: self.config['eval_batch_size'],
                             mtest.dropout_keep: 1.0}
                loss_value, predictions, curlabels = sess.run([mtest._total_loss, mtest._predictions, mtest._labels_class1], feed_dict)
                dev_loss.append(loss_value)
                all_scores.extend(predictions['scores_class1'].flatten().tolist())
                all_labels.extend(curlabels)
                all_predicted_classes.extend(predictions['classes'].flatten().tolist())
                #prec, rec, thresholds = skmetrics.precision_recall_curve(curlabels,predictions['scores_class1'])
                #self.plot_pr(rec, prec, 'SkLearn Validation Set Precision-Recall Curve')
                # try:
                #     auc = skmetrics.auc(rec, prec)
                # except:
                #     auc=0
                # if np.isnan(auc):
                #     auc=0.0
                # dev_auc.append(auc)
                # try:
                #     f1score = skmetrics.f1_score(curlabels, predictions['classes'])
                # except:
                #     f1score=0
                # dev_f1_score.append(f1score)
                # if global_step == self.max_train_steps and i==(num_valid_batch-1):
                #     self.plot_pr(rec, prec, 'SkLearn Validation Set Precision-Recall Curve', 'pr_valid.png')
            except tf.errors.OutOfRangeError:
                break
        prec, rec, thresholds = skmetrics.precision_recall_curve(all_labels, all_scores)
        tpr, fpr, thresholds = skmetrics.roc_curve(all_labels, all_scores)
        try:
            auc = skmetrics.auc(rec, prec)
        except:
            auc=0
        if np.isnan(auc):
            auc=0.0
        try:
            f1score = skmetrics.f1_score(all_labels, all_predicted_classes)
            recall = skmetrics.recall_score(all_labels, all_predicted_classes)
            precision = skmetrics.precision_score(all_labels, all_predicted_classes)
            average_precision_score = skmetrics.average_precision_score(all_labels, all_scores)
            roc_auc_score = skmetrics.roc_auc_score(all_labels, all_scores)
        except:
            f1score=0.0
            recall = 0.0
            precision = 0.0
            average_precision_score =0.0
            roc_auc_score =0.0

        # dev_f1_score.append(f1score)
        if global_step == self.max_train_steps:
            self.plot_pr(rec, prec, 'SkLearn Validation Set Precision-Recall Curve', 'pr_valid.png')
            self.plot_roc(fpr, tpr, 'SkLearn Validation Set ROC Curve', 'roc_valid.png')
        return np.mean(dev_loss), auc, f1score, recall, precision, average_precision_score, roc_auc_score

    def get_num_examples(self, paths):
        sizes=[]
        with tf.Session() as sess:
            for path in paths:
                size = sess.run(i_gen.get_len(path))
                sizes.append(size)
        return sizes

    def plot_pr(self, recall, precision, title,name):
        #plt.draw()
        fig = plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(recall, precision, 'b', label='Precision-Recall curve', linestyle='-', marker='o', color='b')
        plt.title(title)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        savepath = os.path.join(self.config['jobdir'], name)
        plt.savefig(savepath, dpi=fig.dpi)

    def plot_roc(self, fpr, tpr, title,name):
        #plt.draw()
        fig = plt.figure()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.plot(fpr, tpr, 'b', label='ROC curve', linestyle='-', marker='o', color='b')
        plt.title(title)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        savepath = os.path.join(self.config['jobdir'], name)
        plt.savefig(savepath, dpi=fig.dpi)

    def get_embeddings(self):
        with tf.device('/cpu:0'):
            embedding_weights = tf.get_variable('embedding_weights', (71608, 500), trainable=False,
                                                initializer=tf.zeros_initializer)
            random_label = tf.get_variable('random_label', (9, 100), trainable=False, initializer=tf.zeros_initializer)
            random_distance = tf.get_variable('random_distance', (201, 100), trainable=False,
                                              initializer=tf.zeros_initializer)
        return (embedding_weights, random_label, random_distance)

    def test_skmetrics(self, curlabels, predictions, global_step):
        skprec, skrec, skthresholds = skmetrics.precision_recall_curve(curlabels, predictions['scores_class1'])
        auc = skmetrics.auc(skrec, skprec)
        f1score = skmetrics.f1_score(curlabels, predictions['classes'])
        self.plot_pr(skrec, skprec,'SkLearn Train Step %d Precision-Recall Curve'%global_step)
        return f1score


    def train(self):
        train_size, valid_size = self.get_num_examples((self.config['trainpath'], self.config['evalpath']))
        #graph
        with tf.Graph().as_default():
            embeddings = self.get_embeddings()
            self.max_train_steps = int(np.ceil(train_size/self.config['train_batch_size'])*self.config['numepochs'])
            self.num_valid_batch =  int(np.ceil(valid_size/self.config['eval_batch_size']))
            num_train_batch = int(np.ceil(train_size/self.config['train_batch_size']))
            with tf.variable_scope('inputs'):
                train_iterator, train_features, train_labels, train_names = i_gen.create_input(embeddings, self.config['trainpath'],
                                                                                        self.config['train_batch_size'],
                                                                                        self.config['num_shuffle'])
                valid_iterator, valid_features, valid_labels, valid_names = i_gen.create_input(embeddings, self.config['evalpath'],
                                                                                        self.config['eval_batch_size'],
                                                                                        self.config['num_shuffle'])
            with tf.variable_scope('cnn'):
                m = scnn.Model(self.config, num_train_batch)


            # checkpoint
            saver = tf.train.Saver(tf.global_variables())
            save_path = os.path.join(self.config['jobdir'], 'cnnmodel.ckpt')
            summary_op = tf.summary.merge_all()

            # session
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)).as_default() as sess:
                train_summary_writer = tf.summary.FileWriter(os.path.join(self.config['jobdir'], "logs/train"), graph=sess.graph)
                dev_summary_writer = tf.summary.FileWriter(os.path.join(self.config['jobdir'], "logs/dev"), graph=sess.graph)
                sess.run([
                    tf.local_variables_initializer(),
                    tf.global_variables_initializer(),
                ])
                self.init_embeddings(sess, embeddings[0], embeddings[1], embeddings[2])
                global_step=0
                logging.info("\nStart training (save checkpoints in %s)\n" % self.config['jobdir'])
                print("\nStart training (save checkpoints in %s)\n" % self.config['jobdir'])
                for _ in range(self.config['numepochs']):
                    sess.run(train_iterator.initializer)
                    while True:
                        try:
                            start_time = time.time()
                            train_input_vals, train_label_vals = sess.run([train_features['input'], train_labels])
                            #logging.info('Training vals retrieved')
                            feed_dict = {m._inputs:train_input_vals,
                                         m._labels:train_label_vals,
                                         m.batch_size:self.config['train_batch_size'],
                                         m.dropout_keep:self.config['dropout']}
                            _, loss_value, eval_ops, predictions, current_lr, curlabels= sess.run([m._train_op, m._total_loss,
                                                                                                   m._eval_op, m._predictions,
                                                                                                   m._learning_rate,
                                                                                                   m._labels_class1],
                                                                                                  feed_dict)
                            #logging.info('loss retrieved')
                            global_step += 1
                            proc_duration = time.time() - start_time
                            assert not np.isnan(loss_value), "Model loss is NaN."
                            f1, auc_tf, tprecision, trecall = self.calc_eval_values(eval_ops)
                            #logging.info('f1 calculated')
                            #self.plot_pr(eval_ops['recalls_op'], eval_ops['precisions_op'], 'Tensorflow Train Step %d Precision-Recall Curve' % global_step)
                            #self.test_skmetrics(curlabels,predictions,global_step)
                            if global_step==1 or global_step % self.config['log_step'] == 0:
                                self.print_log(global_step, self.config, proc_duration, current_lr,loss_value, f1, auc_tf, tprecision, trecall)
                            #logging.info('log done')
                            if global_step==1 or global_step % self.config['summary_step']==0 or global_step==self.max_train_steps:
                                summary_str = sess.run(summary_op)
                                print("\n===== write summary =====")
                                self.write_summaries(train_summary_writer, summary_str, global_step,
                                                     loss_value, auc_tf, f1, tprecision, trecall)
                                if global_step % (1*self.config['summary_step']) == 0 or global_step == self.max_train_steps:
                                    dev_loss, dev_auc, dev_f1, dev_recall, dev_pre, dev_aps, dev_roc_auc = self.dev_step(m, valid_iterator, self.num_valid_batch,
                                                                            valid_features, valid_labels, global_step,sess)
                                    self.write_summaries(dev_summary_writer, summary_str, global_step, dev_loss,
                                                       dev_auc, dev_f1, dev_pre, dev_recall, aps=dev_aps, roc_auc=dev_roc_auc, is_train=False)
                                    if global_step==self.max_train_steps:
                                        self.plot_pr(eval_ops['recalls_op'], eval_ops['precisions_op'],
                                                     'Tensorflow Train Step %d Precision-Recall Curve' % global_step,
                                                     'pr_train.png')
                                print()

                            # stop learning if learning rate is too low
                            if current_lr < 1e-5:
                                print('Learning rate too low %.8f. Stop training.'%current_lr)
                                break

                            if global_step % self.config['checkpoint_step'] == 0:
                                saver.save(sess, save_path, global_step=global_step)
                                saver.save(sess, save_path, global_step=global_step)
                        except tf.errors.OutOfRangeError:
                            break

            saver.save(sess, save_path, global_step=global_step)

    def calc_eval_values(self, eval_ops):
        pre_tf, rec_tf, auc_tf = eval_ops['precisions_op'], eval_ops['recalls_op'], eval_ops['auc_op']
        index = len(pre_tf)//2
        if(pre_tf[index]==0.0 and rec_tf[index]==0.0):
            f1=0.0
        else:
            f1 = (2.0 * pre_tf[index] * rec_tf[index]) / (pre_tf[index] + rec_tf[index])  # threshold = 0.5
        return f1, auc_tf, pre_tf[index], rec_tf[index]


    def print_log(self, global_step, config, proc_duration, current_lr, loss_value, f1, auc, prec, rec):
        examples_per_sec = config['train_batch_size'] / proc_duration
        format_str = '%s: step %d/%d, f1 = %.4f, prec=%.4f, rec=%.4f, auc = %.4f, loss = %.4f ' + \
                         '(%.1f examples/sec; %.3f sec/batch), lr: %.6f'
        print(format_str % (datetime.now(), global_step, self.max_train_steps, f1, prec, rec, auc, loss_value,
                                examples_per_sec, proc_duration, current_lr))


    def write_summaries(self, summary_writer, summary_str, global_step, curloss, curauc, curf1, prec, rec, aps=0.0, roc_auc=0.0, is_train=True):
        summary_writer.add_summary(summary_str, global_step)
        if(is_train==True):
            #curloss = np.mean(curloss)
            #curauc = np.mean(curauc)
            #curf1 = np.mean(curf1)
            print("%s: step %d/%d: train_loss = %.6f, train_auc = %.4f, train_f1 = %.4f, train_prec = %.4f, train_rec =%.4f" \
                  % (datetime.now(), global_step, self.max_train_steps,
                     curloss, curauc, curf1, prec, rec))
        else:
            print("%s: step %d/%d: valid_loss = %.6f, valid_auc = %.4f, valid_f1 = %.4f, valid_prec = %.4f, valid_rec = %.4f, valid_roc_auc = %.4f, valid_aps = %.4f" \
                  % (datetime.now(), global_step, self.max_train_steps,
                     curloss, curauc, curf1, prec, rec, roc_auc, aps))
        summary_writer.add_summary(
            self._summary_for_scalar('loss', curloss), global_step=global_step)
        summary_writer.add_summary(
            self._summary_for_scalar('auc', curauc), global_step=global_step)
        summary_writer.add_summary(
            self._summary_for_scalar('f1', curf1), global_step=global_step)
        summary_writer.add_summary(
            self._summary_for_scalar('precision', prec), global_step=global_step)
        summary_writer.add_summary(
            self._summary_for_scalar('recall', rec), global_step=global_step)

        if is_train==False:
            summary_writer.add_summary(
                self._summary_for_scalar('roc_auc', roc_auc), global_step=global_step)
            summary_writer.add_summary(
                self._summary_for_scalar('average precision score', aps), global_step=global_step)


    def _summary_for_scalar(self, name, value):
        return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=float(value))])


def main():
    model_trainer = Model_Trainer()
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainpath',
                        default='C:/Users/ttasn/Documents/ASU/Research/PhD/deeplearning/dataset_creation/data/model_type_1/tfrecords/sample/conv_bag_filtered/train3_conv_bag.tfrecords',
                        type=str,
                        help='Training files GCS')
    parser.add_argument('--evalpath',
                        default='C:/Users/ttasn/Documents/ASU/Research/PhD/deeplearning/dataset_creation/data/model_type_1/tfrecords/sample/conv_bag_filtered/valid3_conv_bag.tfrecords',
                        type=str,
                        help='Evaluation files GCS')
    parser.add_argument('--job-dir',
                        default='C:/Users/ttasn/Documents/ASU/Research/PhD/deeplearning/phd/cloud_cnnbag/trainer/outputs_f',
                        type=str,
                        help="""\
                          GCS or local dir for checkpoints, exports, and
                          summaries. Use an existing directory to load a
                          trained model, or a new directory to retrain""")
    parser.add_argument('--embdworddir',
                        type=str,
                        default='C:/Users/ttasn/Documents/ASU/Research/PhD/deeplearning/dataset_creation/data/embd_all2/',
                        # 'gs://zodophd/embd',
                        help='Word embedding directory')
    parser.add_argument('--embdposlabeldir',
                        type=str,
                        default='C:/Users/ttasn/Documents/ASU/Research/PhD/deeplearning/dataset_creation/data/embd_pos_label/',
                        # 'gs://zodophd/embd_pos_label',
                        help='Position and label embedding directory')
    parser.add_argument('--embdsize',
                        type=int,
                        default=700,
                        help='Word embedding size')
    parser.add_argument('--numclasses',
                        type=int,
                        default=2,
                        help='Number of classes')
    parser.add_argument('--trainbatchsize',
                        type=int,
                        default=10,
                        help='Batch size for training steps')
    parser.add_argument('--evalbatchsize',
                        type=int,
                        default=10,
                        help='Batch size for evaluation steps')
    parser.add_argument('--numshuffle',
                        type=int,
                        default=1,
                        help='Batch size for evaluation steps')
    parser.add_argument('--numepochs',
                        type=int,
                        default=1,
                        help='Maximum number of epochs on which to train')
    parser.add_argument('--optimizer',
                        choices=[
                            'adam',
                            'sgd',
                            'adagrad',
                            'adadelta',
                        ],
                        default='adam',
                        help='Set optimizer')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate')
    parser.add_argument('--l2reg',
                        type=float,
                        default=0.001,
                        help='L2reg')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.5,
                        help='Dropout')
    parser.add_argument('--numfilters',
                        type=int,
                        default=125,
                        help='Number of filters for each filter size')
    parser.add_argument('--minwin',
                        type=int,
                        default=5,
                        help='Min window for filter')
    parser.add_argument('--maxwin',
                        type=int,
                        default=10,
                        help='Max window for filter')
    parser.add_argument('--summaryfreq',
                        type=int,
                        default=10,
                        help='Write summary and evaluate model per n steps')
    parser.add_argument('--checkpointfreq',
                        type=int,
                        default=1000,
                        help='Save model per n steps')

    parser.add_argument('--verbosity',
                        choices=[
                            'DEBUG',
                            'ERROR',
                            'FATAL',
                            'INFO',
                            'WARN'
                        ],
                        default='INFO',
                        help='Set logging verbosity')
    parse_args, unknown = parser.parse_known_args()
    # Set python level verbosity
    tf.logging.set_verbosity(parse_args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[parse_args.verbosity] / 10)
    del parse_args.verbosity

    if unknown:
        tf.logging.warn('Unknown arguments: {}'.format(unknown))

    model_trainer.set_config(parse_args)
    model_trainer.train()

if __name__ == "__main__":
    main()

