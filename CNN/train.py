#! /usr/bin/env python 
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 30, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 30, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("mode", "train", "mode: train or test");
tf.flags.DEFINE_string("checkpoint_dir", "./", "dir of checkpoint");

# Training parameters
tf.flags.DEFINE_integer("batch_size", 10, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.iteritems()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv,Word2V , label_inv= data_helpers.load_data()
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
# -622
x_train, x_dev = x_shuffled[:-622], x_shuffled[-622:]
y_train, y_dev = y_shuffled[:-622], y_shuffled[-622:]
#x_train = x_shuffled;
#x_dev = x_shuffled;
#y_train = y_shuffled;
#y_dev = y_shuffled;
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=6,
                    vocab_size=len(vocabulary),
                    #embedding_size=FLAGS.embedding_dim,
                    embedding_size = 400,
                    filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    embedding_maxtrix = Word2V)

       # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        #optimizer = tf.train.GradientDescentOptimizer(0.1)
        #optimizer = tf.train.MomentumOptimizer(0.01, 0.9)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.merge_summary(grad_summaries)

        if FLAGS.mode == "train":
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Dev summaries
            train_dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            train_dev_summary_dir = os.path.join(out_dir, "summaries", "train_dev")
            train_dev_summary_writer = tf.train.SummaryWriter(train_dev_summary_dir, sess.graph_def)


            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.all_variables())
        #p = tf.all_variables();
        #for v in p:
            ##print v.var();
    
        # Initialize all variables
        if FLAGS.mode == "train":
            sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch, step1):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step1)

        def dev_step(x_batch, y_batch, writer=None, step1 = 0):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy,output  = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step1)
            return output
        if FLAGS.mode == "train":
            # Generate batches
            batches = data_helpers.batch_iter(
                zip(x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                current_step = tf.train.global_step(sess, global_step)
                train_step(x_batch, y_batch, current_step)
                #current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer, step1 = current_step)
                    output = dev_step(x_train, y_train,writer=train_dev_summary_writer, step1=current_step)
                    print output
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            path = saver.save(sess, checkpoint_dir+"/result.ckpt")
            print("Saved model checkpoint to {}\n".format(path))
                
        else:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir) 
            if ckpt and ckpt.model_checkpoint_path:
                print ckpt.model_checkpoint_path
                saver.restore(sess, FLAGS.checkpoint_dir+"/result.ckpt")
                print ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print tf.train.global_step(sess, global_step)
                #saver.save(sess, FLAGS.checkpoint_dir+"/result_1.ckpt")
            else:
                print "can't find the model in {}".format(FLAGS.checkpoint_dir)

        feed_dict = {
            cnn.input_x: x_dev,
            cnn.input_y: y_dev,
            cnn.dropout_keep_prob: 1.0
        }
        step,  loss, accuracy,output  = sess.run(
            [global_step, cnn.loss, cnn.accuracy, cnn.predictions],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        print output; 

        wb,step, loss, accuracy,output  = sess.run(
            [cnn.Wb,global_step, cnn.loss, cnn.accuracy, cnn.predictions],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        print output; 
        print wb 
        acc = 0;
        for i in range(0, len(output)):
            if output[i] == np.argmax(y_dev[i]):
                acc += 1;
        print acc;
        #evaluation
        count = 0.0
        for i in range(0, len(x_dev)):
            s, p, softmax= sess.run(
            [cnn.scores, cnn.predictions, cnn.softmax],
            {cnn.input_x:[x_dev[i]], cnn.input_y:[y_dev[i]], cnn.dropout_keep_prob : 1.0})
               
            #print p;
            w_list = [ vocabulary_inv[w] for w in x_dev[i]];
            exp = -1;
            for k in range(0, len(y_dev[i])):
                if y_dev[i][k] > 0.5:
                    exp = k;
            if exp == p[0]:
                    count += 1;
       #     print "".join(w_list).encode("utf-8") + "\t" + label_inv[p[0]].encode("utf-8") + "\t" + label_inv[exp].encode("utf-8")
 
       # print "count : {} len : {} acc : {} ".format(count, len(x_dev), count/len(x_dev))
             
