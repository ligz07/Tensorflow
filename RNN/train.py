import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import seq2seq
from input_data import *
import math
import time
import datetime
class RNN(object):
    def __init__(self, vocab_size, batch_size, sequece_length, embedding_size, num_classes):
        hidden_num = 20
        l2_loss = tf.constant(0.0)
        ##rnnCell = rnn_cell.BasicRNNCell(hidden_num)
        rnnCell = rnn_cell.BasicLSTMCell(hidden_num, forget_bias=1.0) 
        self.input_data = tf.placeholder(tf.int32, shape=[None, sequece_length], name = "input_data")
        self.weights = tf.placeholder(tf.int32, shape=[None, sequece_length], name= "weights")
        self.output_data = tf.placeholder(tf.int32, [None, sequece_length], name = "output_data")
        a = tf.shape(self.output_data)[0]

        #self.inputs = []
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, embedding_size])
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            #for i, v in enumerate(input_refine):
            #    self.inputs.append(tf.nn.embedding_lookup(embedding, input_refine[i]))
        self.inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, sequece_length, inputs)]
        self.output, self.states = rnn.rnn(rnnCell, self.inputs, dtype=tf.float32)
        predictions = [];
        with tf.name_scope("result"):
            W = tf.Variable(tf.truncated_normal([hidden_num, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            output = tf.reshape(tf.concat(1, self.output), [-1, hidden_num])
            logits = tf.matmul(output, W) + b
            self.scores = logits
            #self.new_scores = [tf.squeeze(k, [1]) for k in tf.split(1, sequece_length, tf.reshape(logits, [-1, sequece_length ,num_classes]))]

        losses = 0;
        accuracy = []
        with tf.name_scope("loss"):
            output_refine = tf.reshape(self.output_data, [-1])
            #output_refine = tf.split(1, sequece_length, self.output_data)
            #weigth = tf.ones_like(output_refine, dtype="float32")
            weight = tf.reshape(tf.cast(self.weights, "float32"), [-1])
            loss = seq2seq.sequence_loss_by_example([self.scores], [output_refine], [weight],num_classes);
            self.loss = tf.reduce_sum(loss)/tf.cast(a, "float32") + 3*l2_loss
            #self.accuracy = tf.reduce_mean(tf.cast(tf.concat(0, accuracy), "float"))

        with tf.name_scope("accurcy"):
            self.predictions = tf.argmax(tf.reshape(self.scores, [-1, sequece_length, num_classes]), 2)
            #self.kk = tf.cast(tf.equal(self.predictions, tf.cast(self.output_data, "int64")), "int64")
            aa = tf.expand_dims(tf.reshape(tf.cast(tf.equal(self.predictions, tf.cast(self.output_data, "int64")), "float32"), [-1]), 0)
            bb = tf.expand_dims(tf.cast(tf.reshape(self.weights, [-1]), "float32"), 0)
            self.kk = tf.squeeze(tf.matmul(aa, bb, transpose_b=True))/tf.reduce_sum(tf.cast(self.weights, "float32"), [0,1])
             
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.cast(self.output_data, "int64")), "float32"), name="accrucy")
            #self.predictions = tf.reshape(self.scores, [sequece_length, -1, num_classes]);



def batch_iter(in_data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(in_data)
    data_size = len(data)
    #num_batches_per_epoch = int(len(data)/batch_size) + 1
    num_batches_per_epoch = int(math.ceil(len(data)/float(batch_size)))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        #shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

x,y,voc, voc_inv, w, p, dx, dy, dp= load_data();
vocab_size = len(voc_inv);
batch_size =  10
embedding_size = 400;
num_classes = 2;
sequece_length = len(x[0]);
num_epoch = 50
#em = tf.Variable(tf.truncated_normal([vocab_size,embedding_size], stddev=0.1), name="embedding")
em = tf.Variable(w, name="embedding")

with tf.Graph().as_default():
    rnnobject = RNN(vocab_size,batch_size, sequece_length, embedding_size, num_classes)
    #optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(1e-4)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    grads_and_vars = optimizer.compute_gradients(rnnobject.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    sess = tf.Session()
    loss_sum = tf.scalar_summary("loss", rnnobject.loss)
    acc = tf.scalar_summary("accuracy", rnnobject.kk)
    merged_summary_op = tf.merge_all_summaries()
    summary_writer_t = tf.train.SummaryWriter('/tmp/train_logs/train_t', sess.graph_def) 
    summary_writer = tf.train.SummaryWriter('/tmp/train_logs/train', sess.graph_def)
    summary_writer_dev = tf.train.SummaryWriter('/tmp/train_logs/dev', sess.graph_def)
    with sess.as_default():
        init = tf.initialize_all_variables()
        sess.run(init)
        batches = batch_iter(zip(x, y, p), batch_size, num_epoch)
        for batch in batches:
            x_batch, y_batch, p_batch = zip(*batch)
            current_step = tf.train.global_step(sess, global_step)
            feed_dict = {rnnobject.input_data:x_batch, rnnobject.output_data:y_batch, rnnobject.weights:p_batch}
            s, _, loss_value,a, b, c = sess.run([merged_summary_op, train_op,
                                            rnnobject.loss,
                                            rnnobject.scores,
                                            rnnobject.predictions,
                                            rnnobject.accuracy,
                                            ],
                                            feed_dict=feed_dict)
            summary_writer_t.add_summary(s, global_step=current_step) 
            if current_step % 300 == 0:
                s, loss_value, c,d,k = sess.run([merged_summary_op, rnnobject.loss, rnnobject.accuracy, rnnobject.predictions, rnnobject.kk], feed_dict = {rnnobject.input_data:x, rnnobject.output_data:y, rnnobject.weights:p})
                time_str = datetime.datetime.now().isoformat()
            
                """
                for i in d:
                    print " ".join("{}".format(i));
                for i in y:
                    print " ".join("{}".format(i));
                """
                print("{}: step:{} loss:{} acc: {}, real acc: {}".format(time_str, current_step, loss_value, c, k))
                """
                total = 0;
                correct = 0;
                for i, m in enumerate(d):
                    for j,n in enumerate(m):
                        if p[i][j] == 0:
                            continue;
                        if d[i][j] == y[i][j]:
                            correct+=1;
                        total += 1;
                print ("correct: {} total:{}".format(correct, total))
                """
                summary_writer.add_summary(s, global_step=current_step)
                s, loss_value, c,d,k = sess.run([merged_summary_op, rnnobject.loss, rnnobject.accuracy, rnnobject.predictions, rnnobject.kk], feed_dict = {rnnobject.input_data:dx, rnnobject.output_data:dy, rnnobject.weights:dp})
                summary_writer_dev.add_summary(s, global_step = current_step);
                print("dev_test: {}: step:{} loss:{} acc: {}, real acc: {}".format(time_str, current_step, loss_value, c, k))
 
            #print loss_value

