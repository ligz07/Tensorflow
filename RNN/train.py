import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn

class RNN(object):
    def __init__(vocab_size, batch_size, sequece_length, embedding_size, num_classes):

    rnnCell = rnn_cell.BasicRNNCell(embedding_size)
    input_data = tf.placeholder(tf.int32, shape=[batch_size, sequece_length], name = "input_data")

    output_data = tf.placeholder(tf.int32, [batch_size, sequece_length, num_classes], name = "output_data")

    input_refine = [tf.squeeze(input_, [1]) for input_ in tf.split(1, sequece_length, inputs)]
    with tf.device("/cpu:0"):
          embedding = tf.get_variable("embedding", [vocab_size, input_refine])
          inputs = tf.nn.embedding_lookup(embedding, input_data)
    state = rnncell.zero_state(batch_size)
    self.output, self.states = rnn.rnn(rnnCell, inputs, initial_state = state)

    with tf.name_scope("result"):
        W = tf.Variable(tf.truncated_normal([embedding_size, num_classes], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        self.scores = [];
        self.predictions = [];
        for i in self.output:
            scores =   tf.nn.xw_plus_b(i, W, b, name="scores")
            self.scores.Append(scores);
            prediction = tf.argmax(scores, 1);
            self.predictions.Append(prediction)

    with tf.name_scope("loss"):
        for i in self.scores:
            losses += tf.nn.softmax_cross_entropy_with_logits(i, self.input_y)
        self.loss = tf.reduce_mean(losses)


vocab_size = 0;
batch_size = 10
embedding_size = 400;
num_classes = 2;
sequece_length = 3;
with tf.Graph().as_default():
    rnnobject = RNN(vocab_size,batch_size, sequece_length, embedding_size, num_classes)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    grads_and_vars = optimizer.compute_gradients(rnnobject.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    sess = tf.Session()
    with sess.as_default():
        init = tf.initialize_all_variables()
        sess.run(init)

        for step in xrange(FLAGS.max_steps):
            feed_dict = fill_feed_dict(data_sets.train,
                                   images_placeholder,
                                   labels_placeholder)
            _, loss_value = sess.run([train_op, loss],
                                 feed_dict=feed_dict)
