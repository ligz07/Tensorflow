import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from input_data import *
class RNN(object):
    def __init__(self, vocab_size, batch_size, sequece_length, embedding_size, num_classes):

        rnnCell = rnn_cell.BasicRNNCell(embedding_size)
        self.input_data = tf.placeholder(tf.int32, shape=[batch_size, sequece_length], name = "input_data")

        self.output_data = tf.placeholder(tf.int32, [batch_size, sequece_length, num_classes], name = "output_data")

        input_refine = [tf.squeeze(input_, [1]) for input_ in tf.split(1, sequece_length, self.input_data)]
        with tf.device("/cpu:0"):
              embedding = tf.get_variable("embedding", [vocab_size, embedding_size])
              inputs = tf.nn.embedding_lookup(embedding, input_refine)
        state = rnncell.zero_state(batch_size)
        self.output, self.states = rnn.rnn(rnnCell, inputs, initial_state = state)

        with tf.name_scope("result"):
            W = tf.Variable(tf.truncated_normal([embedding_size, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = [];
            self.predictions = [];
            for i in self.output:
                scores =   tf.nn.xw_plus_b(i, W, b, name="scores")
                self.scores.append(scores);
                prediction = tf.argmax(scores, 1);
                self.predictions.append(prediction)

        with tf.name_scope("loss"):
            for i in self.scores:
                losses += tf.nn.softmax_cross_entropy_with_logits(i, self.output_data)
            self.loss = tf.reduce_mean(losses)

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

x,y,voc, voc_inv, w = load_data();
vocab_size = len(voc_inv);
batch_size = 10
embedding_size = 400;
num_classes = 2;
sequece_length = len(x[0]);
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

        batches = batch_iter(zip(x, y), batch_size, 1)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            feed_dict = {rnnobject.input_data:x_batch, rnnobject.output_data:y_batch}
            _, loss_value = sess.run([train_op, rnn.loss],
                                 feed_dict=feed_dict)
