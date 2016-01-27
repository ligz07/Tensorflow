import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import seq2seq
from input_data import *
class RNN(object):
    def __init__(self, vocab_size, batch_size, sequece_length, embedding_size, num_classes):
        hidden_num = 30
        rnnCell = rnn_cell.BasicRNNCell(hidden_num)
        self.input_data = tf.placeholder(tf.int32, shape=[None, sequece_length], name = "input_data")

        self.output_data = tf.placeholder(tf.float32, [None, sequece_length], name = "output_data")

        input_refine = [tf.squeeze(input_, [1]) for input_ in tf.split(1, sequece_length, self.input_data)]
        self.inputs = []
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, embedding_size])
            
            for i, v in enumerate(input_refine):
                self.inputs.append(tf.nn.embedding_lookup(embedding, input_refine[i]))
                
            #inputs = tf.nn.embedding_lookup(embedding, input_refine[0])
        #self.inputs = np.array(self.inputs) 
#        state = rnnCell.zero_state(batch_size, tf.float32)
        self.output, self.states = rnn.rnn(rnnCell, self.inputs, dtype=tf.float32)
        scores = [];
        predictions = [];
        with tf.name_scope("result"):
            W = tf.Variable(tf.truncated_normal([hidden_num, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            for i in self.output:
                score = tf.nn.xw_plus_b(i, W, b, name="score")
                scores.append(score);
                prediction = tf.argmax(score, 1);
                predictions.append(prediction)

        self.scores = tf.reshape(tf.concat(1, scores), [-1, sequece_length, num_classes])
        self.sco = tf.concat(0, scores);
        self.predictions = tf.concat(0, predictions);
        losses = 0;
        accuracy = []
        with tf.name_scope("loss"):
            output_refine = [tf.squeeze(k, [1]) for k in tf.split(1, sequece_length, self.output_data)]
            for i, v in enumerate(scores):
                losses += tf.nn.softmax_cross_entropy_with_logits(scores[i], output_refine[i])
                accuracy.append(tf.equal(predictions[i], tf.argmax(output_refine[i],1)))
            weigth = [1.0]*sequece_length
            loss = seq2seq.sequence_loss_by_example(scores, self.output_data, num_classes);
            #self.loss = tf.reduce_mean(losses)
            self.loss = loss;
            self.accuracy = tf.reduce_mean(tf.cast(tf.concat(0, accuracy), "float"))


def batch_iter(in_data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = in_data
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        #shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

x,y,voc, voc_inv, w = load_data();
vocab_size = len(voc_inv);
batch_size = 3
embedding_size = 400;
num_classes = 2;
sequece_length = len(x[0]);
em = tf.Variable(w, name="embedding")
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
        i = 0;
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch = np.array(x_batch);
            y_batch = np.array(y_batch);
            feed_dict = {rnnobject.input_data:x_batch, rnnobject.output_data:y_batch}
            _, loss_value,a,b,c = sess.run([train_op, 
                                            rnnobject.loss, 
                                            rnnobject.scores,
                                            rnnobject.sco,
                                            rnnobject.predictions ],
                                            feed_dict=feed_dict)
            print loss_value
            print a
            print b
            print c
