import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

vocalsize = 50
rnncell = rnn_cell.BasicRNNCell(vocalsize)

input_data = tf.placeholder(tf.int32, shape=[NONE, NONE], name = "input_data")

output_data = tf.placeholder(tf.int32, [NONE, 2], name = "output_data")

with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, input_data)

rnn = rnn.rnn(rnncell, inputs)
optimizer = tf.train.GradientDescentOptimizer(0.1)


bal_step = tf.Variable(0, name='global_step', trainable=False)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                        output_data,
                                                        name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
train_op = optimizer.minimize(loss, global_step=global_step)


with tf.Graph().as_default(), tf.Session() as sess:
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for step in xrange(FLAGS.max_steps):
        feed_dict = fill_feed_dict(data_sets.train,
                               images_placeholder,
                               labels_placeholder)
        _, loss_value = sess.run([train_op, loss],
                             feed_dict=feed_dict)
