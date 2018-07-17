import tensorflow as tf
#Since Order of weight and bias is reversed the above will throw an error if name is not explicitly set.

tf.reset_default_graph()
save_file = './naming.ckpt'
weights  = tf.Variable(tf.truncated_normal(shape=[2,3]),name='weights_0')
bias = tf.Variable(tf.truncated_normal([3]),name='bias_0')
saver = tf.train.Saver()

print('Save Weights : {}'.format(weights.name))
print('SAve the Bias :{}'.format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess,save_file)

#Remove previous weight and bias
tf.reset_default_graph()
bias = tf.Variable(tf.truncated_normal([3]),name='bias_0')
weights = tf.Variable(tf.truncated_normal([2,3]),name='weights_0')
saver = tf.train.Saver()


print('Load Weights: {}'.format(weights.name))
print('Load Bias: {}'.format(bias.name))

with tf.Session() as sess:
    saver.restore(sess,save_file)


print('Loaded Weights and Bias successfully.')
