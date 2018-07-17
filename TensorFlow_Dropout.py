import tensorflow as tf

"""
#Example that i wrote to explain the dropout function. 
n_input =10
n_hidden = 5
n_classes = 2
features = tf.placeholder(tf.float32,[None,n_input])
weights = [tf.Variable(tf.truncated_normal([n_input,n_hidden])),tf.Variable(tf.truncated_normal([n_hidden,n_classes]))]
bias = [tf.Variable(tf.truncated_normal([n_hidden])),tf.Variable(tf.truncated_normal([n_classes]))]
keep_probab = tf.placeholder(tf.float32)
hidden_layer = tf.add(tf.matmul(features,weights[0]),bias[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer,keep_probab)

logits = tf.add(tf.matmul(hidden_layer,weights[1]),bias[1])
"""
print("""During training, a good starting value for keep_prob is 0.5.
During testing, use a keep_prob value of 1.0 to keep all units and maximize the power of the model.
""")

hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

weights = [tf.Variable(hidden_layer_weights), tf.Variable(out_weights)]
bias =[tf.Variable(tf.zeros(3)),tf.Variable(tf.zeros(2))]
keep_probab = tf.placeholder(tf.float32)
features = tf.Variable([[0.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model with Dropout
hidden_layer = tf.add(tf.matmul(features,weights[0]),bias[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer,keep_probab)
logits = tf.add(tf.matmul(hidden_layer,weights[1]),bias[1])

# TODO: Print logits from a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(logits,feed_dict={keep_probab:0.5})

print(output)

