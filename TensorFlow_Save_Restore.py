import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data
print("""Training a model can take hours. But once you close your TensorFlow session, you lose all the trained weights and biases. If you were to reuse the model in the future, you would have to train it all over again!
Fortunately, TensorFlow gives you the ability to save your progress using a class called tf.train.Saver. This class provides the functionality to save any tf.Variable to your file system.
""")

'''
save_file = './model.ckpt'
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))
saver = tf.train.Saver()

with tf.Session() as sess:
    # Load the weights and bias
    saver.restore(sess, save_file)

    # Show the values of weights and bias
    print('Weight:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))

print("Since tf.train.Saver.restore() sets all the TensorFlow Variables, "
      "you don't need to call tf.global_variables_initializer().")

# Remove previous Tensors and Operations
tf.reset_default_graph()

'''

print("Since tf.train.Saver.restore() sets all the TensorFlow Variables, "
      "you don't need to call tf.global_variables_initializer().")

learning_rate = 0.001
n_input = 784
n_classes = 10

#import mnist data
mnist = input_data.read_data_sets('.',one_hot=True)
#Features and Labels
features = tf.placeholder(tf.float32,[None,n_input])
labels = tf.placeholder(tf.float32,[None,n_classes])

#weights & Bias
weights = tf.Variable(tf.random_normal(shape=[n_input,n_classes]))
bias = tf.Variable(tf.random_normal(shape=[n_classes]))

#Logits = xW+B
logits = tf.add(tf.matmul(features,weights),bias)

#define Loss and optimizer

cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

#calculate Accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

save_file = './train_model.ckpt'
batch_size =128
n_epochs = 100
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Training Cycle
    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples/batch_size)
        #loop over all batches
        for i in range(total_batch):
            batch_features,batch_labels = mnist.train.next_batch(batch_size)
            sess.run(optimizer,feed_dict={features:batch_features,labels:batch_labels})

        if(epoch % 10 ==0):
            valid_accuracy = sess.run(
                accuracy,
                feed_dict={
                    features: mnist.validation.images,
                    labels: mnist.validation.labels})
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))
    saver.save(sess, save_file)
    print('Trained Model Saved.')

#now to Restore the Trained Set
#saver = tf.train.Saver()
#save_file = './model.ckpt' or path to the saved model

with tf.Session() as sess:
    #The below ill restore the file / model
    saver.restore(sess,save_file)
    test_accuracy = sess.run(accuracy,feed_dict={features : mnist.test.images,labels:mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
