import sys

import tensorflow as tf
import json

print("Loading data...")
f = open('out_x.txt', 'r')
input_data_x = json.load(f)
print("1/2 loaded...")
f = open('out_y.txt', 'r')
input_data_y = json.load(f)
print("Done loading data.")
#print(len(input_data_x))
#sys.exit()

# Parameters
learning_rate = 0.005
training_epochs = 200
batch_size = 50
num_examples = 7500
display_step = 1

print("------------------------------------")
print("learning_rate:",learning_rate)
print("training_epochs:",training_epochs)
print("batch_size:",batch_size)
print("num_examples:",num_examples)
print("data_x:",len(input_data_x))
print("data_y:",len(input_data_y))
print("------------------------------------")

# Network Parameters
n_input = 1* 720
n_hidden_1 = 200 # 1st layer number of neurons
n_classes = 3 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):

            batch_x = input_data_x[i*batch_size : (i+1)*batch_size]
            batch_y = input_data_y[i*batch_size : (i+1)*batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))    
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost), "Accuracy:", accuracy.eval({X: input_data_x[total_batch*batch_size :], Y: input_data_y[total_batch*batch_size :]}))
            
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: input_data_x[total_batch*batch_size :], Y: input_data_y[total_batch*batch_size :]}))


