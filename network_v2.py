import sys

import tensorflow as tf
import json
import random

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
learning_rate = 0.5
training_epochs = 5000
batch_size = 2000
num_examples = 7500 #int(sys.argv[1])
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
n_input = 720
n_hidden_1 = 60 # 1st layer number of neurons
n_hidden_2 = 30 # 1st layer number of neurons
n_classes = 3 

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Hidden fully connected layer with 256 neurons
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss asnd optimizer
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()
'''
out_file_1 = open("dataset_corrente.txt", 'a')
out_file_2 = open("dataset_test.txt", 'a') # dataset di test
out_file_3 = open("dataset_n_examples.txt", 'a')
'''


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
                        #print(c)
                        # Compute average loss
                        avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                        

                        #pred = tf.nn.softmax(logits)  # Apply softmax to logits
                        #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
                        # Calculate accuracy
                        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                        #acc = accuracy.eval({X: input_data_x, Y: input_data_y})
                        print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost)) #x, 'accuracy=',acc)
                        


                                        
        print("Optimization Finished!")

        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #print("Accuracy:", accuracy.eval({X: input_data_x[total_batch*batch_size :], Y: input_data_y[total_batch*batch_size :]}))

        acc1 = accuracy.eval({X: input_data_x[: num_examples], Y: input_data_y[: num_examples]})
        acc2 = accuracy.eval({X: input_data_x[7500 :], Y: input_data_y[7500 :]})

        #out_file_1.write(str(acc1)+", ")
        #out_file_2.write(str(acc2)+", ")
        #out_file_3.write(str(num_examples)+', ')
        print("Accuracy dataset corrente:", acc1)
        print("Accuracy dataset validazione:", acc2)


        '''# Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        prediction = tf.argmax(pred, 1)
        # Calculate accuracy
        output= sess.run(prediction,feed_dict={X: input_data_x})

        ris = [0,0,0]

        for x in output: ris[x] += 1

        print(ris)'''




        

