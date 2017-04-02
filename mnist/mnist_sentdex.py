import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
from tensorflow.python.client import device_lib

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#number of nodes for each hidden layer
num_nodes_hl_1 = 2000
num_nodes_hl_2 = 2000
num_nodes_hl_3 = 2000

num_output_classes = 10
#batch size to use when loading mnist data (number of images)
batch_size = 1000

x = tf.placeholder("float", [None, 784]) #28x28 px images flattened
y = tf.placeholder("float")

def neural_network_model(data):

    #define variables for layers (that is: allocate memory, create a structure)
    hidden_layer_1 = {"weights": tf.Variable(tf.random_normal([784, num_nodes_hl_1])),
                      "biases": tf.Variable(tf.random_normal([num_nodes_hl_1]))}
    # weights initialized as random value. Shape corresponds to number of inputs x number of nodes.
    # biases initialized as random value. Shape corresponds to number of nodes in hidden layer

    hidden_layer_2 = {"weights": tf.Variable(tf.random_normal([num_nodes_hl_1, num_nodes_hl_2])),
                      "biases": tf.Variable(tf.random_normal([num_nodes_hl_2]))}

    hidden_layer_3 = {"weights": tf.Variable(tf.random_normal([num_nodes_hl_2, num_nodes_hl_3])),
                      "biases": tf.Variable(tf.random_normal([num_nodes_hl_3]))}

    output_layer = {"weights": tf.Variable(tf.random_normal([num_nodes_hl_3, num_output_classes])),
                  "biases": tf.Variable(tf.random_normal([num_output_classes]))}

    # implement neural network:
    # for each neuron: output = (input * weight) + bias
    # further: use an activation function

    layer1 = tf.add(tf.matmul(data, hidden_layer_1["weights"]), hidden_layer_1["biases"])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, hidden_layer_2["weights"]), hidden_layer_2["biases"])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2, hidden_layer_3["weights"]), hidden_layer_3["biases"])
    layer3 = tf.nn.relu(layer3)

    output = tf.add(tf.matmul(layer3, output_layer["weights"]), output_layer["biases"])

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    #define cost function
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )

    #define optimizer (minimize cost)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_epochs = 100

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) #load data from mnist dataset
                #x = image, y = class
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch", epoch, " of ", num_epochs, " loss: ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)