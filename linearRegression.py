import tensorflow as tf
import numpy as np
import pandas as pd

# Want to make this a module but it doesnt work for some reason
import data

import matplotlib.pyplot as plt


def parser(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')


series = pd.read_csv('appleStock.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser, usecols=(0,3,4,5))
y = pd.read_csv('appleStock.csv', header=0, usecols=(1,2), index_col=None).drop(columns=['volume'])


series = series.values.tolist()

# Add dummy column for bias term
for i in series:
    i.insert(0,1)

series = np.asarray(series)

# Split into train/test data set
n = int(len(series)*0.25)
trainX, testX = series[:-n], series[-n:]
trainY, testY = y[:-n].values, y[-n:].values

# Constants
cost_history = np.empty(shape=[1],dtype=float)
num_samples = trainX.shape[0]
num_features = trainX.shape[1]

# Optimization parameters
learning_rate = 0.01
training_epochs = 500

# Tensorflow objects
# initialize theta to all 0/1
theta = tf.Variable(tf.ones([num_features, 1]))
#theta = tf.Variable(tf.zeros([num_features, 1]))
X = tf.placeholder("float", [None, num_features])
Y = tf.placeholder("float", [None, 1])

# Tensorflow ops
hyp = tf.matmul(X, theta)
cost = tf.reduce_sum(tf.pow(hyp- Y, 2))/(2*num_samples)
# Cost could be mse
# cost = tf.reduce_mean(tf.square(hyp - Y))

# Adam or gradient descent?
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# Initialize all vars to be used in session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(training_epochs):
        # faster all together
        sess.run(optimizer, feed_dict={X: trainX, Y: trainY})
        cost_history = np.append(cost_history, sess.run(cost, feed_dict={X: trainX, Y: trainY}))

        # slower optimizing one sample at a time
        # for (x,y) in zip(trainX, trainY):
        #     x = x.reshape(1,num_features)
        #     y = y.reshape(1,1)
        #     sess.run(optimizer, feed_dict={X:x, Y:y})
        #training_cost = sess.run(cost, feed_dict={X:trainX,Y: trainY})
        #print("Training cost=", training_cost, "W=", sess.run(theta), '\n')

    plt.plot(range(len(cost_history)),cost_history)
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    plt.show()

    hypothesis = sess.run(hyp, feed_dict={X: testX})
    cost = tf.reduce_sum(tf.pow(hyp - Y, 2)) / (2 * num_samples)
    mse = tf.reduce_mean(tf.square(hypothesis - testY))

    fig, ax = plt.subplots()
    ax.scatter(testY, hypothesis, c='r')
    ax.plot([testY.min(), testY.max()], [hypothesis.min(), hypothesis.max()], 'k--', lw=2)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Predicted')
    plt.show()