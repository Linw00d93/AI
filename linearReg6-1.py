#!/usr/bin/env python
import tensorflow as tf

# Random input values
N = 40
x = tf.random.normal([N])
m_real = tf.random.truncated_normal([N], mean=2.0)
b_real = tf.random.truncated_normal([N], mean=3.0)
y = m_real * x + b_real
# Variables
m = tf.Variable(tf.random_normal([]))
b = tf.Variable(tf.random_normal([]))
# Compute model and loss
model = tf.add(tf.multiply(x, m), b)
loss = tf.reduce_mean(tf.pow(model - y, 2))
# Create optimizer
learn_rate = 0.1
num_epochs = 200
num_batches = N
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learn_rate).minimize(loss)
# Initialize variables
init = tf.compat.v1.global_variables_initializer()
# Launch session
with tf.compat.v1.Session() as sess:
	sess.run(init)
	for epoch in range(num_epochs):
		for batch in range(num_batches):
			sess.run(optimizer)
# Display results
	print('m = ', sess.run(m))
	print('b = ', sess.run(b))
