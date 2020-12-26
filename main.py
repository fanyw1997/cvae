#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import tensorflow as tf
from six.moves import range
import zhusuan as zs
from utils import *
import numpy as np
import shutil

import dataset
data_dir = './data'
num_classes = 10

@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(x_dim, z_dim, n, one_hot, n_particles=1):
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1., group_ndims=1, n_samples=n_particles)
    one_hot = tf.tile(tf.expand_dims(one_hot, 0), [n_particles, 1, 1])
    z = tf.concat([z, one_hot], axis=-1)
    h = tf.layers.dense(z, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim - num_classes)
    bn.deterministic("x_mean", tf.sigmoid(x_logits))
    bn.bernoulli("x", x_logits, group_ndims=1)
    return bn


@zs.reuse_variables(scope="q_net")
def build_q_net(x, z_dim, n_z_per_x):
    bn = zs.BayesianNet()
    h = tf.layers.dense(tf.cast(x, tf.float32), 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1, n_samples=n_z_per_x)
    return bn


def main():
    # Load MNIST
    data_path = os.path.join(data_dir, "mnist.pkl.gz")
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        dataset.load_mnist_realval(data_path)
    x_dim = x_train.shape[1] + num_classes

    # Define model parameters
    z_dim = 40

    # Build the computation graph
    n_particles = tf.placeholder(tf.int32, shape=[], name="n_particles")
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    one_hot_input = tf.placeholder(tf.float32, shape=[None, num_classes], name="one_hot")
    x = tf.cast(tf.less(tf.random_uniform(tf.shape(x_input)), x_input),
                tf.int32)
    n = tf.placeholder(tf.int32, shape=[], name="n")

    model = build_gen(x_dim, z_dim, n, one_hot_input, n_particles)
    variational = build_q_net(x, z_dim, n_particles)

    lower_bound = zs.variational.elbo(
        model, {"x": tf.slice(x, [0, 0], [-1, 784])}, variational=variational, axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)

    # Random generation
    x_gen = tf.reshape(model.observe()["x_mean"], [-1, 28, 28, 1])

    # Define training/evaluation parameters
    epochs = 1000
    batch_size = 4096
    iters = x_train.shape[0] // batch_size
    save_freq = 20
    result_path = "results/vae"

    shutil.rmtree(result_path)

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                one_hot_batch = t_train[t * batch_size:(t + 1) * batch_size]
                x_batch = np.concatenate([x_batch, one_hot_batch], axis=-1)
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x_input: x_batch,
                                            one_hot_input: one_hot_batch,
                                            n_particles: 1,
                                            n: batch_size})
                lbs.append(lb)
            time_epoch += time.time()
            print("Epoch {} ({:.1f}s): Lower bound = {}".format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % save_freq == 0:
                t_batch = np.array([i % 10 for i in range(100)], dtype=int)
                one_hot_batch = dataset.to_one_hot(t_batch, num_classes)
                images = sess.run(x_gen, feed_dict={n: 100,
                                                    one_hot_input: one_hot_batch,
                                                    n_particles: 1})
                name = os.path.join(result_path,
                                    "vae.epoch.{}.png".format(epoch))
                save_image_collections(images, name)


if __name__ == "__main__":
    main()
