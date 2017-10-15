# -*- coding: UTF-8 -*-
import tensorflow as tf
from datasets import train, test, result
import numpy as np
import pandas as pd
class Config:
    """
    best Config={max_step: 4000,
                 batch_size: 64}
    """
    max_step = 10000
    batch_size = 64
    learning_rate = 0.0001
    features_size = len(train.df.columns)

def save_result(data, columns):
    """
    :param data: np.array 
    :param columns: list
    """
    index = []
    for i in range(len(data)):
        index.append(i+1461)
    data = np.column_stack((index, data))
    df = pd.DataFrame(data, columns=columns)
    df['Id'] = df['Id'].astype('int')
    df.to_csv("result.csv", index=False)
    print()

# NN
x_placeholder = tf.placeholder(tf.float32, shape=[None, Config.features_size])
y_placeholder = tf.placeholder(tf.float32, shape=[None, ])
train_phase = tf.placeholder(tf.bool)
hidden = x_placeholder
hidden = tf.layers.dense(inputs=hidden, units=128, activation=tf.nn.relu)
hidden = tf.nn.dropout(hidden, keep_prob=0.8)
hidden = tf.layers.dense(inputs=hidden, units=64, activation=tf.nn.relu)
hidden = tf.nn.dropout(hidden, keep_prob=0.8)
hidden = tf.layers.dense(inputs=hidden, units=32, activation=tf.nn.relu)
hidden = tf.nn.dropout(hidden, keep_prob=0.8)
logits = tf.layers.dense(inputs=hidden, units=1)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits-y_placeholder)))
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(Config.max_step+1):
        features, labels = train.next_batch(batch_size=Config.batch_size)
        sess.run(train_op, feed_dict={x_placeholder: features,
                                      y_placeholder: labels,
                                      train_phase: True})
        if step % 100 == 0:
            print("Step:%d " % step)
            print("Test_set: ")
            _pre = logits.eval(feed_dict={x_placeholder: test.df, train_phase: False})
            _pre = np.reshape(_pre, newshape=(-1,))
            N = len(_pre)
            print(np.c_[_pre, result['SalePrice']])
    _pre = sess.run(logits, feed_dict={x_placeholder: test.df,
                                       train_phase: False})
    _pre = np.reshape(_pre, newshape=(-1,))
    save_result(_pre, ['Id', 'SalePrice'])