# -*- coding: UTF-8 -*-
import tensorflow as tf
from datasets import train, test, result, validation
import numpy as np
import pandas as pd
class Config:
    """
    best Config={max_step: 4000,
                 batch_size: 64}
    """
    max_step = 2300
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
        index.append(i+892)
    data = np.column_stack((index, data))
    df = pd.DataFrame(data, columns=columns)
    df = df.astype('int')
    df.to_csv("result.csv", index=False)
    print()

def do_eval(model, features, labels):
    _acc, _loss = model.run([accuracy, loss], feed_dict={x_placeholder: features,
                                                         y_placeholder: labels,
                                                         train_phase: False})
    print("Accuracy: %.5lf, Loss: %.5lf" %(_acc, _loss))

# NN
x_placeholder = tf.placeholder(tf.float32, shape=[None, Config.features_size])
y_placeholder = tf.placeholder(tf.int32, shape=[None, ])
train_phase = tf.placeholder(tf.bool)
hidden = x_placeholder
hidden = tf.layers.dense(inputs=hidden, units=32, activation=tf.nn.relu)
hidden = tf.nn.dropout(hidden, keep_prob=0.8)
hidden = tf.layers.dense(inputs=hidden, units=32, activation=tf.nn.relu)
hidden = tf.nn.dropout(hidden, keep_prob=0.8)
logits = tf.layers.dense(inputs=hidden, units=2)
y_one_hot = tf.one_hot(y_placeholder, depth=2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits))
correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
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
            print("Train_set: ")
            do_eval(sess, train.df, train.labels)
            # print("Validation_set: ")
            # do_eval(sess, validation.df, validation.labels)
            print("Test_set: ")
            do_eval(sess, test.df, result)
    _pre = sess.run(logits, feed_dict={x_placeholder: test.df,
                                       train_phase: False})
    _pre = np.argmax(_pre, 1)
    save_result(_pre, ['PassengerId', 'Survived'])
