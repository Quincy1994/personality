#coding=utf-8
import tensorflow as tf
import numpy as np

vec = np.array([[1,2]])

#
aux = tf.constant(vec, name="a")
# extend_aux = tf.expand_dims(aux, 1)
aux = tf.reshape(aux, [2])
sess = tf.Session()
org = sess.run(aux)
# result = sess.run(extend_aux)
# extend_aux = tf.expand_dims(aux, 1)
# matrix_aux = tf.tile(aux, [1,2])  # 复制seq_input个向量
# reshape_aux = tf.reshape(matrix_aux, [-1, 2])
print org
# print result
# print sess.run(matrix_aux)
# print sess.run(reshape_aux)


# def load_emo_vector():
#     f = open('data/emo_vector.txt')
#     lines = f.readlines()
#     emo_li = []
#     for line in lines:
#         emo = []
#         line = line.strip()
#         tokens = line.split(" ")
