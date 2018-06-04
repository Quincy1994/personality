# coding=utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

import pickle
import numpy as np

# 返回一个序列中每个元素的长度
def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    # abs 求绝对值,
    # reduce_max 求最大值, reduction_indices 在哪一个维度上求解
    # sign 返回符号-1 if x < 0 ; 0 if x == 0 ; 1 if x > 0
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    # 计算输入tensor元素的和,或者按照reduction_indices指定的轴进行
    return tf.cast(seq_len, tf.int32)
    # 将x的数据格式转化为int32

def biRNNLayer(inputs, hidden_size):

    fw_cell = rnn.GRUCell(hidden_size)
    bw_cell = rnn.GRUCell(hidden_size)
    # fw_cell = rnn.LSTMCell(hidden_size)
    # bw_cell = rnn.LSTMCell(hidden_size)

    ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw = fw_cell,
        cell_bw = bw_cell,
        inputs = inputs,
        sequence_length= length(inputs),
        dtype= tf.float32
    )
    outputs = tf.concat((fw_outputs, bw_outputs), 2)
    return outputs



def attenten_layer(inputs, attention_size):

    sequence_length = inputs.get_shape()[1].value
    hidden_size = inputs.get_shape()[2].value

    # Attention mechanism
    W_omega = tf.get_variable("W_omega", initializer= tf.truncated_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.get_variable("b_omega", initializer= tf.truncated_normal([attention_size], stddev=0.1))
    u_omega = tf.get_variable("u_omega", initializer= tf.truncated_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    return output

# auxiliary attention layer network
def auxAttention(input, aux, attention_size):

    # shape of input: [batch, sequence_length, embedding_size]
    # shape of aux: [batch, vector_length]

    len_aux = int(aux.get_shape()[1])
    seq_input = int(input.get_shape()[1])
    emb_len_input = int(input.get_shape()[2])

    Wm_input = tf.Variable(tf.truncated_normal([emb_len_input, attention_size], stddev=0.1), name="Wm_input")
    Wm_aux = tf.Variable(tf.truncated_normal([len_aux, attention_size], stddev=0.1), name="Wm_aux")
    W_u = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name="W_u")
    W_b = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name="W_b")

    # extend auxiliary vector to matrix
    extend_aux = tf.expand_dims(aux, 1)
    matrix_aux = tf.tile(extend_aux, [1, seq_input, 1])
    reshape_aux = tf.reshape(matrix_aux, [-1, len_aux])

    # attention
    v = tf.matmul(tf.reshape(input, [-1, emb_len_input]), Wm_input) + tf.matmul(reshape_aux, Wm_aux) + tf.reshape(W_b, [1, -1])
    v = tf.tanh(v)
    vu = tf.matmul(v, tf.reshape(W_u, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, seq_input])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
    output = input * tf.reshape(alphas, [-1, seq_input, 1])
    output = tf.reduce_sum(output, 1)
    return output, alphas






class HAN():

    def __init__(self, num_classes, embedding_size, l2_reg_lambda, max_num, max_len):

        self.num_classes = num_classes
        self.embedding_size = embedding_size

        # hidden layer size
        self.w_hidden = 50  # 50
        self.w_atten = 100  # 100
        self.s_hidden = 50 # 50
        self.s_atten = 100 # 100

        # emo_size = 300
        # emo_class = 10

        self.input_w_x = tf.placeholder(tf.int32, [None, max_num, max_len], name="input_x")
        # self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        # self.polar_emotion_vector = tf.placeholder(tf.float32, [2, emo_size], name="polar_emo_vec")
        # self.fine_emotion_vector = tf.placeholder(tf.float32, [8, emo_size], name="fine_emo_vec")
        self.input_y0 = tf.placeholder(tf.float32, [None, num_classes], name="input_y0")
        self.input_y1 = tf.placeholder(tf.float32, [None, num_classes], name="input_y1")
        self.input_y2 = tf.placeholder(tf.float32, [None, num_classes], name="input_y2")
        self.input_y3 = tf.placeholder(tf.float32, [None, num_classes], name="input_y3")
        self.input_y4 = tf.placeholder(tf.float32, [None, num_classes], name="input_y4")

        self.l2_loss = tf.constant(0.0)
        word_emb = pickle.load(open("data/embedding.pickle"))

        # emotion_vector
        # emo_vec = pickle.load(open("data/emo_vector.pickle"))

        with tf.name_scope("embedding"):
            self.embedding_mat = tf.Variable(tf.to_float(word_emb))
            self.w_embedded = tf.nn.embedding_lookup(self.embedding_mat, self.input_w_x)
            self.re_w_embedded = tf.reshape(self.w_embedded, [-1, max_len, embedding_size])
            # polarity_emo = [emo_vec[5], emo_vec[6]]
            # polarity_emo = np.array(polarity_emo)
            # self.polar_emotion_vector = tf.to_float(polarity_emo)

        with tf.variable_scope("sent2vec_main"):
            # self.re_w_embedded = tf.reshape(self.w_embedded, [-1, max_len, embedding_size])
            self.w_birnn = biRNNLayer(self.re_w_embedded, self.w_hidden)
            self.sentences = attenten_layer(self.w_birnn, self.w_atten)
            # self.sentences = self.polarity_emo_attention(self.w_birnn, self.batch_size * max_num, self.polar_emotion_vector, self.w_atten)


        with tf.variable_scope("doc2vec_main"):
            self.re_sen = tf.reshape(self.sentences, [-1, max_num, self.w_hidden * 2])
            self.s_birnn = biRNNLayer(self.re_sen, self.s_hidden)
            self.doc = attenten_layer(self.s_birnn, self.s_atten)
            # self.doc = self.polarity_emo_attention(self.s_birnn, self.batch_size, self.polar_emotion_vector, self.s_atten)

        with tf.variable_scope("aux_sen_1"):
            self.w_birnn_1 = biRNNLayer(self.re_w_embedded, self.w_hidden)
            self.sentences_1 = attenten_layer(self.w_birnn_1, self.w_atten)
        with tf.variable_scope("aux_doc_1"):
            self.re_sen_1 = tf.reshape(self.sentences_1, [-1, max_num, self.w_hidden * 2])
            self.s_birnn_1 = biRNNLayer(self.re_sen_1, self.s_hidden)
            self.doc_1 = attenten_layer(self.s_birnn_1, self.s_atten)

        with tf.variable_scope("aux_2"):
            # self.re_w_embedded_2 = tf.reshape(self.w_embedded, [-1, max_len, embedding_size])
            self.w_birnn_2 = biRNNLayer(self.re_w_embedded, self.w_hidden)
            self.sentences_2 = attenten_layer(self.w_birnn_2, self.w_atten)
        with tf.variable_scope("aux_doc_2"):
            self.re_sen_2 = tf.reshape(self.sentences_2, [-1, max_num, self.w_hidden * 2])
            self.s_birnn_2 = biRNNLayer(self.re_sen_2, self.s_hidden)
            self.doc_2 = attenten_layer(self.s_birnn_2, self.s_atten)

        with tf.variable_scope("aux_3"):
            # self.re_w_embedded_3 = tf.reshape(self.w_embedded, [-1, max_len, embedding_size])
            self.w_birnn_3 = biRNNLayer(self.re_w_embedded, self.w_hidden)
            self.sentences_3 = attenten_layer(self.w_birnn_3, self.w_atten)
        with tf.variable_scope("aux_doc_3"):
            self.re_sen_3 = tf.reshape(self.sentences_3, [-1, max_num, self.w_hidden * 2])
            self.s_birnn_3 = biRNNLayer(self.re_sen_3, self.s_hidden)
            self.doc_3 = attenten_layer(self.s_birnn_3, self.s_atten)

        with tf.variable_scope("aux_4"):
            # self.re_w_embedded_4 = tf.reshape(self.w_embedded, [-1, max_len, embedding_size])
            self.w_birnn_4 = biRNNLayer(self.re_w_embedded, self.w_hidden)
            self.sentences_4 = attenten_layer(self.w_birnn_4, self.w_atten)
        with tf.variable_scope("aux_doc_4"):
            self.re_sen_4 = tf.reshape(self.sentences_4, [-1, max_num, self.w_hidden * 2])
            self.s_birnn_4 = biRNNLayer(self.re_sen_4, self.s_hidden)
            self.doc_4 = attenten_layer(self.s_birnn_4, self.s_atten)

        # with tf.variable_scope("aux_atten"):
        #     concat_aux = tf.concat((self.doc_1, self.doc_2, self.doc_3, self.doc_4), 1)
        #     concat_h = tf.reshape(concat_aux, [-1, 4, self.s_hidden * 2])
        #     concat_h = biRNNLayer(concat_h, self.s_hidden)
        #     aux_vec, alphas = auxAttention(concat_h, self.doc, self.s_atten)

        with tf.variable_scope("concat"):
            self.mix_doc = tf.concat((self.doc, self.doc_4), 1)


        with tf.name_scope("dropout"):
            # h_drop = tf.nn.dropout(self.doc, self.dropout_keep_prob)
            h_drop = tf.nn.dropout(self.mix_doc, self.dropout_keep_prob)
            # self.doc_1 = tf.nn.dropout(self.doc_1, self.dropout_keep_prob)
            # self.doc_2 = tf.nn.dropout(self.doc_2, self.dropout_keep_prob)
            # self.doc_3 = tf.nn.dropout(self.doc_3, self.dropout_keep_prob)
            self.doc_4 = tf.nn.dropout(self.doc_4, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "output_W",
                shape=[h_drop.get_shape()[1].value, num_classes],
                initializer = tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0., shape=[num_classes], name="output_b"))
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.out = tf.nn.softmax(tf.matmul(h_drop, W) + b)
            self.y_pred = tf.argmax(self.out, axis=1, name="predictions")

        with tf.name_scope("output_a1"):
            W1 = tf.get_variable(
                "output_W1",
                shape=[self.doc_1.get_shape()[1].value, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b1 = tf.Variable(tf.constant(0., shape=[num_classes], name="output_b1"))
            self.l2_loss += tf.nn.l2_loss(W1)
            self.l2_loss += tf.nn.l2_loss(b1)
            self.out_1 = tf.nn.softmax(tf.matmul(self.doc_1, W1) + b1)
            self.y_pred_1 = tf.argmax(self.out_1, axis=1, name="predictions_1")

        with tf.name_scope("output_a2"):
            W2 = tf.get_variable(
                "output_W2",
                shape=[self.doc_2.get_shape()[1].value, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b2 = tf.Variable(tf.constant(0., shape=[num_classes], name="output_b2"))
            self.l2_loss += tf.nn.l2_loss(W2)
            self.l2_loss += tf.nn.l2_loss(b2)
            self.out_2 = tf.nn.softmax(tf.matmul(self.doc_2, W2) + b2)
            self.y_pred_2 = tf.argmax(self.out_2, axis=1, name="predictions_2")

        with tf.name_scope("output_a3"):
            W3 = tf.get_variable(
                "output_W3",
                shape=[self.doc_3.get_shape()[1].value, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b3 = tf.Variable(tf.constant(0., shape=[num_classes], name="output_b3"))
            self.l2_loss += tf.nn.l2_loss(W3)
            self.l2_loss += tf.nn.l2_loss(b3)
            self.out_3 = tf.nn.softmax(tf.matmul(self.doc_3, W3) + b3)
            self.y_pred_3 = tf.argmax(self.out_3, axis=1, name="predictions_3")

        with tf.name_scope("output_a4"):
            W4 = tf.get_variable(
                "output_W4",
                shape=[self.doc_4.get_shape()[1].value, num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b4 = tf.Variable(tf.constant(0., shape=[num_classes], name="output_b4"))
            self.l2_loss += tf.nn.l2_loss(W4)
            self.l2_loss += tf.nn.l2_loss(b4)
            self.out_4 = tf.nn.softmax(tf.matmul(self.doc_4, W4) + b4)
            self.y_pred_4 = tf.argmax(self.out_4, axis=1, name="predictions_4")


        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.input_y0)
            # losses_aux_1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.out_1, labels=self.input_y1)
            # losses_aux_2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.out_2, labels=self.input_y2)
            # losses_aux_3 = tf.nn.softmax_cross_entropy_with_logits(logits=self.out_3, labels=self.input_y3)
            losses_aux_4 = tf.nn.softmax_cross_entropy_with_logits(logits=self.out_4, labels=self.input_y4)

            # losses_aux = alphas[0][0] * tf.reduce_mean(losses_aux_1) + alphas[1][0] * tf.reduce_mean(losses_aux_2) + alphas[2][0] * tf.reduce_mean(losses_aux_3) +alphas[3][0] * tf.reduce_mean(losses_aux_4)
            # losses_aux = tf.reduce_mean(losses_aux_1) + tf.reduce_mean(losses_aux_2) +  tf.reduce_mean(losses_aux_3) + tf.reduce_mean(losses_aux_4)
            losses_aux = tf.reduce_mean(losses_aux_4)
            a = 0.7
            self.loss = a * tf.reduce_mean(losses) + (1-a) * losses_aux + l2_reg_lambda * self.l2_loss

        with tf.name_scope("accuracy"):
            label = tf.argmax(self.input_y0, axis=1, name="label")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_pred, label), tf.float32))



    # auxiliary attention layer network
    def polarity_emo_attention(self, inputs, batch, polar_emo, attention_size):

        # shape of input: [batch, sequence_length, embedding_size]
        # shape of aux: [batch, vector_length]

        # emotion length
        len_emo = 300
        positive_emo, negative_emo = polar_emo[0], polar_emo[1]
        # size and length
        print inputs.shape
        seq_input = int(inputs.get_shape()[1])
        emb_len_input = int(inputs.get_shape()[2])
        print emb_len_input

        # paramater
        Wm_input = tf.Variable(tf.truncated_normal([emb_len_input, attention_size], stddev=0.1), name="Wm_input")
        Wm_pos = tf.Variable(tf.truncated_normal([len_emo, attention_size], stddev=0.1), name="Wm_pos")
        Wm_neg = tf.Variable(tf.truncated_normal([len_emo, attention_size], stddev=0.1), name="Wm_neg")
        W_u = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name="W_u")
        W_b = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name="W_b")

        # extend auxiliary vector to matrix
        cp_num = batch * seq_input
        # print "cp", cp_num
        # cp_num = batch * seq_input
        pos_reshape = tf.reshape(positive_emo, [-1, len_emo])
        pos_matrix = tf.tile(pos_reshape, [1, cp_num])
        pos = tf.reshape(pos_matrix, [-1, len_emo])
        # print pos.shape

        neg_reshape = tf.reshape(negative_emo, [-1, len_emo])
        neg_matrix = tf.tile(neg_reshape, [1, cp_num])
        neg = tf.reshape(neg_matrix, [-1, len_emo])

        # print neg.shape
        # attention
        v = tf.matmul(tf.reshape(inputs, [-1, emb_len_input]), Wm_input) + tf.matmul(pos, Wm_pos) + tf.matmul(neg, Wm_neg) + tf.reshape(W_b, [1, -1])
        v = tf.tanh(v)
        vu = tf.matmul(v, tf.reshape(W_u, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, seq_input])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        output = inputs * tf.reshape(alphas, [-1, seq_input, 1])
        output = tf.reduce_sum(output, 1)
        return output





































