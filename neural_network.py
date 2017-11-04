__author__ = 'Florin Bora'

import os
import numpy as np
import tensorflow as tf

def augment_data_set(data):
    augmented_data = data.copy()
    # rotations
    for game in reversed(augmented_data):
        for k in range(1, 4):
            augmented_data += [[[np.rot90(state) for state in game[0]], game[1]]]
    # inversions
    for game in reversed(augmented_data):
        augmented_data += [[[np.transpose(state) for state in game[0]], game[1]]]
    # ignore color reflections
    return augmented_data


def update_nn_training_set(new_games, training_set):
    '''
    Append the new games to the trainig set.
    The training set is a list [x, y] where
        x is the unravel state of the board of length 9
        y is the concatenation of
            one hot encoding of the winner and
            the next move out of 9 choices on the board
    '''
    for game in new_games:
        winner = game[1]
        one_hot_winner = np.zeros(3)
        one_hot_winner[winner+1] = 1
        for i in range(len(game[0])-1):
            initial = game[0][i]
            final = game[0][i+1]
            move = np.abs((final-initial).ravel())
            y = np.concatenate([one_hot_winner, move], axis=0).reshape(-1, 12)
            x = initial.reshape(-1, 9)
            if training_set is None:
                training_set = [x, y]
            else:
                training_set[0] = np.vstack((training_set[0], x))[-100000:, :]
                training_set[1] = np.vstack((training_set[1], y))[-100000:, :]
    return training_set


def train_nn(train_data, iterations=1000):
    X = train_data[0]
    y = train_data[1]

    check_point = nn_predictor.LAST
    sess = tf.Session()
    saver = tf.train.import_meta_graph(nn_predictor.META)

    saver.restore(sess, check_point)
    X_tf = sess.graph.get_tensor_by_name('X:0')
    y_tf = sess.graph.get_tensor_by_name('y:0')
    training_op_tf = sess.graph.get_tensor_by_name('training_op:0')
    loss_tf = sess.graph.get_tensor_by_name('loss:0')
    global_step_tf = sess.graph.get_tensor_by_name('global_step:0')
    feed_dict = { X_tf:X, y_tf: y}

    for i in range(iterations):
        _, g_step, loss = sess.run([training_op_tf, global_step_tf, loss_tf], feed_dict=feed_dict)
        if g_step%500 == 0:
            print('training neural network: global_step={}, loss={}'.format(g_step, loss))
    saved_path = saver.save(sess, nn_predictor.CHECK_POINTS_NAME,
        global_step=g_step, write_meta_graph=False)
    nn_predictor.LAST = saved_path


class model_two_hidden():
    def __init__(self, size_x=9, size_y=12):
        self.size_x = size_x
        self.size_y = size_y
        size_hidden_1 = size_x
        size_hidden_2 = size_y

        if not os.path.isdir(nn_predictor.CHECK_POINTS_DIR):
            os.mkdir(nn_predictor.CHECK_POINTS_DIR)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, [None, size_x], name='X')
            self.y = tf.placeholder(tf.float32, [None, size_y], name='y')

            weight1 = tf.Variable(tf.random_uniform([size_x, size_hidden_1]), name='weight1')
            bias1 = tf.Variable(tf.zeros(size_hidden_1), name='bias1')
            activation1 = tf.nn.relu(tf.nn.xw_plus_b(self.X, weight1, bias1), name='activation1')

            weight2 = tf.Variable(tf.random_uniform([size_hidden_1, size_hidden_2]), name='weight2')
            bias2 = tf.Variable(tf.zeros(size_hidden_2), name='bias2')
            activation2 = tf.nn.relu(tf.nn.xw_plus_b(activation1, weight2, bias2), name='activation2')

            weight3 = tf.Variable(tf.random_uniform([size_hidden_2, size_y]), name='weight3')
            bias3 = tf.Variable(tf.zeros(size_y), name='bias3')
            activation3 = tf.nn.relu(tf.nn.xw_plus_b(activation2, weight3, bias3), name='activation3')

            self.loss = tf.reduce_mean(tf.squared_difference(activation3, self.y), name='loss')

            self.predicted_y = activation3
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            optimizer = tf.train.AdamOptimizer(name='adam_optimizer')
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step, name='training_op')
            self.init_op = tf.global_variables_initializer()


class nn_predictor():
    BEST = None
    LAST = None
    CHECK_POINTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nn_check_pts')
    CHECK_POINTS_NAME = os.path.join(CHECK_POINTS_DIR, 'nn')
    META = os.path.join(CHECK_POINTS_DIR, 'nn.meta')

    def __init__(self, nn_type):
        if nn_type in ['best', 'last']:
            if nn_predictor.LAST is None or nn_predictor.BEST is None:
                # if no model was ever constructed
                self.model = model_two_hidden()
                with tf.Session(graph=self.model.graph) as sess:
                    sess.run(self.model.init_op)
                    saver = tf.train.Saver(tf.global_variables())
                    saver.export_meta_graph(nn_predictor.META)
                    init_model = saver.save(sess, nn_predictor.CHECK_POINTS_NAME,
                        global_step=self.model.global_step, write_meta_graph=False)
                nn_predictor.LAST = init_model
                nn_predictor.BEST = init_model
            check_point = nn_predictor.LAST if nn_type == 'last' else nn_predictor.BEST
        else:
            check_point = nn_type

        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(nn_predictor.META)
        saver.restore(self.sess, check_point)
        self.X_tf = self.sess.graph.get_tensor_by_name('X:0')
        self.y_tf = self.sess.graph.get_tensor_by_name('activation3:0')

    def predict(self, input):
        input_np = input.reshape(-1, 9)
        output_np = self.sess.run(self.y_tf, feed_dict={self.X_tf: input_np})
        output_np = output_np.reshape(12)
        return output_np[:3], output_np[3:]

    @classmethod
    def reset_nn_check_pts(cls):
        if not os.path.isdir(cls.CHECK_POINTS_DIR):
            os.mkdir(cls.CHECK_POINTS_DIR)
        for file in os.listdir(cls.CHECK_POINTS_DIR):
            os.remove(os.path.join(cls.CHECK_POINTS_DIR, file))

def main():
    print('')

if __name__ == '__main__':
    main()
