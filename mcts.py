__author__='Florin Bora'

import os
import numpy as np
import board
import pickle

class MCTS():

    MCTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mcts')
    PUCT_CONSTANT = 10.0
    TREE_FILE = 'tree.pkl'
    EDGES_FILE = 'edges.pkl'
    TREE_PATH = os.path.join(MCTS_DIR, TREE_FILE)
    EDGES_PATH = os.path.join(MCTS_DIR, EDGES_FILE)
    WIN2DICT_MAP = {-1: 'L', 0: 'D', 1: 'W'}

    def __init__(self):
        pass


    @classmethod
    def get_tree_and_edges(cls, reset=False):
        if not os.path.isdir(cls.MCTS_DIR):
            os.mkdir(cls.MCTS_DIR)
        if reset:
            for file in os.listdir(cls.MCTS_DIR):
                os.remove(os.path.join(cls.MCTS_DIR, file))
        if not os.listdir(cls.MCTS_DIR):
            tree, edges = board.Board.generate_state_space()
            cls.save_tree_edges(tree, edges)
        else:
            tree, edges = cls.load_tree_edges()
        return tree, edges


    @classmethod
    def save_tree_edges(cls, tree, edges):
        with open(cls.TREE_PATH, 'wb') as t:
            pickle.dump(tree, t, pickle.HIGHEST_PROTOCOL)
        with open(cls.EDGES_PATH, 'wb') as e:
            pickle.dump(edges, e, pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load_tree_edges(cls):
        with open(cls.TREE_PATH, 'rb') as t:
            tree = pickle.load(t)
        with open(cls.EDGES_PATH, 'rb') as e:
            edges = pickle.load(e)
        return tree, edges


    @classmethod
    def update_mcts_edges(cls, new_games):
        tree, edges = cls.get_tree_and_edges()
        for game in new_games:
            for i in range(len(game[0])-1):
                initial = game[0][i]
                final = game[0][i+1]
                move = (final-initial).sum()
                edge = board.Board.arr2str(initial)+'2'+board.Board.arr2str(final)
                win = game[1] * move
                edges[edge]['N'] += 1
                edges[edge][cls.WIN2DICT_MAP[win]] += 1
                action = (edges[edge]['W']-edges[edge]['L'])/edges[edge]['N']
                edges[edge]['Q'] = action
        cls.save_tree_edges(tree, edges)

    @classmethod
    def PUCT_function(cls, N, edge):
        return edge['Q'] + cls.PUCT_CONSTANT * edge['P'] * np.sqrt(N) / (1+edge['N'])


def print_edges(edges):
    for k in edges.keys():
        if edges[k]['N'] == 0:
            print('|{}|'.format(k), edges[k])


def main():
    t, e = MCTS.get_tree_and_edges()
    print_edges(e)

if __name__ == '__main__':
    main()
