__author__='Florin Bora'

import os
import numpy as np
import functools

class Board():
    INT2STR_MAP = {0: ' ', 1: 'x', -1: 'o'}
    STR2INT_MAP = {' ': 0, 'x': 1, 'o': -1}
    SIZE = np.array([3, 3])
    WIN_SUM = 3

    def __init__(self):
        self.board = np.zeros(Board.SIZE)

    def reset(self):
        self.board.fill(0)

    def add_move(self, play, row, col):
        if self.board[row, col] != 0:
            raise Exception('invalid move')
        self.board[row, col] = play

    def empty(self):
        return self.board.ravel().astype(bool).sum() == 0

    def full(self):
        return self.board.ravel().astype(bool).sum() == Board.SIZE.prod()

    def display(self, clear=False):
        if clear:
            os.system('cls')
        print(
            '\n'
            '\t {0} | {1} | {2}\n'
            '\t---+---+---\n'
            '\t {3} | {4} | {5}\n'
            '\t---+---+---\n'
            '\t {6} | {7} | {8}\n\n'.format(
                *[Board.INT2STR_MAP[x] for x in self.board.ravel()]))


    @classmethod
    def arr2str(cls, arr):
        return ''.join(Board.INT2STR_MAP[i] for i in arr.ravel())

    @classmethod
    def str2arr(cls, strmove):
        return np.array([Board.STR2INT_MAP[i] for i in strmove]).reshape(cls.SIZE)

    @classmethod
    def stringmove2int(cls, stringmove):
        init, final = stringmove.split('2')
        idx = [i for i in range(len(init)) if init[i]!=final[i]][0]
        return idx

    @classmethod
    def winner(cls, board):
        '''Returns the winner of the board by checking rows, columns and diagonals'''
        rows = board.sum(axis=0)
        row_winner = np.sign(rows[np.where(np.abs(rows) == Board.WIN_SUM)])
        if len(row_winner)>0:
            return row_winner[0]

        cols = board.sum(axis=1)
        col_winner = np.sign(cols[np.where(np.abs(cols) == Board.WIN_SUM)])
        if len(col_winner) > 0:
            return col_winner[0]

        diag = np.diag(board).sum()
        if abs(diag) == Board.WIN_SUM:
            return np.sign(diag)

        off_diag = np.diag(np.fliplr(board)).sum()
        if abs(off_diag) == Board.WIN_SUM:
            return np.sign(off_diag)

        return 0


    @classmethod
    def generate_state_space(cls):
        def f(i, p, m):
            l = list(p)
            l[i] = m
            return l

        tree = dict()
        edges = list()

        root = cls.SIZE.prod()*' '
        parents = [root]
        curr_move = 'x'

        for level in range(len(root)):
            for p in parents:
                possible_moves = [i for i in range(len(p)) if p[i] == ' ']
                children = set([''.join(f(x, p, curr_move)) for x in possible_moves])
                tree[p] = children
                edges += [p+'2'+c for c in children]
            parents = functools.reduce(set.union, tree.values())
            parents = set(parents) - set(tree.keys())
            parents = set([x for x in parents if not cls.winner(cls.str2arr(x))])
            curr_move = 'x' if curr_move!='x' else 'o'
        edges = set(edges)
        edges_statistics = dict()
        for k in edges:
            edges_statistics[k] = {'N': 0, 'W': 0, 'D': 0, 'L': 0, 'Q': 0, 'P': 0}

        return tree, edges_statistics


def main():
    print('')

if __name__ == '__main__':
    main()
