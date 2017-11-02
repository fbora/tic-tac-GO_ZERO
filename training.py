__author__='Florin Bora'

import player
import game
import neural_network
import mcts

def train():
    mcts.MCTS.get_tree_and_edges(reset=True)
    neural_network.nn_predictor.reset_nn_check_pts()
    nn_training_set = None

    iterations = 50

    for _ in range(iterations):
        player1 = player.Zero_Player('x', 'Bot_ONE', nn_type='best', temperature=1)
        player2 = player.Zero_Player('o', 'Bot_ONE', nn_type='best', temperature=1)
        self_play_game = game.Game(player1, player2)
        self_play_results = self_play_game.play(500)
        augmented_self_play_results = neural_network.augment_data_set(self_play_results)

        mcts.MCTS.update_mcts_edges(augmented_self_play_results)
        nn_training_set = neural_network.update_nn_training_set(self_play_results, nn_training_set)

        neural_network.train_nn(nn_training_set)

        player1 = player.Zero_Player('x', 'Bot_ONE', nn_type='last', temperature=0)
        player2 = player.Zero_Player('o', 'Bot_ONE', nn_type='best', temperature=0)

        nn_test_game = game.Game(player1, player2)
        wins_player1, wins_player2 = nn_test_game.play_symmetric(100)

        if wins_player1 >= wins_player2:
            neural_network.nn_predictor.BEST = neural_network.nn_predictor.LAST


def zero_vs_random():
    N_games = 100
    player1 = player.Random_Player('o', 'Bot_RANDOM1')
    player2 = player.Random_Player('o', 'Bot_RANDOM2')
    r_vs_r_game = game.Game(player1, player2)
    w1, w2 = r_vs_r_game.play_symmetric(N_games)
    print('{} vs {} summary:'.format(player1.name, player2.name))
    print('wins={}, draws={}, losses={}'.format(w1, N_games - w1 - w2, w2))

    global_step = 50000
    nn_check_pt = neural_network.nn_predictor.CHECK_POINTS_NAME + '-' + str(global_step)
    player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type=nn_check_pt, temperature=0)
    player2 = player.Random_Player('o', 'Bot_RANDOM')
    z_vs_r_game = game.Game(player1, player2)
    w1, w2 = z_vs_r_game.play_symmetric(N_games)
    print('{} vs {} summary:'.format(player1.name, player2.name))
    print('wins={}, draws={}, losses={}'.format(w1, N_games-w1-w2, w2))

    player1 = player.Zero_Player('x', 'Bot_ZERO1', nn_type=nn_check_pt, temperature=0)
    player2 = player.Zero_Player('x', 'Bot_ZERO2', nn_type=nn_check_pt, temperature=0)
    z_vs_z_game = game.Game(player1, player2)
    w1, w2 = z_vs_z_game.play_symmetric(N_games)
    print('{} vs {} summary:'.format(player1.name, player2.name))
    print('wins={}, draws={}, losses={}'.format(w1, N_games - w1 - w2, w2))


def main():
    train()
    zero_vs_random()

if __name__ == '__main__':
    main()
