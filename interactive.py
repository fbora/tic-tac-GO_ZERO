__author__ = 'Florin Bora'

import neural_network
import player
import game
import mcts

def interactive_game():
    mcts.MCTS.PUCT_CONSTANT = 0.33
    global_step = 50000
    nn_check_pt = neural_network.nn_predictor.CHECK_POINTS_NAME + '-' + str(global_step)
    player1 = player.Zero_Player('x', 'Bot_ZERO', nn_type=nn_check_pt, temperature=0)
    player2 = player.Interactive_Player('o', 'Human')
    z_v_h_game = game.Game(player1, player2)
    outcome = z_v_h_game.run()

    z_v_h_game.board.display(clear=True)
    if outcome[1] == 0:
        print('Game ended in draw!')
    else:
        winner = player1 if outcome[1] == player1.type else player2
        print('{} won the game!'.format(winner.name))


def main():
    interactive_game()

if __name__ == '__main__':
    main()
