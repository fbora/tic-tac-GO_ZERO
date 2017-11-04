# tic-tac-GO_ZERO
_Implementation of Alpha Go Zero algorithm for the game of tic-tac-toe._

run training.py then interactive.py

Two weeks ago, Deep Mind team at Google published in [Nature](https://www.nature.com/articles/nature24270.epdf) the algorithm of [Alpha Go Zero](https://deepmind.com/blog/alphago-zero-learning-scratch), "an algorithm based solely on reinforcement learning, without human data, guidance or domain knowledge beyond game rules".  The results are nothig but spectacular: "Starting tabula rasa, our new program AlphaGo Zero achieved superhuman performance, winning 100–0 against the previously published, champion-defeating AlphaGo."

My means are more modest (at least when it comes to computational power); the purpose of this project is to replicate the algorithm to play the game of tic-tac-toe to the same super-human performance (i.e. to never lose against a random player, against itself or against a human player).  Of course the condition of "never lose" is because the game of tic-tac-toe is solvable, and it will always end in a draw if the players play perfectly.  This code is the result of two weeks of paper reading and python hacking.

The problem Alpha Go is trying to solve is how to play optimally a game with a high branching ratio.  The empty board in Go has 361 positions, so even when many stones had been placed on the board (about 200 for a typical game) the player is still left with a large set of possibilities even though only a small subset of those would constitute good moves.  Monte Carlo Tree Search ([MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)) alone would be too computationally intensive, the basic idea of the algorithm is to use to use a neural network for a "guided" MCTS.  The network provides a prediction of the best next moves (intuition) and an MCTS algorithm explores that space (reflection) simulating certain number of paths.  An Alpha Go Player would receive suggestions from both a NN as well as an MCTS policy.  A NN that is not trained suggests random moves because the weights had been initialized randomly.  If a node was not explored by MCTS (or it has poor statistics) the play will follow the NN prediction.

Previous versions of Alpha Go used two distinct networks: one for predicting the outcome of the game and one for predicting the next moves.  As it turns out a single network that predicts both the outcome and the moves is better.  It's as if the next move prediction is more accurate when it is subject to the constraint that it also predicts the outcome of the game.  I find this a pretty big deal!

This is the high level overview of the algorithm:
* Play some games between two identical Alpha Go players (identical NN and MCTS tree).  Allow for a parameter called temperature to generate some noise so that the MCTS space is explored.
* Use the output of the games to:
    * train the NN at predicting the winner as well as the next move given a board configuration.
    * build MCTS statistics
* Play some games between a player with the previous version of the NN and a player with additional training of the NN.  Dial the temperature to zero so that MCTS is deterministic.
    * if the player with additionally trained NN wins, then that network replaces the previous version of the network.
* Repeat

With the caveat: since the board has rotational, inversion and color symmetry perform these operations before feeding the games to the NN.  Additionally, because you don't want your network trained on some games that were pretty much random (at the beginning), the training set for NN is trimmed to the most recent games (that can still be in the hundreds of thousands).

That's the layout of the training.py file:

'''

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
'''

### MCTS

For each move in MCTS the evaluation has 1600 paths as follows:
* for the first 30 moves the paths draw from U[N_a/N] with the temperature set to 1.
* for moves after 30, you shock the prior probability distribution given by the NN
P(s,a) = P(s,a) + Dirichlet(0.03) and chose the node that maximizes the action: Q(t) + c * P(s,a) \sqrt(N)/(1+N_a)
For the procedure there are no roll-outs (i.e. random play outs from the expanded node).  If a new node is encountered the statistics are initialized to N=0, W=0, Q=0 and the process continues recursively until the game ends.  It's useful to note that in the limits
    * lim_{N->0} a_t ~ P(s,a)
    * lim_{N->\infty} = Q(t)
All paths in the MC sample will update the node statistics.  While 1600 paths for a single symulation seems a small number, after many games have been played, we are going to have a good statistics and the sampling will follow the distribution of Q(t)
After MCTS, a move is played according to U[N_a/N], but in this case it's only N=1600 paths that count for the choice.  After training had begun, the statistics a_t is practically proportional to the mean value of the action Q(t), but it's better to use the total number of visits to the edge N_a rather than Q, because for small N Q is more susceptible to outliers.
  
My version of MCTS uses the same version of PUCT function but there is no evaluation or random play for the next move.  For all possible moves (s, a) from a given state s, the next move is a single draw from the set of normalized PUCT values.  Before normalization the values are mapped to positive numbers by adding a constant because in tick tack toe the second player has a negative mean action value $Q(t)$ for most moves.  In the end it doesn't matter what particular choices you make for the exploration function, as long as the measured mean action converges to the true mean action after a long number of steps.
 
The paths statistics are stored in a dictionary with a string key given by 'initialstate2finalstate' string.  After serious considerations I decided to not make the statistics color invariant because from the following board state:
 
    xx_
    ___  
    oo_
 
there is ambiguity on which is the best move for x or o depending what player moves.  The statistics assume that the first player always start with 'x', and for the moves of the second player we accumulate the statistics from their perspective; i.e. a win of the 'o' player improves their average action.
 
### Neural network
 
Go is a more complex game than tick-tac-toe, and the neural network they use is consequently more elaborate.  In our case I use a simple neural network with two layers and RELU activation.  The loss function in alpha go is a sum of MSE for game winner and logits for the move with an L2 penalty.  We use only MSE penalty for the 12 dimensional output of catenated one-hot-encoding winner and next move.
 
 
### Evaluation
During self play, when the purpose is to accumulate self play games the order is not important, but for the evaluation of the better neural network the order of play is important.  The second player in tic-tac-toe has the disadvantage of playing one less move than the first player.  We introduce a symmetric play where each player takes turns starting the game.  The same symmetric play is used after training to compare the Zero player to the Random player.  For two random players the win rate is 46% with the rest of the games ending in draw.  The trained Zero player has a win rate of 95% and a draw rate of 5%.  Occasionally, depending on parameters, and state of the network at the beginning of training, we may have cases where the space of possible move is not fully explored resulting in a defeat rate of <0.1% for the current parameters.  I believe this can be fixed by tuning the network, PUCT constant and more training iterations.
 
**Note that I did not spent time to improve the efficiency, all code is single threaded, and it takes about 2 hours to run training.py script.**

Interesting fact: with the current script settings not all paths in the Monte Carlo are explored, with 1-2% not gathering any statistics.  If we play a large number of games (1 million) against the random player, all these paths are going to be explored, but the trained player still never loses.  When there are no statistics accumulated by MCTS the play is the recommendation made by the neural network, which means that the network learned how to play the game well by predicting what MCTS would play.
 
 
After training, run interactive.py script to play against the machine :)




