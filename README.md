# tic-tac-GO_ZERO
Implementation of Alpha Go Zero algorithm for the game of tic-tac-toe.

A week ago, Deep Mind team at Google published in [Nature](https://www.nature.com/articles/nature24270.epdf) the algorithm of [Alpha Go Zero](https://deepmind.com/blog/alphago-zero-learning-scratch), "an algorithm based solely on reinforcement learning, without human data, guidance or domain knowledge beyond game rules".  The results are nothig but spectacular: "Starting tabula rasa, our new program AlphaGo Zero achieved superhuman performance, winning 100–0 against the previously published, champion-defeating AlphaGo."

My means are more modest (at least when it comes to computational power); the purpose of this project is to replicate the algorithm to play the game of tic-tac-toe to the same super-human performance (i.e. to never lose against a random player, against itself or against a human player).  Of course the condition of "never lose" is because the game of tic-tac-toe is solvable, and it will always end in a draw if the players play perfectly.

The problem Alpha Go is trying to solve is how to play optimally a game with a high branching ratio.  The empty board in Go has 361 positions, so even when many stones had been placed on the board (about 200 for a typical game) the player is still left with a large set of possibilities even though only a small subset of those would constitute good moves.  Monte Carlo Tree Search ([MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)) alone would be too computationally intensive, the basic idea of the algorithm is to use to use a neural network for a "guided" MCTS.  The network provides a prediction of the best next moves (intuition) and an MCTS algorithm explores that space (reflection) simulating certain number of paths.  An Alpha Go Player would receive suggestions from both a NN as well as an MCTS policy.  A NN that is not trained suggests random moves because the weights had been initialized randomly.  If a node was not explored by MCTS (or it has poor statistics) the play will follow the NN prediction.

This is the high level overview of the algorithm:
* Play some games between two identical Alpha Go players (identical NN and MCTS tree).  Allow for a parameter called temperature to generate some noise so that the MCTS space is explored.
* Use the output of the games to:
    * train the NN at predicting the winner as well as the next move given a board configuration.
    * build MCTS statistics
* Play some games between a player with the previous version of the NN and a player with additional training of the NN.  Dial the temperature to zero so that MCTS is deterministic.
    * if the player with additionally trained NN wins, then that network replaces the previous version of the network.
* Repeat

With the caveat: since the board has rotational, inversion and color symmetry perform these operations before feeding the games to the NN.  Additionally, because you don't want your network trained on some games that were pretty much random (at the beginning), the training set for NN is trimmed to the most recent games (that can still be in the hundreds of thousands).

