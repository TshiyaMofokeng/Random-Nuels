import numpy as np

def NuelCal(player_accuracies):

    # Take in an array of each players strength and return the list and a dictionary 
    # containing every possible nuel for each configuration of players, their probability to win
    # and their target.
    # Here, they play in random order.
    
    players = sorted(player_accuracies)
    n = len(players)

    games = [[] for _ in range(n)]
    # The (i-1)-th list in games will store information about an every game with i players.

    binary_arrays = [list(map(int, format(i, f'0{n}b'))) for i in range(1, 2**n)]
    #  creates a list containing all binary arrays of length n, except the all-zero array
    #  each binary array in the above list represents a possible state of the game
    # E.g 1100 players 1 and 2 are participating

    # Populate the games list:
    for b in binary_arrays:
        i = sum(b) - 1  
        game_matrix = np.zeros((3, n), dtype=float)
        # row 0: active players in the game. 1 means a player is participating and 0 means otherwise.
        # row 1: will store a players probablity to win the current game
        # row 2: stores a players target
        game_matrix[0] = b  
        games[i].append(game_matrix)


    lookup = {}
    for i in range(1, n):
        for g in games[i]:
            key = tuple(np.where(g[0] == 1)[0])
            lookup[key] = g

    # Compute the values for every duel
    for duel in games[1]:
        p = np.where(duel[0] == 1)[0]
        i, j = p[0], p[1]
        duel[1,i] = players[i] / (players[i]+players[j])
        duel[1,j] = players[j] / (players[i]+players[j])


    for i in range(2, n):
        # Compute values for every possible (i+1)-th nuel, starting at 3.
        for g in games[i]:
            player_sum = 0
            current_players = np.where(g[0] == 1)[0]
            # Loop over only the players participting in nuel with players g[0].
            for p in current_players:
                player_sum += players[p]
                max_prob = -1
                target = -1
                for q in current_players:
                    if p == q:
                        continue
                    # Consider the i-th game, where player p eliminates player q.
                    remaining_players = tuple(sorted(set(current_players) - {q}))
                    subgame = lookup.get(remaining_players)
                    if subgame is not None:
                        prob = subgame[1, p]
                        if prob > max_prob:
                            max_prob = prob
                            target = q
                g[1,p] = max_prob 
                g[2,p] = target

            # Now, compute each players probability.
            for player in current_players:
                g[1,player] =  g[1,player] * players[player] 
                for other_player in current_players:
                    if player == other_player or g[2,other_player] == player:
                        continue
                    
                    other_target = int(g[2,other_player])
                    remaining_players = tuple(sorted(set(current_players) - {other_target}))
                    subgame = lookup.get(remaining_players)
                    if subgame is not None:
                        g[1,player] +=  players[other_player] * subgame[1,player] 
                g[1,player] =  g[1,player] / player_sum

    return games, lookup