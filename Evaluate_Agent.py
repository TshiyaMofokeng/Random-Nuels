import numpy as np
from NuelCalculator import NuelCal
from GymNuel import NuelOpt
from DQN.DQNAgent import MyDQNAgent

NUM_PLAYERS = 4

def report(nuel_data):
    for i in range(len(nuel_data)):
        print(f"{i + 3} player nuel: {nuel_data[i]}")
    return

def correctness(model :MyDQNAgent, env: NuelOpt, lookup : dict, agent_num :int):
    
    # Returns nuel_data where nuel_data[k] is the % of states with (k + 3) players alive
    # where the agent's target matches NuelCal's optimal target.
    # nuel_data[0] is the 3-player case, nuel_data[1] the 4-player case, etc.

    opponents = env.unwrapped.opponents
    n = env.unwrapped.n
    rows = []
    nuel_data = []

    for alive, game in lookup.items():
        if agent_num not in alive:
            # Only count games the agent is in
            continue      
        if len(alive) < 3:
            # Don't count duels
            continue                             

        
        obs = np.zeros(n, dtype=np.float32)
        obs[list(alive)] = 1
        mask = np.array([o in alive for o in opponents], dtype=bool)

        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        agent_target = opponents[int(a)]
        optimal_target = int(game[2, agent_num])

        rows.append((alive, agent_target, optimal_target, agent_target == optimal_target))

    nuel_data = []
    for size in range(3, n + 1):
        group = [r[3] for r in rows if len(r[0]) == size]
        nuel_data.append(100 * np.mean(group) if group else float("nan"))
    return nuel_data

# Example: 
from pathlib import Path
import torch as th

load_dir = Path(f"models/my_models/{NUM_PLAYERS}_players")
players = np.load(load_dir / "player_accuracies.npy" )

player_num = 0
lookup = NuelCal(players)[1]
env = NuelOpt(player_accuracies=players, agent_num=player_num, lookup=lookup )

agent = MyDQNAgent("MlpPolicy", env)          
agent.load(load_dir / f"nuel_dqn_agent_{player_num}.pt")

data = correctness(model=agent, env=env, lookup=lookup, agent_num=player_num)

report(data)