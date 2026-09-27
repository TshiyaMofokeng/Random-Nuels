
import numpy as np
import torch as th
from stable_baselines3.common.monitor import Monitor
from NuelCalculator import NuelCal
from GymNuel import NuelOpt
from DQN.DQNAgent import MyDQNAgent
from pathlib import Path

NUM_PLAYERS = 9
agent_num = 8
save_dir = Path(f"models/my_models/{NUM_PLAYERS}_players")           
save_dir.mkdir(parents=True, exist_ok=True) 

players = np.load(save_dir / "player_accuracies.npy" )


game, lookup = NuelCal(player_accuracies=players)

env = Monitor(NuelOpt(player_accuracies=players,agent_num=agent_num, lookup= lookup))
n = env.unwrapped.n

model = MyDQNAgent(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=1.0,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[32, 32, 32], activation_fn=th.nn.ReLU),
    verbose=0,
    seed=0,
)

model.learn(total_timesteps=1_000_000)

model.save(save_dir / f"nuel_dqn_agent_{agent_num}.pt")

