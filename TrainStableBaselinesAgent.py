import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from NuelCalculator import NuelCal
from GymNuel import NuelOpt
from pathlib import Path


save_dir = Path("models/stable_baselines_models")            
save_dir.mkdir(parents=True, exist_ok=True) 


players = [0.3 ,0.4, 0.45, 0.5 ,0.55, 0.6 ,0.65, 0.72 , 0.82 ,0.95]
player_num = 8
game, lookup = NuelCal(player_accuracies=players)
env = Monitor(NuelOpt(player_accuracies=players, player_num=player_num, lookup=lookup))
n = env.unwrapped.n

model = DQN(
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
    policy_kwargs=dict(net_arch=[32,32,32]),
    verbose=0,
    seed=0,
)
model.learn(total_timesteps=1_000_000)

model.save(save_dir / f"nuel_dqn_agent_{player_num}.pt")
