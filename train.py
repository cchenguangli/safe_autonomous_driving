
import os
import time
import csv
import numpy as np
from stable_baselines3 import SAC
#from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

from sac_env import CarlaEnv


class BestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

        self.episode_rewards_file = os.path.join(log_dir, "episode_rewards.csv")
        with open(self.episode_rewards_file, mode="w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward"])

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            ep_reward = self.locals["infos"][0]["episode"]["r"]
            ep_length = self.locals["infos"][0]["episode"]["l"]
            with open(self.episode_rewards_file, mode="a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ep_length, ep_reward])

        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(y) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose:
                    print(f"Timesteps: {self.num_timesteps}, Mean Reward: {mean_reward:.2f}, Best: {self.best_mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
#                    self.model.save_replay_buffer(os.path.join(self.log_dir, "best_replay_buffer.pkl"))
        return True


def train():
    import torch
    print("Enable GPU Training：", torch.cuda.is_available(), torch.cuda.get_device_name(0))

    log_dir = "./logs_sac"
    os.makedirs(log_dir, exist_ok=True)

    env = CarlaEnv(town='Town03')

    env = Monitor(env, log_dir)

#    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cuda")
    model = SAC("MlpPolicy",
                env,                
                buffer_size=int(4e5),
                learning_starts=int(5e4),
                batch_size=256,
                device='cuda',
                verbose=1,
                tensorboard_log=log_dir)
    
                
                
    callback = BestTrainingRewardCallback(check_freq=2048, log_dir=log_dir)

    start_time = time.time()
    model.learn(total_timesteps=2_000_000, callback=callback)
    model.save(os.path.join(log_dir, "last_model"))
#    model.save_replay_buffer(os.path.join(log_dir, "last_replay_buffer.pkl"))  
    print("Training complete.")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    train()
    
    
    
    
    
    
    
    

