# train/train_ppo.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.ppo_policy import ActorCritic, PPO
from env.isac_sat_env import ISAC_SatEnv
from collections import deque
import argparse
import wandb
import json

class RolloutBuffer:
    """优化后的经验回放缓冲区"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def add(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def __len__(self):
        return len(self.rewards)
    
    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """计算GAE优势函数和回报"""
        returns = []
        advantages = []
        gae = 0
        next_value = last_value
        next_done = 0
        
        # 反向计算
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t+1]
                next_values = self.values[t+1]
            
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        # 优势归一化
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'obs': np.array(self.obs),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'returns': np.array(returns),
            'advantages': advantages
        }

def train_ppo(config=None):
    """优化后的PPO训练主函数"""
    # 初始化环境（带优化参数）
    if config is None:
        config = {
            "total_power": 50.0,
            "total_bandwidth": 200e6,
            "comm_snr_thresh": 5.0,
            "radar_snr_thresh": -10.0,
            "max_steps": 500
        }
    env = ISAC_SatEnv(config)
    
    # 训练参数
    train_params = {
        "num_episodes": 2000,
        "max_steps": 500,
        "batch_size": 128,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coeff": 0.01,
        "lr": 5e-4,
        "update_interval": 2048  # 按步数更新
    }
    
    # 初始化PPO
    policy = ActorCritic(
        env.observation_space.shape[0],
        [np.prod(env.action_space.shape)],
        hidden_size=256
    )
    optimizer = optim.Adam(policy.parameters(), lr=train_params["lr"])
    ppo = PPO(
        policy, 
        optimizer,
        clip_epsilon=train_params["clip_epsilon"],
        entropy_coeff=train_params["entropy_coeff"]
    )
    
    # 初始化WandB
    wandb.init(
        project="satellite-isac-ppo",
        config={
            "env_config": config,
            "train_params": train_params
        }
    )
    
    # 训练统计
    buffer = RolloutBuffer()
    global_step = 0
    best_reward = -np.inf
    
    for episode in range(train_params["num_episodes"]):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        for step in range(train_params["max_steps"]):
            # 获取动作
            action, log_prob, value = policy.get_action(obs)
            
            # 执行动作（注意reshape）
            next_obs, reward, done, info = env.step(
                action.reshape(env.action_space.shape)
            )
            
            # 存储经验
            buffer.add(obs, action, log_prob, value.item(), reward, done)
            episode_reward += reward
            global_step += 1
            
            # 定期更新策略
            if len(buffer) >= train_params["update_interval"]:
                # 计算最终状态价值
                with torch.no_grad():
                    _, last_value = policy(
                        torch.FloatTensor(next_obs).unsqueeze(0)
                    )
                # 计算GAE和回报
                rollouts = buffer.compute_returns(
                    last_value.item(),
                    gamma=train_params["gamma"],
                    gae_lambda=train_params["gae_lambda"]
                )
                
                # 更新策略
                loss_info = ppo.update(rollouts)
                buffer.reset()
                
                # 记录训练指标
                wandb.log({
                    "train/loss": loss_info["total_loss"],
                    "train/actor_loss": loss_info["actor_loss"],
                    "train/value_loss": loss_info["value_loss"],
                    "train/entropy": loss_info["entropy"],
                    "metrics/global_step": global_step
                })
            
            if done:
                break
        
        # 记录episode指标
        wandb.log({
            "episode/reward": episode_reward,
            "episode/steps": step + 1,
            "episode/episode": episode
        }, step=global_step)
        
        # 定期评估并保存模型
        if episode % 10 == 0:
            avg_reward = evaluate_policy(env, policy)
            wandb.log({"eval/avg_reward": avg_reward}, step=global_step)
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(policy.state_dict(), "models/ppo_best.pth")
                print(f"Episode {episode}: 保存最佳模型，平均奖励 {avg_reward:.2f}")
    
    # 保存最终模型
    torch.save(policy.state_dict(), "models/ppo_final.pth")
    wandb.finish()
    print("训练完成！")

def evaluate_policy(env, policy, n_eval_episodes=5):
    """策略评估函数"""
    total_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _, _ = policy.get_action(obs)
            obs, reward, done, _ = env.step(
                action.reshape(env.action_space.shape)
                )
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=2000)
    args = parser.parse_args()
    
    # 加载配置
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    train_ppo(config)