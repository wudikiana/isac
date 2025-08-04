# models/ppo_policy.py
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    """优化后的PPO策略网络"""
    
    def __init__(self, obs_dim, action_dims, hidden_size=256):
        super().__init__()
        
        # 共享特征提取层（增强版）
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # 演员网络（连续动作）
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dims[0]),
            nn.Tanh()  # 输出在[-1,1]范围
        )
        
        # 评论家网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        
        # 动作缩放参数（注册为buffer）
        self.register_buffer('action_scale', torch.tensor([0.4], dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor([0.5], dtype=torch.float32))
    
    def forward(self, obs):
        """前向传播"""
        features = self.feature_extractor(obs)
        
        # 演员输出（缩放至[0.1,0.9]范围）
        action_mu = self.actor(features) * self.action_scale + self.action_bias
        
        # 评论家输出
        value = self.critic(features)
        
        return action_mu, value
    
    def get_action(self, obs):
        """根据观测获取动作"""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_mu, value = self.forward(obs_tensor)
            
            # 动作分布（连续）
            dist = Normal(action_mu, 0.1)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action.squeeze(0).numpy(), log_prob, value

class PPO:
    """PPO算法实现"""
    
    def __init__(self, 
                 policy, 
                 optimizer, 
                 clip_epsilon=0.2, 
                 value_coeff=0.5, 
                 entropy_coeff=0.01):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
    
    def update(self, rollouts):
        """使用经验更新策略"""
        # 从经验中提取数据
        obs = torch.as_tensor(rollouts['obs'], dtype=torch.float32)
        actions = torch.as_tensor(rollouts['actions'], dtype=torch.float32)
        old_log_probs = torch.as_tensor(rollouts['log_probs'], dtype=torch.float32)
        returns = torch.as_tensor(rollouts['returns'], dtype=torch.float32)
        advantages = torch.as_tensor(rollouts['advantages'], dtype=torch.float32)
        
        # 前向传播获取新策略的值
        action_mu, values = self.policy(obs)
        
        # 计算新策略的对数概率
        dist = Normal(action_mu, 0.1)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # 计算熵
        entropy = dist.entropy().mean()
        
        # 计算概率比
        ratio = torch.exp(log_probs - old_log_probs)
        
        # PPO裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
        
        # 总损失
        loss = actor_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }