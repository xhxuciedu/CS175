#!/usr/bin/env python3
"""
Test script for the improved Actor-Critic implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class PolicyNetwork(nn.Module):
    """Neural network for policy approximation"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

class ValueNetwork(nn.Module):
    """Neural network for value function approximation"""
    
    def __init__(self, state_size, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class ImprovedActorCriticAgent:
    """Improved Actor-Critic algorithm with better training stability"""
    
    def __init__(self, state_size, action_size, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, 
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Actor (policy) network
        self.actor = PolicyNetwork(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic (value) network
        self.critic = ValueNetwork(state_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Experience buffer for batch updates
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
    def act(self, state, training=True):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs = self.actor(state)
        
        if training:
            # Sample action from policy
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            return action.item()
        else:
            # Greedy action selection
            return action_probs.argmax().item()
    
    def get_value(self, state):
        """Get state value from critic"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return self.critic(state)
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in experience buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def clear_buffer(self):
        """Clear experience buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def update(self, batch_size=32):
        """Update both actor and critic networks with improved stability"""
        if len(self.states) < batch_size:
            return None, None
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)
        next_states = torch.FloatTensor(self.next_states).to(device)
        dones = torch.BoolTensor(self.dones).to(device)
        
        # Get current values and next values
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        
        # Calculate TD targets with proper handling of terminal states
        td_targets = rewards + self.gamma * next_values * (~dones)
        advantages = td_targets - values
        
        # Critic loss (MSE) - ensure proper tensor shapes
        critic_loss = F.mse_loss(values, td_targets.detach())
        
        # Actor loss (policy gradient with advantage)
        action_probs = self.actor(states)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        
        # Policy loss with advantage
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Entropy bonus for exploration
        entropy_loss = -action_dist.entropy().mean()
        
        # Total actor loss
        actor_loss = policy_loss + self.entropy_coef * entropy_loss
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Clear buffer
        self.clear_buffer()
        
        return actor_loss.item(), critic_loss.item()

def train_improved_actor_critic(env, agent, episodes=500, max_steps=500, update_frequency=10, batch_size=32):
    """Train Improved Actor-Critic agent with better stability"""
    scores = []
    actor_losses = []
    critic_losses = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = agent.act(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Update networks periodically
            if (step + 1) % update_frequency == 0 or done:
                actor_loss, critic_loss = agent.update(batch_size)
                if actor_loss is not None and critic_loss is not None:
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
            
            if done:
                break
        
        scores.append(total_reward)
        
        # Print progress
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
    
    return scores, actor_losses, critic_losses

def main():
    # Create environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"Environment: {env.spec.id}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Create Improved Actor-Critic agent
    improved_ac_agent = ImprovedActorCriticAgent(
        state_size, action_size, 
        lr_actor=3e-4, lr_critic=1e-3, gamma=0.99,
        entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5
    )
    print("Improved Actor-Critic agent created!")
    
    # Train Improved Actor-Critic agent
    print("Training Improved Actor-Critic agent...")
    improved_ac_scores, improved_ac_actor_losses, improved_ac_critic_losses = train_improved_actor_critic(
        env, improved_ac_agent, episodes=500, max_steps=500, update_frequency=10, batch_size=32
    )
    print("Training complete!")
    
    # Print final statistics
    final_avg_score = np.mean(improved_ac_scores[-100:])
    print(f"\nImproved Actor-Critic Final Statistics:")
    print(f"Final 100-episode average score: {final_avg_score:.2f}")
    print(f"Maximum score achieved: {max(improved_ac_scores)}")
    print(f"Episodes to solve (score > 195): {next((i for i, score in enumerate(improved_ac_scores) if score >= 195), 'Not solved')}")
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_scores = []
    for _ in range(10):
        state, _ = env.reset()
        total_reward = 0
        for _ in range(500):
            action = improved_ac_agent.act(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
        test_scores.append(total_reward)
    
    print(f"Test performance (10 episodes): {np.mean(test_scores):.2f} ± {np.std(test_scores):.2f}")
    
    env.close()

if __name__ == "__main__":
    main()
