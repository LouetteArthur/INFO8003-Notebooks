import torch
import gymnasium as gym
import numpy as np
from ppo import Agent


def run_trained_policy(model_path, env_id="CartPole-v1", render=True, max_steps=1000):
    """Runs a trained PPO policy in the specified environment."""
    
    # Load environment
    env = gym.make(env_id, render_mode="human" if render else None)
    obs, _ = env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def make_env():
        return gym.make("CartPole-v1") 

    envs = gym.vector.SyncVectorEnv([make_env])

    # Load trained agent
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    # Run one episode
    total_reward = 0
    for step in range(max_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        obs, reward, done, truncated, _ = env.step(action.cpu().numpy()[0])
        total_reward += reward
        
        if done or truncated:
            break
    
    env.close()
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    model_filename = "demo_ppo_agent.pth"  # Replace with actual filename
    run_trained_policy(model_filename)