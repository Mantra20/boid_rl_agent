"""
Entrainement DQN.

Usage (depuis la racine du projet) :
    python scripts/train.py
    python scripts/train.py --episodes 5000 --boids 15
    python scripts/train.py --resume models/dqn_best.pt
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

from env import BoidFollowerEnv
from agent import DQNAgent

MODELS_DIR  = Path(__file__).parent.parent / "models"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def plot_curves(all_rewards, avg100, all_losses, tag=""):
    save_path = OUTPUTS_DIR / f"training_curves{tag}.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    eps = np.arange(1, len(all_rewards) + 1)
    ax.plot(eps, all_rewards, color="#b5d4f4", linewidth=0.6, alpha=0.7, label="reward/ep")
    if avg100:
        ax.plot(
            np.arange(100, len(all_rewards) + 1), avg100,
            color="#185fa5", linewidth=1.8, label="moyenne 100 ep",
        )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("episode")
    ax.set_ylabel("reward total")
    ax.set_title("Courbe de recompense")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if all_losses:
        smoothed = np.convolve(all_losses, np.ones(50) / 50, mode="valid")
        ax.plot(all_losses, color="#f5c4b3", linewidth=0.5, alpha=0.6, label="loss brute")
        ax.plot(smoothed,   color="#993c1d", linewidth=1.6, label="lissee (50)")
        ax.set_xlabel("pas d'apprentissage")
        ax.set_ylabel("Huber loss")
        ax.set_title("Courbe de perte (DQN)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"courbes -> {save_path}")
    plt.show()


def train(n_episodes=4000, n_boids=15, seed=42, resume=None):
    env   = BoidFollowerEnv(n_boids=n_boids, seed=seed)
    agent = DQNAgent(obs_dim=env.obs_dim, n_actions=env.n_actions, eps_decay=n_episodes // 2)

    if resume:
        agent.load(resume)
        print(f"reprise depuis {resume}")

    window     = deque(maxlen=100)
    best_avg   = -np.inf
    all_rewards, avg100 = [], []

    for ep in range(1, n_episodes + 1):
        obs, ep_reward, in_flock = env.reset(), 0.0, 0

        while True:
            action            = agent.act(obs)
            next_obs, r, done = env.step(action)
            agent.push(obs, action, r, next_obs, done)
            obs        = next_obs
            ep_reward += r

            if np.linalg.norm(env.agent_pos - env.sim.centroid()) < 3.0:
                in_flock += 1
            if done:
                break

        agent.end_episode()
        all_rewards.append(ep_reward)
        window.append(ep_reward)
        if ep >= 100:
            avg100.append(float(np.mean(window)))

        if ep % 200 == 0:
            avg      = float(np.mean(window))
            loss_avg = float(np.mean(agent.loss_history[-500:])) if agent.loss_history else 0.0
            print(f"ep {ep:5d}/{n_episodes}  avg_r={avg:7.2f}  in_flock={in_flock:2d}/50  eps={agent.eps:.3f}  loss={loss_avg:.4f}")
            if avg > best_avg:
                best_avg = avg
                agent.save(str(MODELS_DIR / "dqn_best.pt"))

    agent.save(str(MODELS_DIR / "dqn.pt"))
    print(f"modeles -> {MODELS_DIR}/dqn.pt  /  dqn_best.pt")

    plot_curves(all_rewards, avg100, agent.loss_history)
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--boids",    type=int, default=15)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--resume",   type=str, default=None)
    args = parser.parse_args()
    train(args.episodes, args.boids, args.seed, args.resume)
