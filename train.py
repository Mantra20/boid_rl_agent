"""
Entraînement DQN.

Usage :
    python train.py
    python train.py --episodes 5000 --boids 15
    python train.py --resume dqn.pt
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from env import BoidFollowerEnv
from agent import DQNAgent


def plot_curves(all_rewards: list[float], avg100: list[float],
                all_losses: list[float], save_path: str = "training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- reward ---
    ax = axes[0]
    eps = np.arange(1, len(all_rewards) + 1)
    ax.plot(eps, all_rewards, color="#b5d4f4", linewidth=0.6, alpha=0.7, label="reward/ep")
    if avg100:
        ax.plot(
            np.arange(100, len(all_rewards) + 1),
            avg100,
            color="#185fa5", linewidth=1.8, label="moyenne 100 ep",
        )
    ax.set_xlabel("épisode")
    ax.set_ylabel("reward total")
    ax.set_title("Courbe de récompense")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- loss ---
    ax = axes[1]
    if all_losses:
        smoothed = np.convolve(all_losses, np.ones(50) / 50, mode="valid")
        ax.plot(all_losses, color="#f5c4b3", linewidth=0.5, alpha=0.6, label="loss brute")
        ax.plot(smoothed,   color="#993c1d", linewidth=1.6, label="lissée (50)")
        ax.set_xlabel("pas d'apprentissage")
        ax.set_ylabel("Huber loss")
        ax.set_title("Courbe de perte (DQN)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    print(f"courbes sauvegardees -> {save_path}")
    plt.show()


def train(n_episodes: int = 4000, n_boids: int = 15, seed: int = 42,
          resume: str | None = None):

    env   = BoidFollowerEnv(n_boids=n_boids, seed=seed)
    agent = DQNAgent(
        obs_dim    = env.obs_dim,
        n_actions  = env.n_actions,
        eps_decay  = n_episodes // 2,
    )
    if resume:
        agent.load(resume)
        print(f"reprise depuis {resume}")

    window      = deque(maxlen=100)
    best_avg    = -np.inf
    all_rewards: list[float] = []
    avg100:      list[float] = []

    for ep in range(1, n_episodes + 1):
        obs       = env.reset()
        ep_reward = 0.0
        in_flock  = 0

        while True:
            action            = agent.act(obs)
            next_obs, r, done = env.step(action)
            agent.push(obs, action, r, next_obs, done)
            obs               = next_obs
            ep_reward        += r

            cen  = env.sim.centroid()
            dist = float(np.linalg.norm(env.agent_pos - cen))
            if dist < 3.0:
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
            print(
                f"ep {ep:5d}/{n_episodes}  "
                f"avg_r={avg:7.2f}  "
                f"in_flock={in_flock:2d}/50  "
                f"eps={agent.eps:.3f}  "
                f"loss={loss_avg:.4f}"
            )
            if avg > best_avg:
                best_avg = avg
                agent.save("dqn_best.pt")

    agent.save("dqn.pt")
    print("modeles sauvegardes -> dqn.pt  /  dqn_best.pt")

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
