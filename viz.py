"""
Visualisation 3D — matplotlib.

Usage :
    python viz.py                        # agent aleatoire
    python viz.py --policy dqn_best.pt   # politique apprise
    python viz.py --policy dqn_best.pt --episodes 3
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from env import BoidFollowerEnv
from agent import DQNAgent


def run_episode(env, agent):
    obs = env.reset()
    frames, total_r = [], 0.0
    while True:
        frames.append({
            "boids": env.sim.pos.copy(),
            "agent": env.agent_pos.copy(),
            "cen":   env.sim.centroid().copy(),
            "t":     env.t,
        })
        action = agent.act(obs, greedy=True) if agent else env.n_actions // 2
        obs, r, done = env.step(action)
        frames[-1]["r"] = r
        total_r += r
        if done:
            break
    return frames, total_r


def animate(frames, box, ep_n):
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, box); ax.set_ylim(0, box); ax.set_zlim(0, box)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    sc_b = ax.scatter([], [], [], c="#1855a5", s=18, alpha=0.65, label="boids")
    sc_c = ax.scatter([], [], [], c="#1855a5", s=55, marker="x", linewidths=2)
    sc_a = ax.scatter([], [], [], c="#993c1d", s=90, marker="^", label="agent")
    trail_line, = ax.plot([], [], [], "r-", alpha=0.35, lw=0.9)
    title = ax.set_title("")
    ax.legend(loc="upper left", fontsize=8)

    hist = []

    def update(i):
        f = frames[i]
        bp, ap, cen = f["boids"], f["agent"], f["cen"]
        hist.append(ap.copy())

        sc_b._offsets3d = (bp[:, 0], bp[:, 1], bp[:, 2])
        sc_c._offsets3d = ([cen[0]], [cen[1]], [cen[2]])
        sc_a._offsets3d = ([ap[0]], [ap[1]], [ap[2]])

        if len(hist) > 1:
            h = np.array(hist)
            trail_line.set_data(h[:, 0], h[:, 1])
            trail_line.set_3d_properties(h[:, 2])

        dist = np.linalg.norm(ap - cen)
        title.set_text(
            f"ep {ep_n+1}  t={f['t']:02d}/50  "
            f"r={f.get('r', 0):+.2f}  dist={dist:.2f}"
        )
        return sc_b, sc_c, sc_a, trail_line, title

    return animation.FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",   type=str, default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--boids",    type=int, default=15)
    args = parser.parse_args()

    env   = BoidFollowerEnv(n_boids=args.boids)
    agent = None
    if args.policy:
        agent = DQNAgent(env.obs_dim, env.n_actions)
        agent.load(args.policy)
        print(f"politique chargee : {args.policy}  (eps={agent.eps:.3f})")

    for ep in range(args.episodes):
        frames, total_r = run_episode(env, agent)
        print(f"episode {ep+1}: reward total = {total_r:.2f}")
        ani = animate(frames, env.box, ep)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
