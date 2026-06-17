"""
Environnement RL — épisodes de 50 pas, observation continue 12D, 27 actions discrètes.

Observation (12 floats, normalisée) :
  [0:3]  position relative agent → centroïde swarm   / (box/2)
  [3:6]  vitesse de l'agent                           / max_agent_speed
  [6:9]  vitesse moyenne du swarm                     / max_boid_speed
  [9:12] position relative agent → boid le plus proche / box

Actions discrètes : 27 = {-1, 0, +1}^3 sur l'accélération

Reward :
  - distance normalisée au centroïde
  + alignement cosinus avec le swarm
  + bonus si dans le nuage (dist < seuil)
  - pénalité de mur
"""
import numpy as np
from sim import BoidSim3D

MAX_STEPS   = 50
MAX_AGENT_SPEED = 5.0
ACCEL_MAG   = 3.0
WALL_MARGIN = 1.5   # distance au bord qui déclenche la pénalité
IN_FLOCK_R  = 3.0   # rayon pour bonus "dans le nuage"

ACTIONS = np.array(
    [(ax, ay, az) for ax in (-1, 0, 1) for ay in (-1, 0, 1) for az in (-1, 0, 1)],
    dtype=np.float32,
)
N_ACTIONS = len(ACTIONS)   # 27
OBS_DIM   = 12


class BoidFollowerEnv:
    def __init__(self, n_boids: int = 15, box: float = 20.0, seed: int = 0):
        assert 1 <= n_boids <= 20
        self.sim  = BoidSim3D(n_boids=n_boids, box=box, seed=seed)
        self.box  = box
        self.rng  = np.random.default_rng(seed)
        self.n_actions = N_ACTIONS
        self.obs_dim   = OBS_DIM
        self._t = 0
        self.agent_pos = np.zeros(3, np.float32)
        self.agent_vel = np.zeros(3, np.float32)

        # perturbation périodique du swarm
        self._perturb_every = 12   # tous les N pas un coup de vent sur le swarm

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.sim.reset(self.rng)
        offset = self.rng.uniform(-self.box * 0.3, self.box * 0.3, 3).astype(np.float32)
        self.agent_pos = np.clip(self.sim.centroid() + offset, 0, self.box).astype(np.float32)
        self.agent_vel = (BoidSim3D._rand_unit(1, self.rng)[0] * MAX_AGENT_SPEED * 0.4).astype(np.float32)
        self._t = 0
        return self._obs()

    # ------------------------------------------------------------------
    def step(self, action: int):
        assert 0 <= action < N_ACTIONS
        accel = ACTIONS[action] * ACCEL_MAG
        dt = self.sim.dt

        # mise à jour agent
        self.agent_vel = self.agent_vel + accel * dt
        spd = float(np.linalg.norm(self.agent_vel))
        if spd > MAX_AGENT_SPEED:
            self.agent_vel = (self.agent_vel / spd * MAX_AGENT_SPEED).astype(np.float32)

        self.agent_pos = np.clip(self.agent_pos + self.agent_vel * dt, 0, self.box).astype(np.float32)

        # rebond sur les murs (vitesse inversée si hors boîte)
        for ax in range(3):
            if self.agent_pos[ax] <= 0:
                self.agent_pos[ax] = 0.0
                self.agent_vel[ax] = abs(self.agent_vel[ax])
            elif self.agent_pos[ax] >= self.box:
                self.agent_pos[ax] = float(self.box)
                self.agent_vel[ax] = -abs(self.agent_vel[ax])

        # avance le swarm
        self.sim.step()
        self._t += 1

        # perturbation périodique du swarm
        if self._t % self._perturb_every == 0:
            self._perturb_swarm()

        reward = self._reward()
        done = self._t >= MAX_STEPS
        return self._obs(), reward, done

    # ------------------------------------------------------------------
    def _perturb_swarm(self):
        """Coup de vent aléatoire sur tous les boids → changement de direction collectif."""
        impulse = self.rng.standard_normal(3).astype(np.float32)
        impulse = impulse / (np.linalg.norm(impulse) + 1e-8) * self.sim.max_speed * 0.6
        self.sim.vel += impulse

    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        cen     = self.sim.centroid().astype(np.float32)
        avg_vel = self.sim.avg_vel().astype(np.float32)

        rel_cen  = (self.agent_pos - cen) / (self.box / 2)

        # boid le plus proche (toroïdal)
        diffs = self.sim.pos - self.agent_pos
        diffs = ((diffs + self.box / 2) % self.box) - self.box / 2
        dists = np.linalg.norm(diffs, axis=1)
        nearest = diffs[np.argmin(dists)] / self.box

        obs = np.concatenate([
            rel_cen,
            self.agent_vel / MAX_AGENT_SPEED,
            avg_vel / self.sim.max_speed,
            nearest.astype(np.float32),
        ], dtype=np.float32)
        return np.clip(obs, -3.0, 3.0)

    def _reward(self) -> float:
        cen = self.sim.centroid()
        rel = self.agent_pos - cen
        dist = float(np.linalg.norm(rel))

        r_dist = -dist / (self.box / 2)

        # alignement
        avg_v = self.sim.avg_vel()
        v_n = np.linalg.norm(self.agent_vel)
        s_n = np.linalg.norm(avg_v)
        if v_n > 1e-6 and s_n > 1e-6:
            align = float(np.dot(self.agent_vel, avg_v) / (v_n * s_n))
        else:
            align = 0.0

        # bonus si dans le nuage
        in_flock = 1.0 if dist < IN_FLOCK_R else 0.0

        # pénalité de mur
        wall_pen = 0.0
        for ax in range(3):
            d_wall = min(float(self.agent_pos[ax]), self.box - float(self.agent_pos[ax]))
            if d_wall < WALL_MARGIN:
                wall_pen -= (WALL_MARGIN - d_wall) / WALL_MARGIN

        return r_dist + 0.4 * align + 0.5 * in_flock + 0.2 * wall_pen

    # ------------------------------------------------------------------
    @property
    def t(self) -> int:
        return self._t
