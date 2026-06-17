"""
Simulation physique 3D de boids — pas discret avec dt.
Max 20 oiseaux, boîte toroïdale.
"""
import numpy as np


class BoidSim3D:
    def __init__(
        self,
        n_boids: int = 15,
        box: float = 20.0,
        dt: float = 0.1,
        max_speed: float = 3.0,
        sep_r: float = 1.5,
        ali_r: float = 4.0,
        coh_r: float = 8.0,
        w_sep: float = 2.0,
        w_ali: float = 1.0,
        w_coh: float = 0.8,
        seed: int = 0,
    ):
        assert 1 <= n_boids <= 20, "max 20 boids"
        self.n = n_boids
        self.box = box
        self.dt = dt
        self.max_speed = max_speed
        self.sep_r = sep_r
        self.ali_r = ali_r
        self.coh_r = coh_r
        self.w_sep = w_sep
        self.w_ali = w_ali
        self.w_coh = w_coh
        self.rng = np.random.default_rng(seed)
        self.pos = np.empty((n_boids, 3))
        self.vel = np.empty((n_boids, 3))
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, rng: np.random.Generator | None = None) -> None:
        rng = rng or self.rng
        # spawn groupés au centre ± bruit
        center = np.full(3, self.box / 2)
        self.pos = center + rng.uniform(-self.box * 0.2, self.box * 0.2, (self.n, 3))
        self.pos = np.clip(self.pos, 0, self.box)

        dirs = self._rand_unit(self.n, rng)
        self.vel = dirs * (self.max_speed * 0.6)

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Un pas discret : calcule les forces puis intègre (Euler)."""
        accel = np.zeros((self.n, 3))

        for i in range(self.n):
            diff = self._toroidal_diff(i)          # (n-1, 3)
            dist = np.linalg.norm(diff, axis=1)    # (n-1,)

            sep_mask = dist < self.sep_r
            ali_mask = (dist >= self.sep_r) & (dist < self.ali_r)
            coh_mask = (dist >= self.sep_r) & (dist < self.coh_r)

            # séparation : repousse
            if sep_mask.any():
                d = dist[sep_mask, None]
                accel[i] += self.w_sep * (-diff[sep_mask] / (d + 1e-6)).sum(0)

            # alignement : moyenne des vitesses voisines
            others_vel = np.delete(self.vel, i, axis=0)
            if ali_mask.any():
                target = others_vel[ali_mask].mean(0)
                accel[i] += self.w_ali * (target - self.vel[i])

            # cohésion : tire vers le centre local
            others_pos = np.delete(self.pos, i, axis=0)
            if coh_mask.any():
                center = others_pos[coh_mask].mean(0)
                # toroidal centroid correction
                delta = center - self.pos[i]
                delta = ((delta + self.box / 2) % self.box) - self.box / 2
                accel[i] += self.w_coh * delta

        # intégration Euler
        self.vel = self.vel + accel * self.dt
        # clamp vitesse
        speeds = np.linalg.norm(self.vel, axis=1, keepdims=True)
        over = speeds > self.max_speed
        self.vel = np.where(over, self.vel / speeds * self.max_speed, self.vel)
        # vitesse minimale pour éviter l'arrêt complet
        min_spd = self.max_speed * 0.2
        under = (speeds < min_spd) & (speeds > 1e-8)
        self.vel = np.where(under, self.vel / (speeds + 1e-8) * min_spd, self.vel)

        self.pos = (self.pos + self.vel * self.dt) % self.box

    # ------------------------------------------------------------------
    def centroid(self) -> np.ndarray:
        return self.pos.mean(axis=0)

    def avg_vel(self) -> np.ndarray:
        return self.vel.mean(axis=0)

    # ------------------------------------------------------------------
    def _toroidal_diff(self, i: int) -> np.ndarray:
        """Vecteurs de boid i vers tous les autres (distance toroïdale)."""
        others = np.delete(self.pos, i, axis=0)
        d = others - self.pos[i]
        d = ((d + self.box / 2) % self.box) - self.box / 2
        return d

    @staticmethod
    def _rand_unit(n: int, rng: np.random.Generator) -> np.ndarray:
        v = rng.standard_normal((n, 3))
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
