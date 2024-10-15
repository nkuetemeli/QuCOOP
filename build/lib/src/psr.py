import numpy as np
import random
import matplotlib.pyplot as plt
import dimod
import dwave.inspector
from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite, LazyFixedEmbeddingComposite
import random
import neal
from . import utils
import logging

from sympy.physics.paulialgebra import delta

np.set_printoptions(linewidth=100)


class PointSetRegistration:
    def __init__(self, ref_points, tmp_points, R_star, P_star):
        self.ref_points = ref_points
        self.tmp_points = tmp_points
        self.R_star = R_star
        self.P_star = P_star
        self.d, self.n = ref_points.shape
        self.p = int(self.d * (self.d - 1) / 2)
        self.T = utils.t(self.n)
        self.n_trans = self.T.shape[1]
        self.P_sol = np.eye(self.n)
        self.R_sol = np.eye(self.d)


    def objective(self, R, P):
        return np.linalg.norm(R @ self.tmp_points - self.ref_points @ P, ord='fro') ** 2

    def solve(self, K, maxiter=15, alpha_rot=.1, simulated_anneal=True):
        eval_r = utils.eval_r_2d if self.d == 2 else utils.eval_r_3d

        scale = 2**K - 1
        delta = np.pi * np.ones((self.p, 1))
        tau = 4 * delta / scale

        x0_rot = 0.0 * np.ones((self.p,)).reshape(self.p, 1)
        x0_perm = np.zeros(self.n_trans).reshape(self.n_trans, 1)

        P_sol = np.eye(self.n, dtype='float64')
        R_sol = np.eye(self.d, dtype='float64')
        sol = self.objective(R_sol, P_sol)
        history = [sol,]

        X = np.kron(self.ref_points, P_sol)
        Y = np.kron(R_sol, self.tmp_points)

        # Select solver and sampler
        if simulated_anneal:
            sampler = neal.SimulatedAnnealingSampler()           # Simulated annealing
        else:
            qpu_advantage = DWaveSampler(solver={'topology__type': 'pegasus'})
            sampler = LazyFixedEmbeddingComposite(qpu_advantage)  # Quantum annealing

        alpha_rot *= np.linalg.norm(self.tmp_points, ord='fro') ** 2
        alpha_perm = np.linalg.norm(self.ref_points, ord='fro') ** 2
        for j in range(maxiter):
            Rx0, dRx0 = eval_r(x0_rot)
            Rc = Rx0 - dRx0 @ delta

            Px0, dPx0 = utils.eval_p(self.T, x0_perm)
            Pc = Px0 - dPx0 @ x0_perm

            U = (2 / scale) * np.kron(np.diag(delta.flatten()), np.array([2**k for k in range(K)]))

            x_dPx0 = np.einsum('di, ij -> dj', X, dPx0)
            y_dRx0 = np.einsum('di, dj -> ij', Y, dRx0)

            Dij_rot = U.T @ np.einsum('di, dj -> ij', dRx0, dRx0) @ U * alpha_rot
            Dii_rot = 2 * np.einsum('di, d -> i', dRx0, Rc.flatten()) @ U * alpha_rot

            Dij_perm = np.einsum('dj, dk -> jk', dPx0, dPx0) * alpha_perm
            Dii_perm = 2 * np.einsum('dk, d -> k', dPx0, Pc.flatten()) * alpha_perm

            Dij_mixed = -U.T @ np.einsum('di, dj -> ij', y_dRx0, x_dPx0) / 2
            Dii_mixed = -np.block([np.einsum('di, d -> i', y_dRx0, X @ Pc.flatten()) @ U,
                                   np.einsum('dk, d -> k', x_dPx0, Rc.flatten() @ Y)])

            Dij = np.block([[Dij_rot, Dij_mixed], [Dij_mixed.T, Dij_perm]])
            Dii = np.block([Dii_rot, Dii_perm]) + Dii_mixed

            W = {(i, j): Dij[i, j] for i in range(self.p * K + self.n_trans) for j in range(self.p * K + self.n_trans)}
            c = {i: Dii[i] for i in range(self.p * K + self.n_trans)}
            bqm = dimod.BinaryQuadraticModel(c, W, vartype=dimod.BINARY)

            num_reads = 100 if simulated_anneal else 50
            response = sampler.sample(bqm, num_reads=num_reads)
            # dwave.inspector.show(response)  # If quantum annealing and if needed

            best = response.first.sample
            q = np.array(list(best.values()))
            q_rot = q[:K * self.p].reshape(K * self.p, 1)
            q_perm = q[K * self.p:].reshape(self.n_trans, 1)

            x_new_tmp = U @ q_rot

            R_sol = (Rc + dRx0 @ x_new_tmp).reshape(self.d, self.d)
            P_sol = (Pc + dPx0 @ q_perm).reshape(self.n, self.n)
            sol = self.objective(R_sol, P_sol)
            history.append(sol)

            x0_rot_new = x0_rot - delta + x_new_tmp
            # Decrease search interval
            if (j + 1) % 2 == 0:
                delta = delta/2
                tau = tau/2

            x0_rot = x0_rot_new
            x0_perm = q_perm

            logging.info(f"Iter {j}, \t"
                      f"delta = {np.round(delta.flatten(), 8)}, \t"
                      f"error_R = {round(np.linalg.norm(R_sol - self.R_star), 8) if self.R_star is not None else '-'}, \t"
                      f"error_P = {round(np.linalg.norm(P_sol - self.P_star) ** 2 / 2) if self.P_star is not None else '-' }, \t"
                      f"objective = {round(sol, 8)}")

            self.R_sol = R_sol
            self.P_sol = P_sol
        return history


def example_2d():
    # Create random problem
    seed = 1
    np.random.seed(seed)

    n = 10

    ref_points = utils.get_random_2d_points(n=n, scale=.5)
    tmp_points = ref_points.copy()

    theta = np.deg2rad(np.random.randint(10, 80))
    R = utils.r_2d(theta)
    R_star = utils.r_2d(-theta)

    P = np.eye(n)
    np.random.shuffle(P)
    P_star = P.T

    tmp_points = R @ tmp_points
    ref_points = ref_points @ P
    ref_shape_original = np.array(utils.get_bezier_curve(ref_points.T, rad=0.2, edgy=0.05)[:2])
    tmp_shape_original = np.array(utils.get_bezier_curve(tmp_points.T, rad=0.2, edgy=0.05)[:2])

    ref_points -= np.mean(ref_points, axis=1, keepdims=True)
    tmp_points -= np.mean(tmp_points, axis=1, keepdims=True)
    ref_shape_original -= np.mean(ref_shape_original, axis=1, keepdims=True)
    tmp_shape_original -= np.mean(tmp_shape_original, axis=1, keepdims=True)

    colors = list(map(lambda i: "#" + "%06x" % np.random.randint(0, 0xFFFFFF), range(n)))

    # Set seed back to None
    np.random.seed(None)

    # Start solver
    psr = PointSetRegistration(ref_points=ref_points, tmp_points=tmp_points, R_star=R_star, P_star=P_star)
    history = psr.solve(K=5, maxiter=15, alpha_rot=.1, simulated_anneal=True)  # Adjust alpha_rot

    # Plots
    # Initial
    ref_points_plot = ref_points
    tmp_points_plot = tmp_points
    plt.Polygon((ref_points @ P.T).T, closed=True, fill='w', edgecolor='k')
    plt.Polygon(tmp_points_plot.T, closed=True, fill='w', edgecolor='k')
    plt.plot(*ref_shape_original)
    plt.plot(*tmp_shape_original)
    for i, c in zip(list(range(n)), colors):
        plt.scatter(*tmp_points_plot[:, i], c=c, marker='o')
        plt.scatter(*ref_points_plot[:, i], c=c, marker='+', s=100)
    plt.axis('equal')
    plt.show()

    # Result
    ref_points_plot = ref_points @ psr.P_sol
    tmp_points_plot = psr.R_sol @ tmp_points
    plt.Polygon((ref_points @ P.T).T, closed=True, fill='w', edgecolor='k')
    plt.Polygon(tmp_points_plot.T, closed=True, fill='w', edgecolor='k')
    plt.plot(*ref_shape_original)
    plt.plot(*psr.R_sol @ tmp_shape_original)
    for i, c in zip(list(range(n)), colors):
        plt.scatter(*ref_points_plot[:, i], c=c, marker='o')
        plt.scatter(*tmp_points_plot[:, i], c=c, marker='+', s=100)
    plt.axis('equal')
    plt.show()


def example_3d():
    fig = plt.figure()
    ax = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
    size = 7

    # Generate random problem
    seed = 1
    n = 20
    np.random.seed(seed)
    ref_points = utils.get_random_3d_points('spiral', n)
    tmp_points = ref_points.copy()

    ref_points -= np.mean(ref_points, axis=1, keepdims=True)
    tmp_points -= np.mean(tmp_points, axis=1, keepdims=True)

    theta = np.deg2rad(np.random.randint(-10, 45))
    v = np.random.randn(3, 1)
    R = utils.r_3d(theta * v)[0]
    R_star = R.T

    P = np.eye(n)
    np.random.shuffle(P)
    P_star = P.T

    tmp_points = R @ tmp_points
    ref_points = ref_points @ P

    colors = list(map(lambda i: "#" + "%06x" % np.random.randint(0, 0xFFFFFF), range(n)))

    # Set seed back to None
    np.random.seed(None)

    # Start solver
    psr = PointSetRegistration(ref_points=ref_points, tmp_points=tmp_points, R_star=R_star, P_star=P_star)
    history = psr.solve(K=5, maxiter=15, alpha_rot=.08, simulated_anneal=True)  # Adjust alpha_rot

    # Plots
    # Initial
    tmp_color, ref_color = 'b', 'y'
    ref_points_plot = ref_points
    tmp_points_plot = tmp_points
    for i, c in zip(list(range(n)), colors):
        try:
            ax[0].scatter(*tmp_points_plot[:, i], color=tmp_color, marker='s', s=size)
            ax[0].scatter(*ref_points_plot[:, i], color=ref_color, marker='s', s=size)
            ax[0].plot([ref_points_plot[0, i], tmp_points_plot[0, i]],
                       [ref_points_plot[1, i], tmp_points_plot[1, i]],
                       [ref_points_plot[2, i], tmp_points_plot[2, i]], c='gray', linewidth=.5, alpha=.5)
        except:
            ax[0].scatter(*ref_points_plot[:, i], color=ref_color, marker='s', s=size)
    ax[0].axis('equal')
    ax[0].axis(False)

    # Ours
    ref_points_plot = ref_points @ psr.P_sol
    tmp_points_plot = psr.R_sol @ tmp_points
    for i, c in zip(list(range(n)), colors):
        try:
            ax[1].scatter(*tmp_points_plot[:, i], color=tmp_color, marker='s', s=size)
            ax[1].scatter(*ref_points_plot[:, i], color=ref_color, marker='s', s=size)
            ax[1].plot([ref_points_plot[0, i], tmp_points_plot[0, i]],
                       [ref_points_plot[1, i], tmp_points_plot[1, i]],
                       [ref_points_plot[2, i], tmp_points_plot[2, i]], c='gray', linewidth=.5, alpha=.5)
        except:
            ax[1].scatter(*ref_points_plot[:, i], color=ref_color, marker='s', s=size)
    ax[1].set_title(None)
    ax[1].axis('equal')
    ax[1].axis(False)

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # example_2d()
    example_3d()
