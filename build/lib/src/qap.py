import numpy as np
import matplotlib.pyplot as plt
import dimod
import dwave.inspector
from dwave.cloud import Client
from dwave.system import DWaveSampler, EmbeddingComposite, LazyFixedEmbeddingComposite
import neal
from . import utils
import logging

np.set_printoptions(linewidth=100)

class QAP:
    def __init__(self, Q, P_star):
        self.Q = Q
        self.P_star = P_star
        self.n = int(np.sqrt(Q.shape[0]))
        self.T = utils.t(self.n)
        self.n_trans = self.T.shape[1]
        self.P_sol = np.eye(n)

    def objective(self, P):
        P = P.reshape(-1)[:, None]
        return (P.T @ self.Q @ P).squeeze()

    def solve(self, maxiter=15, simulated_anneal=True):
        x0 = np.zeros(self.n_trans).reshape(self.n_trans, 1)

        P_sol = utils.eval_p(self.T, x0)[0].reshape(n, n)
        sol = self.objective(P_sol)
        history = [sol,]

        # Do nothing if Q is zero
        if np.array_equal(Q, np.zeros_like(Q)):
            return P_sol, history

        # Select solver and sampler
        if simulated_anneal:
            sampler = neal.SimulatedAnnealingSampler()           # Simulated annealing
        else:
            qpu_advantage = DWaveSampler(solver={'topology__type': 'pegasus'})
            sampler = LazyFixedEmbeddingComposite(qpu_advantage)  # Quantum annealing

        for j in range(maxiter):
            Px0, dPx0 = utils.eval_p(self.T, x0)
            Pc = Px0 - dPx0 @ x0

            Dij = dPx0.T @ Q @ dPx0
            Dii = (2 * Pc.T @ Q @ dPx0).flatten()

            W = {(i, j): Dij[i, j] for i in range(self.n_trans) for j in range(self.n_trans)}
            c = {i: Dii[i] for i in range(self.n_trans)}
            bqm = dimod.BinaryQuadraticModel(c, W, vartype=dimod.BINARY)

            num_reads = 100 if simulated_anneal else 50
            response = sampler.sample(bqm, num_reads=num_reads)
            # dwave.inspector.show(response)  # If quantum annealing and if needed

            best = response.first.sample
            q = np.array(list(best.values())).reshape(self.n_trans, 1)

            x0 = q
            P_sol = (Pc + dPx0 @ x0).reshape(n, n)
            sol = self.objective(P_sol)
            history.append(sol)

            logging.info(f"\nIter {j}"
                         f"\nPermutation matrix found: \nP = \n{P_sol}"
                         f"\nObjective = {sol}, norm P = {int(np.linalg.norm(P_sol) ** 2)}, "
                         f"# wrong matches = {round(np.linalg.norm(P_sol - self.P_star) ** 2 / 2) if self.P_star is not None else '-'}\n")

            if history[-1] == history[-2]:
                break

        self.P_sol = P_sol
        return history


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Create random problem
    seed = 5
    np.random.seed(seed)
    n = 10
    P = np.eye(n)
    np.random.shuffle(P)
    P_star = P

    ref_points = np.random.random((2, n))
    tmp_points = ref_points @ P_star

    A = np.array([[np.linalg.norm(ref_points[:, i] - ref_points[:, j]) for i in range(n)] for j in range(n)])
    B = np.array([[np.linalg.norm(tmp_points[:, i] - tmp_points[:, j]) for i in range(n)] for j in range(n)])
    Q = -np.kron(A, B)
    alpha = np.min(np.linalg.eigvalsh(Q))
    Q -= alpha * np.eye(n * n)

    # Set seed back to None
    np.random.seed(None)

    # Start Solver
    qap = QAP(Q, P_star)
    history = qap.solve()

    # ResultsFLFalse
    print(f'\nResults:            {(qap.P_sol @ np.arange(n)).astype(int)}, objective: {qap.objective(qap.P_sol)}'
          f'\nGround through:     {(P_star @ np.arange(n)).astype(int)}, objective: {qap.objective(P_star)}.')
    plt.plot(history, label='Objective')
    plt.hlines(qap.objective(P_star), 0, len(history) - 1, linestyles='--', color='r', label='Objective target')
    plt.legend()
    plt.show()
