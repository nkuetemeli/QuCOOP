import numpy as np
from QuCOOP.src import qap, utils
import matplotlib.pyplot as plt
import re
import scipy as sp
from os import listdir, path
from QuCOOP.Qmatch import qmatch
from scipy.optimize import linear_sum_assignment


if __name__ == '__main__':
    problem = '../data/Qaplib/chr12a'
    n = int(re.findall(r'\d+', problem)[0])

    # Parse P_star
    sol_star = np.loadtxt(problem + '.sln', skiprows=1).astype(np.int32)

    # Parse Q matrix
    data = np.loadtxt(problem + '.dat', skiprows=2)
    A = data[:n, :]
    B = data[n:, :]
    if np.array_equal(A.T, A) and not np.array_equal(B.T, B):
        B = ( 1 /2) * (B + B.T)
    if not np.array_equal(A.T, A) and np.array_equal(B.T, B):
        A = ( 1 /2) * (A + A.T)

    Q = np.kron(A, B)
    Q = (1 / 2) * (Q + Q.T)
    P_star = utils.permutation_idx2mat(sol_star - 1)


    # Add regularization
    alpha = np.min(np.linalg.eigvalsh(Q))

    # Our solver
    qap = qap.QAP(Q, alpha, P_star)
    history = qap.solve()

    # Scipy faq
    # print(np.array_equal(B, P_star.T @ A @ P_star))  # Sanity check
    res_faq = sp.optimize.quadratic_assignment(A=A, B=B, method='faq', options={'maximize': False})
    P_scipy_faq = utils.permutation_idx2mat(res_faq.col_ind)
    sol_scipy_faq = (P_scipy_faq @ np.arange(n)).astype(int)

    # Scipy 2opt
    res_2opt = sp.optimize.quadratic_assignment(A=A, B=B, method='2opt', options={'maximize': False})
    P_scipy_2opt = utils.permutation_idx2mat(res_2opt.col_ind)
    sol_scipy_2opt = (P_scipy_2opt @ np.arange(n)).astype(int)

    # Q-Match
    num_iter = 10
    sweeps, num_reads = 500, 1000
    Permutations = []
    for i in range(num_iter):
        if i == 0:
            newArrangement = qmatch.optimize(sweeps, num_reads, Q, None, None, i, None)
        else:
            newArrangement = qmatch.optimize(sweeps, num_reads, Q, newArrangement, None, i, None)
        Permutations.append(newArrangement)
    P_qmatch = utils.permutation_idx2mat(np.array(Permutations[-1]))
    sol_qmatch = (P_qmatch @ np.arange(n)).astype(int)

    # ResultsFLFalse
    plt.plot(history, label='Objective')
    plt.hlines(qap.objective(P_star), 0, len(history) - 1, linestyles='--', color='r', label='Objective target')
    plt.legend()
    plt.show()

    print(f'\nScipy FAQ:          {(qap.P_sol @ np.arange(n)).astype(int)}, objective: {qap.objective(P_scipy_faq)}'
          f'\nScipy 2-OPT:        {(qap.P_sol @ np.arange(n)).astype(int)}, objective: {qap.objective(P_scipy_2opt)}'
          f'\nQ-Match:            {(qap.P_sol @ np.arange(n)).astype(int)}, objective: {qap.objective(P_qmatch)}'
          f'\nQuCOOP:             {(qap.P_sol @ np.arange(n)).astype(int)}, objective: {qap.objective(qap.P_sol)}'
          f'\nGround through:     {(P_star @ np.arange(n)).astype(int)}, objective: {qap.objective(P_star)}.')
