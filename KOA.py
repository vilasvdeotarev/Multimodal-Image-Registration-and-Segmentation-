import numpy as np
import time


# Kookaburra Optimization Algorithm (KOA)
def KOA(X, fitness_function, lb, ub, max_iterations):
    [N, D] = X.shape

    def hunting_phase(X, F):
        CP = [k for k, Fi in enumerate(F) if Fi < F[0]]
        if CP:
            k = np.random.choice(CP)
            SCPi = X[k] - X
            r = np.random.rand()
            X_P1 = X + r * SCPi
            # mask = F[X_P1[0] < F[0]]
            # X[mask] = X_P1[mask]
        return X

    def killing_phase(X, r, t):
        X_P2 = X + (1 - 2 * r) * (ub - lb) * t
        mask = fitness_function(X_P2) < fitness_function(X)
        X[mask] = X_P2[mask]
        return X

    # Initialization
    F = np.apply_along_axis(fitness_function, 1, X)
    convergence = np.zeros(max_iterations)
    best_solution = X[np.argmin(F)]
    ct = time.time()
    # Main loop
    for t in range(max_iterations):
        for i in range(N):
            # Phase 1: Hunting Strategy (Exploration)
            X = hunting_phase(X, F)

            # Phase 2: Ensuring that the prey is killed (Exploitation)
            r = np.random.rand()
            X = killing_phase(X, r, t)

        # Save the best candidate solution so far
        current_best = X[np.argmin(F)]
        if fitness_function(current_best) < fitness_function(best_solution):
            best_solution = current_best
        convergence[t] = np.min(best_solution)
    best_score = fitness_function(best_solution)
    ct = time.time() - ct
    return best_score, convergence, best_solution, ct

