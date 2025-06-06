import time
import numpy as np


# Zebra Optimization Algorithm (ZOA)
def ZOA(X, fitness, lowerbound, upperbound, Max_iterations):
    SearchAgents, dimension = X.shape
    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        fit[i] = fitness(X[i, :])

    best_so_far = np.zeros(Max_iterations)
    average = np.zeros(Max_iterations)

    ct = time.time()
    for t in range(Max_iterations):
        best_idx = np.argmin(fit)
        if t == 0:
            PZ = X[best_idx, :]  # Optimal location
            fbest = fit[best_idx]  # Best fitness value
        elif fit[best_idx] < fbest:
            fbest = fit[best_idx]
            PZ = X[best_idx, :]

        # Phase 1: Foraging Behaviour
        for i in range(SearchAgents):
            I = np.round(1 + np.random.rand())
            X_newP1 = X[i, :] + np.random.rand(dimension) * (PZ - I * X[i, :])
            X_newP1 = np.maximum(X_newP1, lowerbound[i])
            X_newP1 = np.minimum(X_newP1, upperbound[i])

            # Update X_i if new position improves fitness
            f_newP1 = fitness(X_newP1)
            if f_newP1 <= fit[i]:
                X[i, :] = X_newP1
                fit[i] = f_newP1

        # Phase 2: Defense strategies against predators
        Ps = np.random.rand()
        k = np.random.randint(SearchAgents)
        AZ = X[k, :]  # Attacked zebra

        for i in range(SearchAgents):
            if Ps < 0.5:
                # S1: Lion attacks the zebra (escape strategy)
                R = 0.1
                X_newP2 = X[i, :] + R * (2 * np.random.rand(dimension) - 1) * (1 - t / Max_iterations) * X[i, :]
            else:
                # S2: Other predators attack (offensive strategy)
                I = np.round(1 + np.random.rand())
                X_newP2 = X[i, :] + np.random.rand(dimension) * (AZ - I * X[i, :])

            X_newP2 = np.maximum(X_newP2, lowerbound[i, :])
            X_newP2 = np.minimum(X_newP2, upperbound[i, :])

            # Update X_i if new position improves fitness
            f_newP2 = fitness(X_newP2)
            if f_newP2 <= fit[i]:
                X[i, :] = X_newP2
                fit[i] = f_newP2

        # Store best_so_far and average values
        best_so_far[t] = fbest
        average[t] = np.mean(fit)

    Best_score = fbest
    Best_pos = PZ
    ZOA_curve = best_so_far
    ct = time.time() - ct
    return Best_score, ZOA_curve, Best_pos, ct

