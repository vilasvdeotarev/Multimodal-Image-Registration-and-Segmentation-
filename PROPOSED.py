import numpy as np
import time


# Lotus Effect Optimization (LEO)
def PROPOSED(population, objective_function, lower_bound, upper_bound, max_iter):
    population_size, num_dimensions = population.shape
    lower_bound = lower_bound[0,:]
    upper_bound = upper_bound[0,:]
    fit = np.zeros(population_size)
    for n in range(population_size):
        fit[n] = objective_function(population[n])

    convergence = np.zeros(max_iter)
    best_solution = None
    best_fitness = float("inf")
    ct = time.time()

    for epoch in range(max_iter):
        for idx in range(population_size):
            fitness = objective_function(population[idx])

            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = population[idx]

            fit[idx] = objective_function(population[idx])

            # Phase 1: Spreading the pollen for exploitation
            r = np.random.rand()
            rr = r * (1 - epoch / max_iter) * population[idx]
            pos_new = population[idx] + (2 * np.random.rand() - 1) * rr
            pos_new = np.clip(pos_new, lower_bound, upper_bound)

            new_fitness = objective_function(pos_new)
            if new_fitness < fitness:
                population[idx] = pos_new
                fitness = new_fitness
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = pos_new

            # Lotus Effect Optimization (LEO) Phase 2: Recovering from the disturbance for exploration
            kk = np.random.choice([i for i in range(population_size) if i != idx])
            if objective_function(population[kk]) < fitness:
                pos_new = population[idx] + np.random.rand() * (population[kk] - np.random.randint(1, 4) * population[idx])
            else:
                pos_new = population[idx] + np.random.rand() * (population[idx] - population[kk])

            pos_new = np.clip(pos_new, lower_bound, upper_bound)
            new_fitness = objective_function(pos_new)
            if new_fitness < fitness:
                population[idx] = pos_new
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_solution = pos_new

        convergence[epoch] = np.min(fit)

    ct = time.time() - ct
    return best_fitness, convergence, best_solution, ct

