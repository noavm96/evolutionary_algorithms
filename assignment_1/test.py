import pandas as pd
import numpy as np

# settings 
TRAIN_TEST = 'train'
NUM_GENERATIONS = 10
MIN_BOUNDS = -1 
MAX_BOUNDS = 1
POP_SIZE = 5 
NUM_WEIGHTS = 4 # depends on the amount of neurons 

# hyperparameters
MUTATION_FACTOR = 0.5 # ideally between 0 and 2
RECOMBINATION_FACTOR = 0.5 

class DifferentialEvolution:

    def __init__(self, obj_function, pop_size, min_bounds, max_bounds):
        self.pop_size = pop_size

    def parent_selection():
        return

    def recombination():
        return

    def mutation_operator():
        return

    def survival_selection():
        return

    def evaluation(self, x):
        # function to calculate fitness value for new genes 

        return

    def execute_steps(self, old_parents, old_fitness):
        # select the parents
        parents = self.parent_selection(old_parents, old_fitness)
        # recomine the parents to create the offspring 
        offspring = self.recombination(parents) 
        # mutate the offspring
        offspring = self.mutation_operator(offspring)
        # evaluate the offspring by means of their fitness values
        offspring_fitness = self.evaluation(offspring)
        # select the fittest candidates for the new population
        pop, fitness = self.survival_selection(old_parents, 
                                               offspring, 
                                               old_fitness, 
                                               offspring_fitness)
        return pop, fitness


def train_ea(ea):

    # initialize random population and its fitness values
    pop = np.random.uniform(MIN_BOUNDS, MAX_BOUNDS, size = (POP_SIZE, NUM_WEIGHTS))
    fitness = ea.evaluation(pop)

    # store the population in a list and save the best fitness value
    population = pop.tolist()
    best_fitness = [fitness.max()]

    for i in range(NUM_GENERATIONS):
        
        new_population, new_fitness = ea.execute(population, fitness)
        population.append(new_parents)

       # if new_fitness.max() < f_best[-1]: 
       #     f_best.append(f.max())
       # else: 
       #     f_best.append(f_best[-1])

       print('Generation: %d, best fitness: %d' % (i, new_fitness.max()))

    return # save results in file 


def test_ea():
    # function to test the ea with the best result from the above trained ea 
    return 


if __name__ == '__main__':

    objective_function = x**2
    ea = EvolutionaryAlgorithm(objective_function,
                               POP_SIZE,
                               MIN_BOUNDS,
                               MAX_BOUNDS)
    
    if TRAIN_TEST == 'train':
        train_ea(ea)
    else:
        test_ea(ea)

    


   


