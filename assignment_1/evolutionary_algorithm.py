from scipy.optimize import rosen, differential_evolution
import numpy as np
import pandas as pd

# default parameters
METHOD = 'differential_evolution'
BOUNDS = [(-1,1), (-1,1), (-1,1), (-1,1)]
GENERATIONS = 10
STRATEGY = 'rand1bin'
POP_SIZE  =  100

class EvolutionaryAlgorithm:

    def __init__(self, method, obj_function, bounds, strategy, generations):

        self.method = method
        self.obj_function = obj_function
        self.bounds = bounds
        self.strategy = strategy
        self.generations = generations

    def optimize(self):

        if self.method == 'differential_evolution':
            self.ea = differential_evolution(func=self.obj_function, 
                                             bounds=self.bounds, 
                                             strategy=self.strategy,
                                             disp=True,
                                             maxiter=self.generations)
        return self.ea


def start_evoman():

    hidden_neurons = 10
    weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    pop = np.random.uniform(BOUNDS[0], BOUNDS[1], (POP_SIZE, weights))


    #initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=[2],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest")




def initialize_ea():
    return

    
    

if __name__ == '__main__':

    ea = EvolutionaryAlgorithm(METHOD, rosen, BOUNDS, STRATEGY, GENERATIONS)
    optimize = ea.optimize()
    print(optimize.fun)
    

    

    # printing optimalization with 10 generations
    #print(ea.optimize())

    # printing optimalization with 5 generations 
    #ea.generations = 5
    #print(ea.optimize())