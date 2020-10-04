from Gameplay import Gameplay
import numpy as np
from tqdm import tqdm
from time import time

class EP:
    def __init__(self, nsigma=False, popsize=15, num_hidden_layers=1, hidden_size=128):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.popsize = popsize
        self.epsilon0 = 0.001
        self.dimension = 33*hidden_size + hidden_size + 1 
        self.tau = 1/np.sqrt(self.dimension)
        self.sr1 = 0
        self.sr2 = 0
        self.game = Gameplay()
        self.fitness_values1 = np.zeros(2*popsize)
        self.fitness_values2 = np.zeros(2*popsize)
        self.k_1 = 2.0
        self.k_2 = 2.0
        self.nsigma = nsigma
        self.tau = 1/np.sqrt(self.dimension)
        
    def reconstructWeightsMatrix(self, weights_vector, layer):
        if layer == 0:
            weights_matrix = weights_vector.reshape((33, self.hidden_size))
        else: #ultima
            weights_matrix = weights_vector.reshape((self.hidden_size+1, 1))
        return weights_matrix
            
        
    def fitness(self, i_1, p_1, i_2, p_2):
        limit = 33*self.hidden_size
        w_1 = [self.reconstructWeightsMatrix(p_1[:limit], 0), self.reconstructWeightsMatrix(p_1[limit:], 1)]
        w_2 = [self.reconstructWeightsMatrix(p_2[:limit], 0), self.reconstructWeightsMatrix(p_2[limit:], 1)]
        param1 = [self.num_hidden_layers, self.hidden_size, w_1]
        param2 = [self.num_hidden_layers, self.hidden_size, w_2]
        winner = self.game.play(self.k_1, self.k_2, param1, param2)
        if winner == 1:
            self.fitness_values1[i_1] += 1
            self.fitness_values2[i_2] -= 2
            self.sr1 += 1
        elif winner == 2:
            self.fitness_values1[i_1] -= 2
            self.fitness_values2[i_2] += 1
            self.sr2 += 1
            
    def generateWeights(self):
        layer1_weights = np.random.uniform(-0.2, 0.2, size=(33, self.hidden_size))
        layer2_weights = np.random.uniform(-0.2, 0.2, size=(self.hidden_size+1, 1))
        return layer1_weights, layer2_weights
        
    def generateInitialPopulation(self):
        #Gera populacao para peças brancas
        if self.nsigma:
            self.population_1 = np.zeros((self.popsize, self.dimension*2))
        else:
            self.population_1 = np.zeros((self.popsize, self.dimension+1))
        for i in range(self.popsize):
            layer1_weights, layer2_weights = self.generateWeights()
            individual = np.concatenate((layer1_weights.flatten(), layer2_weights.flatten()))
            
            if self.nsigma:
                individual = np.concatenate((individual, np.random.uniform(-0.1, 0.1, self.dimension)))
            else:
                individual = np.concatenate((individual, np.random.uniform(-0.1, 0.1, 1)))
            self.population_1[i,:] = individual
            
        #Gera populacao para peças pretas
        if self.nsigma:
            self.population_2 = np.zeros((self.popsize, self.dimension*2))
        else:
            self.population_2 = np.zeros((self.popsize, self.dimension+1))
        for i in range(self.popsize):
            layer1_weights, layer2_weights = self.generateWeights()
            individual = np.concatenate((layer1_weights.flatten(), layer2_weights.flatten()))

            if self.nsigma:
                individual = np.concatenate((individual, np.random.uniform(-0.1, 0.1, self.dimension)))
            else:
                individual = np.concatenate((individual, np.random.uniform(-0.1, 0.1, 1)))
            self.population_2[i,:] = individual
    
    def mutation(self, x, alpha):
        x_mutated = x.copy()
        if self.nsigma:
            # sigma = x_mutated[self.dimension:] * (1 + alpha*np.random.normal(0, 1, self.dimension))
            sigma = x_mutated[self.dimension:] * np.exp(self.tau*np.random.normal(0, 1, self.dimension))
            sigma[sigma < self.epsilon0] = self.epsilon0
            x_mutated[:self.dimension] += sigma * np.random.normal(0, 1, self.dimension)
            x_mutated[self.dimension:] = sigma
        else:
            # sigma = x_mutated[-1] * (1 + alpha*np.random.normal(0, 1))
            sigma = x_mutated[-1] * np.exp(self.tau*np.random.normal(0, 1))
            if sigma < self.epsilon0:
                sigma = self.epsilon0
            x_mutated[:-1] += sigma * np.random.normal(0, 1, self.dimension)
            x_mutated[-1] = sigma
        return x_mutated
    
    def nextGen(self, parents1, parents2, num_games=5, alpha=0.5):
        
        children1 = []
        children2 = []
        while len(children1) < self.popsize and len(children2) < self.popsize:
            selection1 = np.random.choice(np.arange(len(parents1)), 1)
            selection2 = np.random.choice(np.arange(len(parents2)), 1)
            parent1 = parents1[selection1[0]]
            child1 = self.mutation(parent1, alpha)
            parent2 = parents2[selection2[0]]
            child2 = self.mutation(parent2, alpha)
            
            children1.append(child1)
            children2.append(child2)
        
        self.k_1 += np.random.uniform(-0.1, 0.1)
        self.k_2 += np.random.uniform(-0.1, 0.1)

        self.k_1 = min(max(self.k_1, 1), 3)
        self.k_2 = min(max(self.k_2, 1), 3)
        
        candidates1 = np.vstack([parents1, np.asarray(children1)])
        candidates2 = np.vstack([parents2, np.asarray(children2)])
        
        shuffle1 = np.repeat(np.arange(2*self.popsize).reshape(-1,1), num_games)
        shuffle2 = np.repeat(np.arange(2*self.popsize).reshape(-1,1), num_games)
        np.random.shuffle(shuffle1); np.random.shuffle(shuffle2)
        matches = np.hstack((shuffle1.reshape(-1,1), shuffle2.reshape(-1,1)))
        for match in tqdm(matches, desc='Playing games', leave=False):
            self.fitness(match[0], candidates1[match[0], :self.dimension], match[1], candidates2[match[1], :self.dimension])
            
        self.fitness_values1[::-1].sort()
        self.fitness_values2[::-1].sort()
        print('\nPecas brancas: ', self.fitness_values1)
        print('Pecas pretas: ', self.fitness_values2)

        self.population_1 = candidates1[:self.popsize]
        self.population_2 = candidates2[:self.popsize]
        
        
        self.fitness_values1 = np.zeros(2*self.popsize)
        self.fitness_values2 = np.zeros(2*self.popsize)
    
    def minimize(self, max_gen=10, num_games=5, alpha=0.5):
        t1 = time()
        self.generateInitialPopulation()
       
        for gen in tqdm(range(max_gen)):
            print(40*'-')
            print("Gen ", gen+1)
            print(40*'-')
            self.nextGen(self.population_1, self.population_2, num_games, alpha)
        t2 = time()
        deltaTime = t2-t1
        
        self.solution1 = self.population_1[0, :self.dimension]
        self.solution2 = self.population_2[0, :self.dimension]
        
        self.sr1 = self.sr1 / (num_games*self.popsize*2*max_gen)
        self.sr2 = self.sr2 / (num_games*self.popsize*2*max_gen)
        
        print("SR do modelo para peças brancas: {0:3f}".format(np.round(self.sr1,3)))
        print("SR do modelo para peças pretas: {0:3f}".format(np.round(self.sr2,3)))
        
        return self.solution1, self.solution2, deltaTime