#!/usr/bin/python

import sys
from EP import EP
import pickle

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python run_training.py popsize max_gen num_games nsigma")
    print(sys.argv)
    params = sys.argv[1:]
    popsize= int(params[0])
    max_gen = int(params[1])
    num_games = int(params[2])
    nsigma = int(params[3]) == 1

    print("Tamanho da população = ", popsize)
    print("Número de gerações = ", max_gen)
    print("Numero de jogos do torneio = ", num_games)
    if nsigma:
        print("Utilizando N sigmas para mutação")
    else:
        print("Utilizando 1 sigma para mutação")
    ep = EP(nsigma=nsigma, popsize=popsize)
    ep.generateInitialPopulation()

    solution_branca, solution_preta, deltaTime = ep.minimize(max_gen=max_gen, num_games=num_games)

    rede_branca = {'time': deltaTime, 'sr': ep.sr1, 'k': -ep.k_1, 'num_hidden_layers': 1, 'hidden_size':128, 'weights': solution_branca}
    rede_preta = {'time': deltaTime, 'sr': ep.sr2, 'k': ep.k_2, 'num_hidden_layers': 1, 'hidden_size':128, 'weights': solution_preta}

    pickle.dump( rede_branca, open( "modelos/rede_branca_popsize_{0}_maxgen_{1}_numgames_{2}_nsigma_{3}.p".format(popsize, max_gen, num_games, nsigma), "wb" ) )
    pickle.dump( rede_preta, open( "modelos/rede_preta_popsize_{0}_maxgen_{1}_numgames_{2}_nsigma_{3}.p".format(popsize, max_gen, num_games, nsigma), "wb" ) )