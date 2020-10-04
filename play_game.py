#!/usr/bin/python

import sys
from CheckersGameRandom import CheckersGameRandom
from CheckersGameManual import CheckersGameManual
from CheckersGame import CheckersGame
import pandas as pd

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print("Usage: python play_game.py manual num_matches nsigma")

    params = sys.argv[1:]
    manual= int(params[0]) == 1
    num_matches= int(params[1])

    path_rede_branca_1sigma = "modelos/rede_branca_popsize_15_maxgen_500_numgames_5_nsigma_False.p"
    path_rede_branca_nsigma = "modelos/rede_branca_popsize_15_maxgen_500_numgames_5_nsigma_True.p"

    path_rede_preta_1sigma = "modelos/rede_preta_popsize_15_maxgen_500_numgames_5_nsigma_False.p"
    path_rede_preta_nsigma = "modelos/rede_preta_popsize_15_maxgen_500_numgames_5_nsigma_True.p"
    
    if manual:
        nsigma = int(params[2]) == 1
        if nsigma:
            game = CheckersGameManual(path_rede_branca_nsigma, path_rede_preta_nsigma)
        else:
            game = CheckersGameManual(path_rede_branca_1sigma, path_rede_preta_1sigma)
    else:
        num_execucoes = 10


        brancoXAleatorioSingleSigma = CheckersGameRandom(path_rede_branca_1sigma, path_rede_preta_1sigma)
        brancoXAleatorioNSigma = CheckersGameRandom(path_rede_branca_nsigma, path_rede_preta_nsigma)
        print("1 sigma")
        brancoXAleatorioSingleSigma.startGame(num_matches, 1, num_execucoes)
        print("N sigmas")
        brancoXAleatorioNSigma.startGame(num_matches, 1, num_execucoes)

        pretoXAleatorioSingleSigma = CheckersGameRandom(path_rede_branca_1sigma, path_rede_preta_1sigma)
        pretoXAleatorioNSigma = CheckersGameRandom(path_rede_branca_nsigma, path_rede_preta_nsigma)
        print("1 sigma")
        pretoXAleatorioSingleSigma.startGame(num_matches, 2, num_execucoes)
        print("N sigmas")
        pretoXAleatorioNSigma.startGame(num_matches, 2, num_execucoes)

        brancoSingleSigmaXPretoNSigma = CheckersGame(path_rede_branca_1sigma, path_rede_preta_nsigma)
        pretoSingleSigmaXBrancoNSigma = CheckersGame(path_rede_branca_nsigma, path_rede_preta_1sigma)
        print("1 sigma x N sigmas")
        brancoSingleSigmaXPretoNSigma.startGame(num_matches, num_execucoes)
        print("N sigmas x 1 sigma")
        pretoSingleSigmaXBrancoNSigma.startGame(num_matches, num_execucoes)

        
        tempo_ajuste = [brancoXAleatorioSingleSigma.train_time/3600, brancoXAleatorioNSigma.train_time/3600, pretoXAleatorioSingleSigma.train_time/3600, pretoXAleatorioNSigma.train_time/3600, brancoSingleSigmaXPretoNSigma.train_time/3600, pretoSingleSigmaXBrancoNSigma.train_time/3600]
        modelos = ["1 sigma x aleatório", "n sigmas x aleatório", "aleatório x 1 sigma", "aleatório x n sigmas", "1 sigma x N sigmas", "N sigmas x 1 sigma"]

        results_dict = {"Modelo": modelos, "Tempo de treino (em horas)": tempo_ajuste}
        # results_dict = {"Modelo": modelos, "SR Peças Brancas": sr_branca, "SR Peças Pretas": sr_preta, "Taxa de Empates" : empate, "Tempo de treino (em horas)": tempo_ajuste}
        for e in range(num_execucoes):
            sr_branca = [brancoXAleatorioSingleSigma.sr_branca[e], brancoXAleatorioNSigma.sr_branca[e], pretoXAleatorioSingleSigma.sr_branca[e], pretoXAleatorioNSigma.sr_branca[e], brancoSingleSigmaXPretoNSigma.sr_branca[e], pretoSingleSigmaXBrancoNSigma.sr_branca[e]]
            results_dict["SR Peças Brancas Execução {0}".format(e)] = sr_branca

        for e in range(num_execucoes):
            sr_preta = [brancoXAleatorioSingleSigma.sr_preta[e], brancoXAleatorioNSigma.sr_preta[e], pretoXAleatorioSingleSigma.sr_preta[e], pretoXAleatorioNSigma.sr_preta[e], brancoSingleSigmaXPretoNSigma.sr_preta[e], pretoSingleSigmaXBrancoNSigma.sr_preta[e]]
            results_dict["SR Peças Pretas Execução {0}".format(e)] = sr_preta
        
        for e in range(num_execucoes):
            empate = [brancoXAleatorioSingleSigma.draw[e], brancoXAleatorioNSigma.draw[e], pretoXAleatorioSingleSigma.draw[e], pretoXAleatorioNSigma.draw[e], brancoSingleSigmaXPretoNSigma.draw[e], pretoSingleSigmaXBrancoNSigma.draw[e]]
            results_dict["Taxa de Empates Execução {0}".format(e)] = empate

        df = pd.DataFrame.from_dict(results_dict)
        df.to_csv("resultados/resultados.csv")