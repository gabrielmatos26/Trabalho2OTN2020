import numpy as np
import pickle
from checkers.game import Game, Board
import copy
from ANN import ANN

class CheckersGameManual:
    def __init__(self, path_rede_branca, path_rede_preta):
        self.rede_branca = pickle.load(open( path_rede_branca, "rb"))
        self.rede_preta = pickle.load(open( path_rede_preta, "rb"))
        self.game = Game()
        self.game.consecutive_noncapture_move_limit = 30
        self.players = {1:-1, 2:1}
        self.playersPrint = {1:'B', 2:'P'}
        
        index = np.arange(1, 33)
        self.positions_map = {}
        for i, j in zip(np.arange(1,5),np.arange(1, 8, 2)):
            self.positions_map[i] = [0,j]
            self.positions_map[i+8] = [2,j]
            self.positions_map[i+16] = [4,j]
            self.positions_map[i+24] = [6,j]

        for i, j in zip(np.arange(5,9),np.arange(0, 8, 2)):
            self.positions_map[i] = [1,j]
            self.positions_map[i+8] = [3,j]
            self.positions_map[i+16] = [5,j]
            self.positions_map[i+24] = [7,j]
    
    def printBoard(self, board_encoded, demo=False):
        board64 = np.zeros((8,8), dtype=object)
        for i, coord in self.positions_map.items():
            if demo:
                board64[coord[0], coord[1]] = i
            else:
                if board_encoded[i-1] == 0:
                    board64[coord[0], coord[1]] = '0'
                else:
                    board64[coord[0], coord[1]] = board_encoded[i-1]

        print(board64)
    
    def createBoard(self, game, k_1, k_2, print=False):
        if print:
            board_encoded = np.zeros(32, dtype=object)
        else:
            board_encoded = np.zeros(32)
        for piece in game.board.pieces:
            if piece.captured:
                continue
            if piece.king:
                if piece.player == 1:
                    if print:
                        board_encoded[piece.position - 1] = '-D'
                    else:
                        board_encoded[piece.position - 1] = k_1
                else:
                    if print:
                        board_encoded[piece.position - 1] = 'D'
                    else:
                        board_encoded[piece.position - 1] = k_2
            else:
                if print:
                    board_encoded[piece.position - 1] = self.playersPrint[piece.player]
                else:
                    board_encoded[piece.position - 1] = self.players[piece.player]
        return board_encoded
    
    def reconstructWeightsMatrix(self, weights_vector, layer, hidden_size):
        if layer == 0:
            weights_matrix = weights_vector.reshape((33, hidden_size))
        else: #ultima
            weights_matrix = weights_vector.reshape((hidden_size+1, 1))
        return weights_matrix
    
    def generateNextStates(self, k_1, k_2):
        estados_tabuleiro = {}
        for move in self.game.get_possible_moves():
            game_copy = copy.copy(self.game)
            game_copy.move(move)
            board_encoded = self.createBoard(game_copy, k_1, k_2)
            estados_tabuleiro[tuple(move)] = board_encoded
        return estados_tabuleiro
    
    def play(self, k_1, k_2, rede_branca, rede_preta):
        ann_1 = ANN(rede_branca[2], rede_branca[0], rede_branca[1])
        ann_2 = ANN(rede_preta[2], rede_preta[0], rede_preta[1])
        outputs = []
        states = self.generateNextStates(k_1, k_2)
        if self.game.whose_turn() == 1:
            for k, v in states.items():
                outputs.append([k, ann_1.predict(v)])
        else:
            for k, v in states.items():
                outputs.append([k, ann_2.predict(v)])
        outputs = sorted(outputs, key=lambda x: x[-1], reverse=True)
        move = list(outputs[0][0])
        print("\nJogada escolhida: ", move, '\n')
        self.game.move(move)
        
    def selectMove(self):
        print("Escolha um dos possíveis movimentos: ")
        moves = self.game.get_possible_moves()
        for i, move in enumerate(moves):
            print(i+1, ': ', move)
        while 1:
            try:
                selected = int(input("Movimento escolhido: "))
                if selected < 0:
                    return -1
                elif selected < 0 or selected > len(moves):
                    print("Escolha inválida")
                else:
                    return moves[selected-1]
            except: 
                print("Escolha inválida")
    
    def startGame(self):
        self.game = Game()
        while 1:
            try:
                player = int(input("Deseja jogar com as peças brancas (1) ou pretas (2): "))
                if player == 1 or player == 2:
                    self.player = player
                    break
                else:
                    print("Escolha inválida")
            except: 
                print("Escolha inválida")
        if self.player == 2:
            k1 = self.rede_branca['k']
            weights = self.rede_branca['weights']
            num_hidden_layers = self.rede_branca['num_hidden_layers']
            hidden_size = self.rede_branca['hidden_size']
        else:
            k1 = self.rede_preta['k']
            weights = self.rede_preta['weights']
            num_hidden_layers = self.rede_preta['num_hidden_layers']
            hidden_size = self.rede_preta['hidden_size']
        k2 = -1 * k1
        
        limit = 33*hidden_size
        weights = [self.reconstructWeightsMatrix(weights[:limit], 0, hidden_size),
             self.reconstructWeightsMatrix(weights[limit:-1], 1, hidden_size)]#temporario pra contornar o sigma
        rede = [num_hidden_layers, hidden_size, weights]
        
        print("\nAs posições são as seguintes: \n")
        self.printBoard(board_encoded=None, demo=True)
        
        print("\nJogador 1 começa\n")
        
        while not self.game.is_over():
            board_encoded = self.createBoard(self.game, k1, k2, print=True)
            self.printBoard(board_encoded)
            
            if self.game.whose_turn() != self.player:
                self.play(k1, k2, rede, rede)
            else:
                selected = self.selectMove()
                if not isinstance(selected, list) and selected < 0:
                    break
                self.game.move(selected)
            
            print("Vez do jogador ", self.game.whose_turn())
        
        if not isinstance(selected, list) and selected < 0:
            print("Player ", self.player, " desistiu. Jogador ", 3-self.player, " venceu.")
        else:
            print("Player ", self.game.get_winner(), " venceu")
            
            
            