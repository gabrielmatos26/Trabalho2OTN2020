import numpy as np
import pickle
from checkers.game import Game, Board
import copy
from ANN import ANN
from tqdm import tqdm

class CheckersGame:
    def __init__(self, path_rede_branca, path_rede_preta):
        self.rede_branca = pickle.load(open( path_rede_branca, "rb"))
        self.rede_preta = pickle.load(open( path_rede_preta, "rb"))
        self.train_time = self.rede_branca['time']
        self.game = Game()
        self.game.consecutive_noncapture_move_limit = 300
        # self.game.consecutive_noncapture_move_limit = 40
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

        self.sr_branca = 0
        self.draw = 0
        self.sr_preta = 0
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
        self.game.move(move)
        
    def playRandom(self):
        moves = self.game.get_possible_moves()
        choice = np.random.choice(np.arange(len(moves)))
        self.game.move(moves[choice])
    
    def startGame(self, num_matches, num_executions=1):
        self.game = Game()
        self.sr_branca = 0
        self.draw_branca = 0
        self.sr_preta = 0
        self.draw_preta = 0
        
        k1 = self.rede_branca['k']
        k2 = self.rede_preta['k']


        weights = self.rede_branca['weights']
        num_hidden_layers = self.rede_branca['num_hidden_layers']
        hidden_size = self.rede_branca['hidden_size']
        limit = 33*hidden_size
        weights = [self.reconstructWeightsMatrix(weights[:limit], 0, hidden_size),
            self.reconstructWeightsMatrix(weights[limit:], 1, hidden_size)]
        rede_branca = [num_hidden_layers, hidden_size, weights]

        weights = self.rede_preta['weights']
        num_hidden_layers = self.rede_preta['num_hidden_layers']
        hidden_size = self.rede_preta['hidden_size']
        limit = 33*hidden_size
        weights = [self.reconstructWeightsMatrix(weights[:limit], 0, hidden_size),
            self.reconstructWeightsMatrix(weights[limit:], 1, hidden_size)]
        rede_preta = [num_hidden_layers, hidden_size, weights]


        self.sr_branca = np.zeros(num_executions)
        self.sr_preta = np.zeros(num_executions)
        self.draw = np.zeros(num_executions)
        for e in tqdm(range(num_executions)):
            for i in tqdm(range(num_matches), leave=False):
                while not self.game.is_over():
                    if self.game.whose_turn() == 1:
                        self.play(k1, k1, rede_branca, rede_preta)
                    else:
                        self.play(k2, k2, rede_branca, rede_preta)

                if self.game.get_winner() is not None:
                    if self.game.get_winner() == 1:
                        self.sr_branca[e] += 1/num_matches
                        # self.sr_branca += 1/num_matches
                    else:
                        self.sr_preta[e] += 1/num_matches
                        # self.sr_preta += 1/num_matches
                else:
                    self.draw[e] += 1/num_matches
                    # self.draw += 1/num_matches
                
                self.game = Game()

        # self.sr_branca = sr_branca.mean()
        # self.sr_preta = sr_preta.mean()
        # self.draw = draw.mean()
        
            