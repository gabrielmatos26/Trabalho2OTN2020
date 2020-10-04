from checkers.game import Game, Board
from ANN import ANN
import copy
import numpy as np

class Gameplay:
    def __init__(self):
        self.game = Game()
        self.game.consecutive_noncapture_move_limit = 150
        self.players = {1:-1, 2:1}
    
    def createBoard(self, game, k_1, k_2):
        board_encoded = np.zeros(32)
        for piece in game.board.pieces:
            if piece.captured:
                continue
            if piece.king:
                if piece.player == 1:
                    board_encoded[piece.position - 1] = -k_1
                else:
                    board_encoded[piece.position - 1] = k_2
            else:
                board_encoded[piece.position - 1] = self.players[piece.player]
        return board_encoded
    
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
        if self.game.is_over():
            self.game = Game() 
        
        while not self.game.is_over():
            outputs = []
            if self.game.whose_turn() == 1:
                states = self.generateNextStates(k_1, k_1)
                for k, v in states.items():
                    outputs.append([k, ann_1.predict(v)])
            else:
                states = self.generateNextStates(k_2, k_2)
                for k, v in states.items():
                    outputs.append([k, ann_2.predict(v)])
            outputs = sorted(outputs, key=lambda x: x[-1], reverse=True)
            move = list(outputs[0][0])
            self.game.move(move)
#             print(self.createBoard(k_1, k_2))
        return self.game.get_winner()
