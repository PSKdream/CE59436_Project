import numpy as np

class TicTaeToe:
    def __init__(self, player='X'):
        self.player = player
        self.ai = "O" if player == "X" else "X"
        self.board = np.full((3, 3), '_')

    def is_end(self, player, boardInput=None):
        if boardInput is None:
            boardInput = self.board
        board = boardInput == player
        if np.any(np.sum(board, axis=0) == 3): return player
        if np.any(np.sum(board, axis=1) == 3): return player
        if np.sum(np.diag(boardInput == player)) == 3: return player
        if np.sum(np.sum(np.diag(np.fliplr(boardInput)) == player) == 3): return player
        if np.any(boardInput == "_") == False: return None
        return '?'

    def rule(self, playerCheck, playerOpposite, board=None):
        if board is None:
            board = self.board
        r = board
        a = (board == playerCheck)
        if np.any(np.sum(a, axis=0) == 2):
            col = np.where(np.sum(a, axis=0) == 2)[0][0]
            for i in range(len(a)):
                if a[i, col] == False and r[i, col] != playerOpposite:
                    return (i, col)
        if np.any(np.sum(a, axis=1) == 2):
            row = np.where(np.sum(a, axis=1) == 2)[0][0]
            # print(row)
            for i in range(len(a[row])):
                if a[row, i] == False and r[row, i] != playerOpposite:
                    return (row, i)
        if np.sum(np.diag(a)) == 2:
            index = np.diag(a)
            for i in range(len(index)):
                if index[i] == False and r[i, i] != playerOpposite:
                    return (i, i)
        if np.sum(np.diag(np.fliplr(a))) == 2:
            index = np.diag(np.fliplr(a))
            # print(index)
            for i in range(len(index)):
                if (index[i] == False):
                    if i == 2 and r[2, 0] != playerOpposite:
                        return (2, 0)
                    if i == 1 and r[1, 1] != playerOpposite:
                        return (1, 1)
                    if i == 0 and r[0, 2] != playerOpposite:
                        return (0, 2)

    def move(self, loc, board=None):
        if board is None:
            board = self.board
        if board[loc] == '_':
            board[loc] = self.player
            if self.is_end(self.player, board) == '?':
                board[self.bot_move_rd(board)] = self.ai
        else:
            print("error")

    def bot_move_rd(self, board=None):
        if board is None:
            board = self.board
        hint = self.rule(self.ai, self.player, board)
        if hint != None:
            return hint
        hint = self.rule(self.player, self.ai, board)
        if hint != None:
            return hint
        loc = np.where(board == '_')
        i = np.random.choice(len(loc[0]))
        return loc[0][i], loc[1][i]

    def display(self):
        print(self.board)
        # hint1 = self.rule(self.player, self.ai)
        # print(hint1)
        if self.is_end(self.ai) == self.ai:
            print("ai win")
        elif self.is_end(self.player) == self.player:
            print("player win")
        elif self.is_end(self.player) is None:
            print("not win")

    def predict(self):
        board = self.board
        hint = self.rule(self.ai, self.player, board)
        if hint != None:
            return hint
        hint = self.rule(self.player, self.ai, board)
        if hint != None:
            return hint

        Q_score = np.full((3, 3), 0.0)
        loc = np.where(self.board == '_')
        for location in range(len(loc[0])):
            avg = 0
            for count in range(1000):
                board = self.board.copy()
                board[loc[0][location], loc[1][location]] = self.ai
                while True:
                    #player
                    locPlayer = np.where(board == '_')
                    if len(locPlayer[0]) > 1:
                        i = np.random.choice(len(locPlayer[0]))
                    else:
                        i = 0
                    self.move((locPlayer[0][i] , locPlayer[1][i]), board)

                    #check win
                    if self.is_end(self.ai, board) == self.ai:
                        # print("ai win")
                        avg += 1
                        break
                    elif self.is_end(self.player, board) == self.player:
                        # print("player win")
                        break
                    elif self.is_end(self.player, board) is None:
                        # print("not win")
                        break
            Q_score[loc[0][location], loc[1][location]] = avg / 1000.0
        # print(Q_score)
        if np.all(Q_score == 0) == False:
            # print(np.unravel_index(np.argmax(Q_score, axis=None), Q_score.shape))
            return np.unravel_index(np.argmax(Q_score, axis=None), Q_score.shape)
        else:
            random_loc = self.bot_move_rd()
            # print(random_loc)
            return random_loc

    def move_vs_ai(self, loc, board=None):
        if board is None:
            board = self.board
        if board[loc] == '_':
            board[loc] = self.player
            if self.is_end(self.player, board) == '?':
                index = self.predict()
                board[index] = self.ai
                return index
            return None
        else:
            print('Error : position duplicate')
            return 'Error : position duplicate'




# ox = TicTaeToe()
# ox.move_vs_ai((0, 1))
# ox.display()