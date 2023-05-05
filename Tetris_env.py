import numpy as np
import random
import math
import time
import copy
import cv2
from PIL import Image, ImageDraw, ImageFont
import time


class Tetrino:
    # Each Tetrinos' coordinates within the smallest mxm array possible (2x2, 3x3, or 4x4)
    TETRINOS = {  # Shape, color, num_unique_rotations, BGR
        1: [np.array([[1, 1], [1, 1]]), [0, 255, 255], 1],  # O
        2: [np.array([[2, 2, 2, 2]]), [255, 255, 0], 2],  # I
        3: [np.array([[3, 3, 0], [0, 3, 3]]), [0, 0, 255], 2],  # Z
        4: [np.array([[0, 4, 4], [4, 4, 0]]), [0, 255, 0], 2],  # S
        5: [np.array([[0, 0, 5], [5, 5, 5]]), [0, 127, 255], 4],  # L
        6: [np.array([[6, 0, 0], [6, 6, 6]]), [255, 0, 0], 4],  # J
        7: [np.array([[0, 7, 0], [7, 7, 7]]), [255, 0, 255], 4]  # T
    }

    def __init__(self, id):
        self.rotation = 0  # 0 - 3
        self.ref_point = np.array([0, 3])  # Coordinates of the pieces' top left corner
        self.id = id  # 0-6 int which represents the pieces shape in TETRINOS
        self.shape, self.color, self.num_unique_rotations = self.TETRINOS.get(id, "No such shape exists")
        self.dim = [self.shape.shape[0], self.shape.shape[1]]

    def reset(self):  # return piece to default position
        self.rotation = 0
        self.ref_point = [0, 3]

    def move(self, direction: [int, int]):  # Move the piece by shifting its reference point
        self.ref_point = np.add(self.ref_point, direction)

    def rotate_right(self):
        self.shape = np.flipud(self.shape).T  # Rotate right 90
        self.dim = self.dim[::-1]  # Invert the dimension lengths
        self.rotation += 1
        if self.rotation > self.num_unique_rotations:
            self.rotation = 0


class Tetris:
    BOARD_DIMS = (24, 10)  # 20 rows + 4 for spawning the tile, 10 columns

    def __init__(self):
        self.score = 0
        self.level = 0
        self.lines_cleared = 0
        self.playing = True
        self.can_store = True
        self.bag = list(range(1, 8))
        self.game_board = np.zeros(shape=self.BOARD_DIMS, dtype=int)
        self.active_piece = self._get_next_piece()
        self.next_piece = self._get_next_piece()

    def reset(self):
        self.score = 0
        self.level = 0
        self.lines_cleared = 0
        self.playing = True
        self.can_store = True
        self.bag = list(range(1, 8))
        self.game_board = np.zeros(shape=self.BOARD_DIMS, dtype=int)
        self.active_piece = self._get_next_piece()
        self.next_piece = self._get_next_piece()

    def _get_next_piece(self):
        index = random.randint(0, len(self.bag)-1)
        piece_id = self.bag.pop(index)
        piece = Tetrino(piece_id)
        if len(self.bag) == 0:
            self.bag = list(range(1, 8))
        return piece

    def print_board(self):
        print(self.game_board)

    def _is_legal_move(self, board, piece: Tetrino, move: [int, int]):
        new_ref_point = piece.ref_point + move
        for y in range(piece.dim[0]):
            for x in range(piece.dim[1]):
                board_y = new_ref_point[0] + y
                board_x = new_ref_point[1] + x
                try:
                    if board[board_y][board_x] != 0 and piece.shape[y][x] != 0:
                        return False
                except IndexError:
                    return False
        return True

    # Add the piece to the board
    def _place_piece(self, board, piece):  # Stores the piece after it is ready to be locked in place
        for y in range(piece.dim[0]):
            for x in range(piece.dim[1]):
                board_y = piece.ref_point[0] + y
                board_x = piece.ref_point[1] + x
                board[board_y][board_x] += piece.shape[y][x]

    def _clear_full_rows(self, board):
        # Clear full rows
        for row in range(24):
            if 0 not in board[row]:
                for i in range(row - 1):  # Shift all rows above the removed one down
                    board[row - i] = board[row - i - 1]

    def _check_game_end(self, board):
        return True if not np.all(
            (board[:4] == 0)) else False  # If all 0 in the top four rows, game is not over

    def _get_lines_cleared(self, board):
        new_lines_cleared = 0
        for row in range(24):
            if 0 not in board[row]:
                new_lines_cleared += 1
        return new_lines_cleared

    def _get_holes(self, board):
        num_holes = 0
        for col in board.T:
            roof = False
            for tile in col:
                if roof and tile == 0:
                    num_holes += 1
                elif tile != 0:
                    roof = True
        return num_holes

    def _get_bumps(self, board):
        bumpiness = 0
        heights = []
        for col in board.T:
            col_height = 24
            for tile in col:
                if tile == 0 and col_height == 1:
                    heights.append(0)
                elif tile == 0:
                    col_height -= 1
                else:
                    heights.append(col_height)
                    break
        for height in range(len(heights) - 1):
            bumpiness += abs(heights[height] - heights[height + 1])
        return bumpiness

    def _get_heights(self, board):
        heights = []
        for col in board.T:
            col_height = 24
            for tile in col:
                if tile == 0 and col_height == 1:
                    heights.append(0)
                elif tile == 0:
                    col_height -= 1
                else:
                    heights.append(col_height)
                    break
        return sum(heights), max(heights), min(heights)

    def get_next_states(self, board):
        piece = Tetrino(self.active_piece.id)
        next_boards = []
        next_states = {}
        for rotation in range(piece.num_unique_rotations):
            for i in range(0, 11 - piece.dim[1]):
                piece.ref_point = [0, i]
                temp_board = copy.deepcopy(board)
                while self._is_legal_move(temp_board, piece, [1, 0]):
                    piece.move([1, 0])

                self._place_piece(temp_board, piece)
                rows_cleared = self._get_lines_cleared(temp_board)
                self._clear_full_rows(temp_board)
                holes = self._get_holes(temp_board)
                bumpiness = self._get_bumps(temp_board)
                sum_height, max_height, min_height = self._get_heights(temp_board)
                next_piece = self.next_piece.id
                game_over = self._check_game_end(temp_board)
                placement_height = piece.ref_point[0]

                next_boards.append(temp_board)
                next_state = [rows_cleared,
                              holes,
                              bumpiness,
                              sum_height]
                """
                              max_height,
                              min_height,
                              next_piece,
                              placement_height,
                              game_over]"""

                action = (piece.ref_point[1], piece.rotation)  # top left corner x pos, rotation
                next_states[action] = next_state
            piece.rotate_right()
        return next_states

    def execute_action(self, action: (int, int), show):
        self.active_piece.ref_point = [0, 3]
        x_pos = action[0]
        rotation = action[1]
        score_per_line = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}
        # start = time.process_time()  # Set show=False in Train.py to test piece movement speed
        # Fastest gravity drops a tetrino every frame, ran at 60 fps.  1/60 = 0.0167 seconds
        while x_pos != self.active_piece.ref_point[1]:
            if x_pos < self.active_piece.ref_point[1]:
                self.active_piece.move([0, -1])
            else:
                self.active_piece.move([0, 1])
            if show:
                self.show()
        # print(time.process_time() - start)  # Used to test if the model can outpace the quickest gravity level
        for _ in range(rotation):
            self.active_piece.rotate_right()
            if show:
                self.show()
        while self._is_legal_move(self.game_board, self.active_piece, [1, 0]):
            self.active_piece.move([1, 0])
            """if show:
                self.show()"""
        self.show()

        self._place_piece(self.game_board, self.active_piece)
        lines_cleared = self._get_lines_cleared(self.game_board)
        self.lines_cleared += lines_cleared
        reward = 1 + 10 * (lines_cleared ** 2)
        self.level = self.lines_cleared // 10
        self.score += score_per_line.get(lines_cleared) * (self.level + 1) + self.active_piece.ref_point[0]
        self._clear_full_rows(self.game_board)
        self.active_piece = self.next_piece
        self.next_piece = self._get_next_piece()
        if self._check_game_end(self.game_board):
            return -5, True
        return reward, False

    def show(self):
        empty_tile_color = (255, 210, 210)  # BGR
        spawn_zone_color = (255, 190, 190)
        img = np.zeros(shape=(26, 23, 3), dtype=np.uint8)

        # Draw the boards tiles
        for y in range(self.BOARD_DIMS[0]):
            for x in range(self.BOARD_DIMS[1]):
                if self.game_board[y][x] != 0:
                    img[y + 1][x + 1] = Tetrino(self.game_board[y][x]).color
                else:
                    if y < 4:
                        img[y + 1][x + 1] = spawn_zone_color
                    else:
                        img[y + 1][x + 1] = empty_tile_color

        # Draw the active piece
        for y in range(self.active_piece.dim[0]):
            for x in range(self.active_piece.dim[1]):
                if self.active_piece.shape[y][x] != 0:
                    img[y + 1 + self.active_piece.ref_point[0]][x + 1 + self.active_piece.ref_point[1]] = Tetrino(
                        self.active_piece.shape[y][x]).color
                else:
                    if self.active_piece.ref_point[0] < 4:
                        img[y + 1][x + 1] = spawn_zone_color
                    else:
                        img[y + 1][x + 1] = spawn_zone_color

        # Draw next piece
        for y in range(1, 6):
            for x in range(12, 18):
                img[y][x] = spawn_zone_color
        for y in range(self.next_piece.dim[0]):
            for x in range(self.next_piece.dim[1]):
                if self.next_piece.shape[y][x] != 0:
                    img[y + 3][x + 13] = self.next_piece.color

        # Turn img from an array to an image
        img_upscale = 40  # block width in pixels
        title_color = (217, 45, 243)
        img = Image.fromarray(img, "RGB")
        img = img.resize((img.size[0] * img_upscale, img.size[1] * img_upscale), resample=Image.BOX)
        myFont = ImageFont.truetype('OpenSans-VariableFont_wdth,wght.ttf', img_upscale-2)

        # Draw Next Piece Title
        draw = ImageDraw.Draw(img)
        draw.text((img_upscale * 12, img_upscale), "Next Piece", font=myFont, fill=title_color)

        # Draw Score section
        draw.rectangle((img_upscale * 12, img_upscale * 7, img_upscale * 18, img_upscale * 11), fill=spawn_zone_color)
        draw.text((img_upscale * 12, img_upscale * 7), "Score", font=myFont, fill=title_color)
        draw.text((img_upscale * 12, img_upscale * 8), str(self.score), font=myFont, fill=title_color)

        # Draw Rows Cleared section
        draw.rectangle((img_upscale * 12, img_upscale * 12, img_upscale * 18, img_upscale * 16), fill=spawn_zone_color)
        draw.text((img_upscale * 12, img_upscale * 12), "Rows Cleared", font=myFont, fill=title_color)
        draw.text((img_upscale * 12, img_upscale * 13), str(self.lines_cleared), font=myFont, fill=title_color)

        # Draw Level section
        draw.rectangle((img_upscale * 12, img_upscale * 17, img_upscale * 18, img_upscale * 21), fill=spawn_zone_color)
        draw.text((img_upscale * 12, img_upscale * 17), "Level", font=myFont, fill=title_color)
        draw.text((img_upscale * 12, img_upscale * 18), str(self.level), font=myFont, fill=title_color)

        # Convert the image into an upscaled array
        img = np.array(img)

        # Show the game state as an image
        cv2.imshow("board", img)
        cv2.waitKey(1)
