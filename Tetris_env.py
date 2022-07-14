import numpy as np
import pygame

import random
import math
import time
import copy

"""
Tetrino class is responsible for storing the pieces shape, and moving the piece through translation or rotations
"""


class Tetrino:
    # Each Tetrino's coordinates within the smallest mxm array possible (2x2, 3x3, or 4x4)
    TETRINOS = {  # [[y, x]], Color
        0: [np.array([[1, 1], [1, 1]]), [255, 255, 0]],  # O
        1: [np.array([[0, 0, 1], [1, 1, 1], [0, 0, 0]]), [255, 127, 0]],  # L
        2: [np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]]), [0, 0, 255]],  # J
        3: [np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]]), [0, 255, 0]],  # S
        4: [np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]]), [255, 0, 0]],  # Z
        5: [np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]), [0, 255, 255]],  # I
        6: [np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]), [225, 0, 225]]  # T
    }

    def __init__(self):
        self.shape = None  # Initial shape is blank
        self.ref_point = np.array([0, 0], dtype=int)  # All reference points are the cords of the top left of its mxm array
        self.dim = [0, 0]  # Dimensions of the shapes current orientation
        self.rotation = 0  # 0, 1, 2, 3 for how many times it has rotated
        self.color = [0, 0, 0]  # Default color
        self.shape_id = -1  # 0-6 int which represents the pieces shape in TETRINOS

    def set_shape(self, shape_id: int):  # Initialize the properties of the piece
        self.shape_id = shape_id
        self.shape, self.color = Tetrino.TETRINOS.get(shape_id, "No such shape exists")
        self.ref_point = np.array([0, 3])  # (y, x)

        if self.shape.shape is not None:  # If the shape is not blank, set the current dimensions
            height = 0
            width = 0
            for i in range(len(self.shape[0])):
                if np.any(self.shape[i] == 1):
                    height += 1
                if np.any(self.shape.T[i] == 1):
                    width += 1
            self.dim = [height, width]

    def move(self, move: [int, int]):  # Move the piece by shifting its reference point
        self.ref_point += move

    def rotate_right(self, translation=[0,0]):
        self.shape = np.flipud(self.shape).T  # Rotate
        self.dim = self.dim[::-1]  # Adjust dimensions
        self.move(translation) # wall kick


class Tetris:
    BOARD_HEIGHT = 900
    BOARD_WIDTH = 600
    TILE_SIZE = 30  # Make the tile size based on the smaller axis (height or width)
    BOARD_DIMS = (24, 10)  # 20 rows + 4 for spawning the tile, 10 columns

    def __init__(self):
        self.score = 0  # Game score
        self.level = 0  # Current level
        self.last_drop = 0  # Time of the last piece drop
        self.lines_cleared = 0  # Total lines cleared
        self.gravity_timer = 0  # Current time between piece drops
        self.current_piece = None  # The current active piece
        self.next_piece = None  # -_o
        self.playing = False  # True while game is being played
        self.game_board = None  # Board tile data
        self.window = None  # Pygame display to blit visuals to
        self.bag = []  # Empty bag

    def render(self):
        pygame.display.init()
        self.playing = True
        self.last_drop = time.time()
        self.gravity_timer = 3  # Seconds
        self.game_board = np.zeros(shape=self.BOARD_DIMS, dtype=int)  # 20 rows + 4 for the piece to spawn in
        self.window = pygame.display.set_mode((self.BOARD_WIDTH, self.BOARD_HEIGHT), 0, 32)
        self.reset_bag()  # Bag of pieces to randomly pick from

    def draw_graphics(self):
        # Setup pygame text
        pygame.font.init()
        font = pygame.font.SysFont("OCR A Extended", 50, bold=True)
        font2 = pygame.font.SysFont("OCR A Extended", 30, bold=False)
        font3 = pygame.font.SysFont("OCR A Extended", 15, bold=False)

        # Draw graphics outside the game_board
        pygame.draw.rect(self.window, (45, 45, 45), (400, 0, 10, 640))
        pygame.draw.rect(self.window, (30, 30, 30), (410, 0, 480, 640))

        # Draw sound icon
        pygame.draw.rect(self.window, (255, 255, 255), (420, 617.5, 10, 15))
        pygame.draw.rect(self.window, (255, 255, 255), (434, 620, 2, 10))
        pygame.draw.rect(self.window, (255, 255, 255), (438, 618, 2, 13))
        pygame.draw.rect(self.window, (255, 255, 255), (442, 617.5, 2, 15))
        volume_surface = font3.render('(M)', True, (255, 255, 255))
        self.window.blit(volume_surface, (450, 615))

        # Draw mute icon
        sound_off = False
        if sound_off:
            pygame.draw.rect(self.window, (153, 51, 51), (418, 623, 29, 5))

        # Draw "T E T R A CUBE" text
        TETRO_surface = font.render('Tetris', True, (255, 255, 255))
        self.window.blit(TETRO_surface, (420, 10))
        """        
        CUBE_surface = font.render('CUBE', True, (255, 255, 255))
        board.window.blit(CUBE_surface, (26.666666666666668, 0))
        """

        # Draw "restart" text
        restart_surface = font2.render('RESTART(R)', True, (255, 255, 255))
        self.window.blit(restart_surface, (417, 550))

        # Draw "score" text and "score" number
        score_surface = font2.render('SCORE', True, (255, 255, 255))
        self.window.blit(score_surface, (460, 200))
        self.window.blit(font.render(str(round(self.score)), 1, (255, 255, 255)), (420, 230))

        pygame.display.update()

    def print_board(self):
        print(self.game_board)

    def get_board(self):
        return self.game_board

    # The board is made up of tiles. This function returns a given tile and its correct color
    def get_tile(self, color=[135, 135, 135], border_color='black'):
        tile = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE))
        tile.fill(color=border_color)  # border
        pygame.draw.rect(tile, color=color, rect=(1, 1, self.TILE_SIZE - 2, self.TILE_SIZE - 2))  # tile color
        return tile

    # Draw board, the active piece, the stored piece, and shadow
    def draw_board(self):
        next_board_surface = pygame.Surface(
            (self.TILE_SIZE * 10, self.TILE_SIZE * 24))  # Blank board to update, then replace the current board
        next_board_surface.fill(color=[55, 55, 55])
        blank_tile = self.get_tile()

        spawn_surface = pygame.Surface((10 * self.TILE_SIZE, 4 * self.TILE_SIZE), pygame.SRCALPHA, 32)
        spawn_surface.fill(color="blue")
        spawn_surface = spawn_surface.convert_alpha()
        pygame.draw.rect(spawn_surface, color='black', rect=(0, 4 * self.TILE_SIZE - 3, 10 * self.TILE_SIZE, 3))
        next_board_surface.blit(spawn_surface, (0, 0))

        # Draw the whole board
        for y in range(24):
            for x in range(10):
                if self.game_board[y][x] != 0:
                    next_board_surface.blit(blank_tile, (x * self.TILE_SIZE, y * self.TILE_SIZE))

        # Prep to draw the active piece
        piece_tile = self.get_tile(self.current_piece.color)
        piece_surface = pygame.Surface((4 * self.TILE_SIZE, 4 * self.TILE_SIZE), pygame.SRCALPHA, 32)
        piece_surface = piece_surface.convert_alpha()

        # Prep to draw active pieces shadow
        shadow = copy.deepcopy(self.current_piece)
        shadow_tile = self.get_tile([100, 100, 100])
        shadow_surface = pygame.Surface((self.TILE_SIZE * 4, self.TILE_SIZE * 4), pygame.SRCALPHA, 32)
        while self.is_legal_move(shadow, [1, 0]):
            shadow.move([1, 0])

        # Draw active piece and shadow
        for y in range(len(self.current_piece.shape[0])):  # For each row
            for x in range(len(self.current_piece.shape[1])):  # For each column
                if self.current_piece.shape[y][x] != 0:
                    piece_surface.blit(piece_tile, (x * self.TILE_SIZE, y * self.TILE_SIZE))
                    shadow_surface.blit(shadow_tile, (x * self.TILE_SIZE, y * self.TILE_SIZE))

        # Push changes to the screen
        next_board_surface.blit(shadow_surface,
                                (self.TILE_SIZE * shadow.ref_point[1], self.TILE_SIZE * shadow.ref_point[0]))
        next_board_surface.blit(piece_surface,
                                (self.TILE_SIZE * self.current_piece.ref_point[1], self.TILE_SIZE * self.current_piece.ref_point[0]))
        # Draw score and the rest of the visual add-ons
        self.draw_graphics()
        self.window.blit(next_board_surface, (50, 50))
        pygame.display.update()

    def update_board(self):  # Stores the piece after it is ready to be locked in place
        for y in range(len(self.current_piece.shape[0])):  # For each row
            for x in range(len(self.current_piece.shape[1])):  # For each column
                if self.current_piece.shape[y][x] != 0:
                    self.game_board[y + self.current_piece.ref_point[0]][x + self.current_piece.ref_point[1]] += self.current_piece.shape[y][x]
        pygame.display.update()

    def is_legal_move(self, piece: Tetrino, move: [int, int]):  # Create a copy of the piece, move it, and check if it was legal
        new_ref_point = piece.ref_point + move
        if new_ref_point[1] not in range(0 + piece.dim[1] - len(piece.shape[1]), 11 - piece.dim[1]):
            return False
        if new_ref_point[0] not in range(25 - piece.dim[0]):
            return False
        for y in range(len(piece.shape[0])):  # For each row
            for x in range(len(piece.shape[1])):  # For each column
                if piece.shape[y][x] == 1:  # if the checked tile is part of the active pieces shape
                    vert_pos = y + new_ref_point[0]  # Get each piece tiles location
                    hor_pos = x + new_ref_point[1]
                    if vert_pos not in range(24) or hor_pos not in range(10):  # If the piece has a tile off the board
                        return False
                    if self.game_board[vert_pos][hor_pos] == 1:  # If the piece tries to move over an occupied tile
                        return False
        return True

    def is_legal_wall_kick(self, piece):  # Cross-reference an illegal rotation with potential 'safety spots' it can translate to
        positions_to_move = [[0, 1], [0, -1], [0, 2], [0, -2], [1, 0], [-1, 0], [2, 0], [-2, 0]]
        for position in positions_to_move:
            if self.is_legal_move(piece, position):
                return True, position
        return False, [0, 0]

    def check_game_end(self):
        return True if not np.all((self.game_board[:4] == 0)) else False  # If all 0, game is not over

    def draw_game_over(self):
        font = pygame.font.SysFont("OCR A Extended", 50, bold=True)
        # Draw game over screen
        game_over_surface = pygame.Surface((self.TILE_SIZE * 10, self.TILE_SIZE * 24))
        game_over_surface.fill(color="black")
        game_over_surface.set_alpha(128)
        game_over_title = font.render('Game Over', True, (200, 200, 200))
        game_over_surface.blit(game_over_title, (self.TILE_SIZE, self.TILE_SIZE))
        self.window.blit(game_over_surface, (50, 50))

    def gravity_check(self):
        if time.time() - self.last_drop > self.gravity_timer:
            self.last_drop = time.time()
            return True
        return False

    def is_legal_rotation(self):
        next_piece_state = copy.deepcopy(self.current_piece)
        next_piece_state.rotate_right()
        for y in range(len(next_piece_state.shape[0])):  # For each row
            for x in range(len(next_piece_state.shape[1])):  # For each column
                if next_piece_state.shape[y][x] == 1:
                    tile_vert_pos = y + next_piece_state.ref_point[0]
                    tile_hor_pos = x + next_piece_state.ref_point[1]
                    if (tile_vert_pos not in range(24)) or (tile_hor_pos not in range(10)) or \
                            (self.game_board[tile_vert_pos][tile_hor_pos] == 1):  # If the current rotation is not legal, check for wall kick options
                        return self.is_legal_wall_kick(next_piece_state)   # If a wall kick is possible, return true for can rotate
        return True, [0, 0]

    def reset_bag(self):
        for i in range(7):  # Refill the bag with one of each Tetrino
            p = Tetrino()
            p.set_shape(i)
            self.bag.append(p)

    def set_next_piece(self):
        if len(self.bag) == 0:
            self.reset_bag()
        if self.next_piece is None:
            self.current_piece = self.bag.pop(random.randint(0, len(self.bag) - 1))
        else:
            self.current_piece = self.next_piece
        self.next_piece = self.bag.pop(random.randint(0, len(self.bag) - 1))

    def execute_curr_state(self, move=-1):
        # move_options = {0: Left, 1: Right, 2: Soft-drop, 3: Hard-drop, 4: Rotate, 5: Store}
        match move:
            case -1:  # None
                return 0
            case 0:  # Left
                if self.is_legal_move(self.current_piece, [0, -1]):
                    self.current_piece.move([0, -1])
            case 1:  # Right
                if self.is_legal_move(self.current_piece, [0, 1]):
                    self.current_piece.move([0, 1])
            case 2:  # Soft drop
                if self.is_legal_move(self.current_piece, [1, 0]):
                    self.current_piece.move([1, 0])
                    self.last_drop = time.time()  # Soft drop resets gravity clock
                else:
                    self.update_board()
                    self.set_next_piece()
            case 3:  # Hard drop
                while self.is_legal_move(self.current_piece, [1, 0]):  # Move the piece as far down as it can go legally
                    self.current_piece.move([1, 0])
                self.update_board()  # Add the pieces occupied tiles to the game board data
                self.current_piece = self.next_piece
                self.set_next_piece()  # Change to the next piece and fetch a new next piece from bag
            case 4:  # Rotate
                can_kick, kick_dir = self.is_legal_rotation()
                if can_kick:
                    self.current_piece.rotate_right(kick_dir)
            case 5:  # Need to implement
                print("Store")

        self.draw_board()
        if self.check_game_end():
            self.playing = False

        # pygame.time.delay(100)
        return self.get_reward()

    def check_full_rows(self):  # Handles changes when a row is filled
        # Check for which rows to clear and how many cleared in a single move
        new_lines_cleared = 0
        for row in range(24):
            if 0 not in self.game_board[row]:
                for i in range(row - 1):  # Shift all rows above the removed one down
                    self.game_board[row - i] = self.game_board[row - i - 1]
                self.game_board[0] = [0] * 10
                new_lines_cleared += 1
        if new_lines_cleared == 0:
            return
        # Score: P = {1: 40 * (n + 1), 2: 100 * (n + 1), 3: 300 * (n + 1), 4: 1200 * (n + 1): n = Level}
        scores_per_line = {1: 40, 2: 100, 3: 300, 4: 1200}
        self.score += scores_per_line.get(new_lines_cleared) * (self.level + 1)
        self.lines_cleared += new_lines_cleared
        self.level = math.floor(self.lines_cleared / 10)

    def get_reward(self):
        score_before = self.score
        self.check_full_rows()
        pygame.display.update()
        return self.score - score_before  # Return the change in score
        # self.draw_board(self.current_piece)  # Draw the board and the current active piece

    def play(self):  # For the user to play the game
        self.render()
        self.set_next_piece()  # Set the first two pieces
        self.draw_board()  # Show the blank board and current piece at the top of the board

        while self.playing:  # Keep running until terminal condition is met
            if self.gravity_check():  # Drop the piece every t seconds, t is based on current level
                self.execute_curr_state(2)

            event = pygame.event.get()  # Get user action and check if it does anything
            if len(event) > 0:
                if event[0].type == pygame.KEYDOWN:
                    match event[0].key:
                        # Quit
                        case pygame.K_ESCAPE:
                            self.playing = False
                        # Move left
                        case pygame.K_LEFT | pygame.K_a:
                            self.execute_curr_state(0)
                        # Move right
                        case pygame.K_RIGHT | pygame.K_d:
                            self.execute_curr_state(1)
                        # Move down
                        case pygame.K_DOWN | pygame.K_s:
                            self.execute_curr_state(2)
                        # Hard drop
                        case pygame.K_SPACE:
                            self.execute_curr_state(3)
                        # Rotation
                        case pygame.K_UP | pygame.K_w:
                            self.execute_curr_state(4)
            self.draw_board()  # Update to display the last action
            # pygame.time.delay(75)  # Delay between moves to prevent one key input counting several times

        self.draw_game_over()
        pygame.time.delay(1000)
        pygame.quit()


"""env = Tetris()
env.render()
pygame.display.init()
while env.playing:
    env.set_next_piece()
    print(env.current_piece.shape_id)
    move = 1  # Down
    env.execute_curr_state(move)  # Move the piece and update board
    env.draw_board()
    pygame.time.delay(100)"""
T = Tetris()
T.play()
pygame.quit()

