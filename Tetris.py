import pygame
import numpy as np
import random
import time
import copy

from numpy import character


class Tetrino:
    TETRINOS = {  # Shape, Color
        0: [np.array([[1, 1], [1, 1]]), [255, 255, 0]],  # O
        1: [np.array([[0, 0, 1], [1, 1, 1], [0, 0, 0]]), [255, 127, 0]],  # L
        2: [np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]]), [0, 0, 255]],  # J
        3: [np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]]), [0, 255, 0]],  # S
        4: [np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]]), [255, 0, 0]],  # Z
        5: [np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]), [0, 255, 255]],  # I
        6: [np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]), [225, 0, 225]]  # T
    }

    def __init__(self):
        self.shape = None
        self.ref_point = np.array([0, 0], dtype=int)
        self.dim = [0, 0]
        self.rotation = 0
        self.color = [0, 0, 0]
        self.shape_id = -1

    def set_shape(self, shape_id: int):
        self.shape_id = shape_id
        self.shape, self.color = Tetrino.TETRINOS.get(shape_id, "No such shape exists")
        self.ref_point = np.array([0, 3])  # (y, x)

        if self.shape.shape is not None:
            height = 0
            width = 0
            for i in range(len(self.shape[0])):
                if np.any(self.shape[i] == 1):
                    height += 1
                if np.any(self.shape.T[i] == 1):
                    width += 1

            self.dim = [height, width]

    def move(self, move: [int, int]):
        self.ref_point += move

    def rotate_right(self):
        self.shape = np.flipud(self.shape).T  # Rotate
        self.dim = self.dim[::-1]  # Adjust dimensions


class Tetris:
    BOARD_HEIGHT = 900
    BOARD_WIDTH = 600
    TILE_SIZE = 30  # Make the tile size based on the smaller axis (height or width)
    BOARD_DIMS = (24, 10)  # 20 rows + 4 for spawning the tile
    GRAVITY_TIMER = 60

    def __init__(self, score=0, level=0, gravity_timer=30, current_piece=None):
        self.score = score
        self.level = level
        self.gravity_timer = gravity_timer
        self.current_piece = current_piece
        self.next_piece = None
        self.playing = False

        # Sloppy.  Move this to a method and call it in play to update
        self.gameboard = np.zeros(shape=Tetris.BOARD_DIMS, dtype=int)  # 20 plus a few lines for it to fall into place
        self.window = pygame.display.set_mode((Tetris.BOARD_WIDTH, Tetris.BOARD_HEIGHT), 0, 32)
        self.last_drop = time.time()

    def render(self):
        pygame.display.init()

    def draw_graphics(self):
        # Setup pygame text
        pygame.font.init()
        font = pygame.font.SysFont("OCR A Extended", 50, bold=True)
        font2 = pygame.font.SysFont("OCR A Extended", 30, bold=False)
        font3 = pygame.font.SysFont("OCR A Extended", 15, bold=False)

        # Draw graphics outside the gameboard
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
        print(self.gameboard)

    def get_board(self):
        return self.gameboard

    def get_tile(self, color=[135, 135, 135],
                 border_color='black'):  # The board is made up of tiles. This function returns a given tile and its correct color
        tile = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE))
        tile.fill(color=border_color)  # border
        pygame.draw.rect(tile, color=color, rect=(1, 1, self.TILE_SIZE - 2, self.TILE_SIZE - 2))  # tile color
        return tile

    def draw_board(self, piece: Tetrino):
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
                if self.gameboard[y][x] != 0:
                    next_board_surface.blit(blank_tile, (x * self.TILE_SIZE, y * self.TILE_SIZE))

        # Prep to draw the active piece
        piece_tile = self.get_tile(piece.color)
        piece_surface = pygame.Surface((4 * self.TILE_SIZE, 4 * self.TILE_SIZE), pygame.SRCALPHA, 32)
        piece_surface = piece_surface.convert_alpha()

        # Prep to draw active pieces shadow
        shadow = copy.deepcopy(piece)
        shadow_tile = self.get_tile([100, 100, 100])
        shadow_surface = pygame.Surface((self.TILE_SIZE * 4, self.TILE_SIZE * 4), pygame.SRCALPHA, 32)
        while self.is_legal_move(shadow, [1, 0]):
            shadow.move([1, 0])

        # Draw active piece and shadow
        for y in range(len(piece.shape[0])):  # For each row
            for x in range(len(piece.shape[1])):  # For each column
                if piece.shape[y][x] != 0:
                    piece_surface.blit(piece_tile, (x * self.TILE_SIZE, y * self.TILE_SIZE))
                    shadow_surface.blit(shadow_tile, (x * self.TILE_SIZE, y * self.TILE_SIZE))

        # Push changes to the screen
        next_board_surface.blit(shadow_surface,
                                (self.TILE_SIZE * shadow.ref_point[1], self.TILE_SIZE * shadow.ref_point[0]))
        next_board_surface.blit(piece_surface,
                                (self.TILE_SIZE * piece.ref_point[1], self.TILE_SIZE * piece.ref_point[0]))

        self.draw_graphics()
        self.window.blit(next_board_surface, (50, 50))
        pygame.display.update()

    def update_board(self, piece: Tetrino):
        for y in range(len(piece.shape[0])):  # For each row
            for x in range(len(piece.shape[1])):  # For each column
                if piece.shape[y][x] != 0:
                    self.gameboard[y + piece.ref_point[0]][x + piece.ref_point[1]] += piece.shape[y][x]
            pygame.display.update()

    def is_legal_move(self, piece: Tetrino, move: [int, int]):
        new_ref_point = piece.ref_point + move

        if new_ref_point[1] not in range(0 + piece.dim[1] - len(piece.shape[1]), 11 - piece.dim[1]):
            return False

        if new_ref_point[0] not in range(25 - piece.dim[0]):
            return False

        for y in range(len(piece.shape[0])):  # For each row
            for x in range(len(piece.shape[1])):  # For each column

                if piece.shape[y][x] == 1:
                    vert_pos = y + new_ref_point[0]
                    hor_pos = x + new_ref_point[1]
                    print(hor_pos)

                    if vert_pos not in range(24):
                        return False
                    if hor_pos not in range(10):
                        return False
                    if self.gameboard[vert_pos][hor_pos] == 1:
                        return False
        return True

    def check_full_rows(self):
        # Check for which rows to clear and how many cleared in a single move
        full_count = 0
        for row in range(24):
            if 0 not in self.gameboard[row]:
                for i in range(row - 1):  # Shift all rows above the removed one down
                    self.gameboard[row - i] = self.gameboard[row - i - 1]
                self.gameboard[0] = [0] * 10
                full_count += 1
        if full_count == 0:
            return
        # Score: P = 40 * (n + 1) 100 * (n + 1) 300 * (n + 1) 1200 * (n + 1): n = Level
        score_per_line = {1: 40, 2: 100, 3: 300, 4: 1200}
        points = score_per_line.get(full_count) * (self.level + 1)
        self.score += points

    def check_game_end(self):
        return True if np.all((self.gameboard[3] == 0)) else False  # Only need to check the bottom row from the spawn layer

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
        if time.time() - self.last_drop > self.gravity_timer < time.time():
            self.last_drop = time.time()
            return True
        return False

    def is_legal_wall_kick(self, piece):
        positions_to_move = [[1, 0], [-1, 0], [0, 1], [0, 2], [0, -1], [0, -2]]
        for position in positions_to_move:
            if self.is_legal_move(piece, position):
                piece.move(position)
                return True
        return False

    def is_legal_rotation(self, piece):
        next_state = copy.deepcopy(piece)
        next_state.rotate_right()
        for y in range(len(next_state.shape[0])):  # For each row
            for x in range(len(next_state.shape[1])):  # For each column
                if next_state.shape[y][x] == 1:
                    tile_vert_pos = y + next_state.ref_point[0]
                    tile_hor_pos = x + next_state.ref_point[1]
                    if (tile_vert_pos not in range(24)) or (tile_hor_pos not in range(10)) or \
                            (self.gameboard[tile_vert_pos][tile_hor_pos] == 1):
                        return self.is_legal_wall_kick(next_state)
        return True


    def reset_bag(self):
        bag = []
        for id in range(7):
            p = Tetrino()
            p.set_shape(id)
            bag.append(p)
        return bag

    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_piece

        if piece_id == 6:
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states

    def set_next_piece(self):
        self.current_piece = self.next_piece




    def execute_curr_state(self, move=0):
        # Check for computer input, else human input
        # if loc:
            # board.update_board( # loc tiles as a piece)
        # mouse_pos = pygame.mouse.get_pos()  # Setup mouse
        # Gravity
        running = True
        while running:
            if self.gravity_check():
                if self.is_legal_move(self.current_piece, [1, 0]):
                    self.current_piece.move([1, 0])
                else:
                    return

            # Not working
            # Single tap key actions
            for event in pygame.event.get():  # For every action the user does
                if event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if event.key == pygame.K_ESCAPE:
                        self.playing = False
                        pygame.quit()

                    if event.key == pygame.K_SPACE:
                        while self.is_legal_move(self.current_piece, [1, 0]):
                            self.current_piece.move([1, 0])
                        self.update_board(self.current_piece)
                        self.current_piece = self.next_piece
                        self.next_piece =

                    if (event.key == pygame.K_UP or event.key == pygame.K_w) and self.is_legal_rotation(self.current_piece):
                        self.current_piece.rotate_right()

                    if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and self.is_legal_move(self.current_piece, [0, 1]):
                        self.current_piece.move([0, 1])  # Move right

                    if (keys[pygame.K_LEFT] or keys[pygame.K_a]) and self.is_legal_move(self.current_piece, [0, -1]):
                        self.current_piece.move([0, -1])  # Move left

                    if (keys[pygame.K_DOWN] or keys[pygame.K_s]) and self.is_legal_move(self.current_piece, [1, 0]):
                        self.current_piece.move([1, 0])  # Move down


            # maybe nest the hold keys within the pressed ones for the initial time delay.  Have a delay for each single pressed key
            # With no delay for held keys, and the held keys start one delay after the initial key press.  This allows the user
            # to tap the piece one to the side without overshooting, but also allowing them to hold the key for a faster move

            # Hold key down movements



            self.check_full_rows()
            pygame.display.update()
            self.draw_board(self.current_piece)  # Draw the board and the current active piece


    # User play
    def play(self):
        self.render()
        bag = self.reset_bag()
        self.current_piece = bag.pop(random.randint(0, len(bag) - 1))
        self.next_piece = bag.pop(random.randint(0, len(bag) - 1))
        self.playing = True
        while self.playing:

            if len(bag) == 0:
                bag = self.reset_bag()

            # Pygame timer delay
            pygame.time.delay(50)
            self.execute_curr_state()
            self.update_board(self.current_piece)
            self.current_piece = self.next_piece
            self.next_piece = bag.pop(random.randint(0, len(bag) - 1))


        self.draw_game_over()
        pygame.time.delay(1000)



    def step(self, move: int):
        moves = {0: [pygame.K_LEFT, pygame.K_a], 1: [pygame.K_RIGHT, K_d]}




env = Tetris()
env.play()
