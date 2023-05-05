from tkinter import *
from tkinter.ttk import *
import pygame
from Piece import Piece
import numpy as np
import copy


class Board:

    def __init__(self):
        self.tile_size = 35
        self.board_width = self.tile_size * 10
        self.board_height = self.tile_size * 21
        self.board_display = pygame.Surface([self.board_width, self.board_height])
        self.stored_piece_display = pygame.Surface([5 * self.tile_size, 5 * self.tile_size])
        self.next_piece_display = pygame.Surface([5 * self.tile_size, 14 * self.tile_size])
        self.board_data = np.array([[[0, 0, 0] for y in range(21)] for x in range(10)])
        self.stored_piece = None
        self.curr_piece = None
        self.score = 0
        self.bag = [0, 1, 2, 3, 4, 5, 6]
        self.pieces = [

            # Square
            # Has 4 rotations to simplify piece movements so all pieces have the same amount of rotation
            Piece([[[4, 0], [5, 0], [4, 1], [5, 1]],
                   [[4, 0], [5, 0], [4, 1], [5, 1]],
                   [[4, 0], [5, 0], [4, 1], [5, 1]],
                   [[4, 0], [5, 0], [4, 1], [5, 1]]], color=[255, 255, 0]),

            # Z Shape and all rotations
            Piece([[[3, 0], [4, 0], [4, 1], [5, 1]],
                   [[4, 1], [4, 2], [5, 0], [5, 1]],
                   [[3, 1], [4, 1], [4, 2], [5, 2]],
                   [[3, 1], [3, 2], [4, 0], [4, 1]]], color=(255, 0, 0)),

            # S shape and all rotations
            Piece([[[3, 1], [4, 1], [4, 0], [5, 0]],
                   [[4, 0], [4, 1], [5, 1], [5, 2]],
                   [[3, 2], [4, 2], [4, 1], [5, 1]],
                   [[3, 0], [3, 1], [4, 1], [4, 2]]], color=(0, 255, 0)),

            # T shape piece and all rotations
            Piece([[[3, 1], [4, 0], [4, 1], [5, 1]],
                   [[4, 0], [4, 1], [4, 2], [5, 1]],
                   [[3, 1], [4, 1], [4, 2], [5, 1]],
                   [[3, 1], [4, 0], [4, 1], [4, 2]]], color=(225, 0, 225)),

            # J shape and all rotations
            Piece([[[3, 0], [3, 1], [4, 1], [5, 1]],
                   [[4, 0], [4, 1], [4, 2], [5, 0]],
                   [[3, 1], [4, 1], [5, 1], [5, 2]],
                   [[3, 2], [4, 0], [4, 1], [4, 2]]], color=(0, 0, 255)),

            # L shape and all rotations
            Piece([[[3, 1], [4, 1], [5, 0], [5, 1]],
                   [[4, 0], [4, 1], [4, 2], [5, 2]],
                   [[3, 1], [3, 2], [4, 1], [5, 1]],
                   [[3, 0], [4, 0], [4, 1], [4, 2]]], color=(255, 127, 0)),

            # Line shape and all rotations
            Piece([[[3, 0], [4, 0], [5, 0], [6, 0]],
                   [[5, -1], [5, 0], [5, 1], [5, 2]],
                   [[3, 1], [4, 1], [5, 1], [6, 1]],
                   [[4, -1], [4, 0], [4, 1], [4, 2]]], color=(0, 255, 255)),
        ]


    # Stores one piece for future use
    def store_piece(self, piece):
        # Takes current piece and repositions it to the top, so when stored it will be the same as a new piece
        if self.stored_piece is None:
            self.stored_piece = piece
            return None

        else:
            prev_stored = self.stored_piece
            self.stored_piece = piece
            return prev_stored


    def set_curr_piece(self, piece):
        self.curr_piece = piece


    # Return the display of the next 3 pieces in the queue
    def get_next_piece_display(self):

        pieces = [self.curr_piece.next, self.curr_piece.next.next, self.curr_piece.next.next.next, self.curr_piece.next.next.next.next]

        tile = pygame.Surface([self.tile_size, self.tile_size])
        self.next_piece_display.fill((0, 0, 0))
        pygame.draw.rect(self.next_piece_display, (42, 42, 42), (0, self.tile_size, 5 * self.tile_size, 2))
        i = 0

        for piece in pieces:

            min_x = 5
            max_y = 0
            max_x = 0

            for x, y in piece.start_pos[0]:
                if x < min_x:
                    min_x = x
                elif x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

            width = max_x - min_x + 1
            height = max_y + 1

            board_center = (2.5 * self.tile_size, 3 * self.tile_size)
            piece_to_center = (board_center[0] - (width / 2 * self.tile_size), board_center[1] - (height / 2 * self.tile_size))

            tile.fill((piece.color[0], piece.color[1], piece.color[2]))
            piece_surf = pygame.Surface([width * self.tile_size, height * self.tile_size])
            piece_surf.fill((0, 0, 0))

            for x, y in piece.start_pos[0]:
                piece_surf.blit(tile, ((x - min_x) * self.tile_size, y * self.tile_size))

            self.next_piece_display.blit(piece_surf, (piece_to_center[0], piece_to_center[1] + (i * 3 * self.tile_size)))
            i += 1

        pygame.font.init()
        my_font = pygame.font.SysFont(None, 30)
        title = my_font.render('Next Pieces', True, (222, 222, 222))
        self.next_piece_display.blit(title, (30, 8))

        return self.next_piece_display


    # Returns the display of the stored piece
    def get_stored_piece_display(self):

        tile = pygame.Surface([self.tile_size, self.tile_size])
        self.stored_piece_display.fill((0, 0, 0))
        pygame.draw.rect(self.stored_piece_display, (42, 42, 42), (0, self.tile_size, 5 * self.tile_size, 2))

        min_x = 5
        max_y = 0
        max_x = 0

        if self.stored_piece is not None:

            # Creates a surface of the piece
            for x, y in self.stored_piece.start_pos[0]:
                if x < min_x:
                    min_x = x
                elif x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

            width = max_x - min_x + 1
            height = max_y + 1

            # Centers the new surface
            board_center = (2.5 * self.tile_size, 3 * self.tile_size)
            piece_to_center = (board_center[0] - (width / 2 * self.tile_size), board_center[1] - (height / 2 * self.tile_size))

            # Print the piece onto the new surface
            tile.fill((self.stored_piece.color[0], self.stored_piece.color[1], self.stored_piece.color[2]))
            piece_surf = pygame.Surface([width * self.tile_size, height * self.tile_size])
            piece_surf.fill((0, 0, 0))

            for x, y in self.stored_piece.start_pos[0]:
                piece_surf.blit(tile, ((x - min_x) * self.tile_size, y * self.tile_size))

            # Move the surface with the piece into the center of the stored piece display area
            self.stored_piece_display.blit(piece_surf, piece_to_center)

        pygame.font.init()
        my_font = pygame.font.SysFont(None, 30)
        title = my_font.render('Stored Piece', True, (222, 222, 222))
        self.stored_piece_display.blit(title, (25, 8))

        return self.stored_piece_display


    # Returns the display of all pieces stored on the board excluding the currently active piece
    def get_board_display(self):
        tile = pygame.Surface((self.tile_size, self.tile_size))
        self.board_display.fill((0, 0, 0))

        for x in range(10):
            for y in range(21):
                tile.fill((self.board_data[x][y][0], self.board_data[x][y][1], self.board_data[x][y][2]))
                self.board_display.blit(tile, (x * self.tile_size, y * self.tile_size))

        return self.board_display


    # Shows the currently active pieces' location and shadow
    def get_piece_display(self, piece):
        temp_board = self.get_board_display()
        tile = pygame.Surface((self.tile_size, self.tile_size))

        shadow = copy.deepcopy(piece)
        while not shadow.bottom:
            # if the space below each tile is open, move down
            if self.is_legal_down(shadow):
                shadow.move_down()
            else:
                shadow.bottom = True

        for x, y in shadow.rotations[piece.rotation_count % 4]:
            tile.fill((155, 155, 155))
            temp_board.blit(tile, (x * self.tile_size, y * self.tile_size))

        for x, y in piece.rotations[piece.rotation_count % 4]:
            tile.fill((piece.color[0], piece.color[1], piece.color[2]))
            temp_board.blit(tile, (x * self.tile_size, y * self.tile_size))

        return temp_board


    # Once the active piece has been placed, add its position and color to the boards data table
    def update_board_data(self, piece):
        for x, y in piece.rotations[piece.rotation_count]:
            self.board_data[x][y] = np.array([piece.color[0], piece.color[1], piece.color[2]])


    # Checks if a row is full. If true, remove the filled row row and shift all the rows above it down to fill the gap
    def check_for_full_row(self):
        # Check for finished rows
        for y in range(21):
            for x in range(10):

                # If a row has a blank tile its not full, so move to next row
                if np.array_equal(self.board_data[x][y], np.array([0, 0, 0])):
                    break

                # Remove full rows
                elif not np.array_equal(self.board_data[x][y], np.array([0, 0, 0])) and x == 9:
                    blank_row = np.array([[0, 0, 0] for k in range(10)])
                    for i in range(y - 1):
                        self.board_data[:, y - i] = self.board_data[:, y - i - 1]
                    self.board_data[:, 0] = blank_row


    # Checks is the current piece can legally move down
    def is_legal_down(self, piece):
        for x, y in piece.rotations[piece.rotation_count]:
            if y == 20 or not np.array_equal(self.board_data[x][y + 1], np.array([0, 0, 0])):
                return False
        return True


    # Checks is the current piece can legally move rotate
    def is_legaL_rotation(self, piece):
        for x, y in piece.rotations[piece.rotation_count]:
            if (x < 0) or (x > 9) or not (np.array_equal(self.board_data[x][y], np.array([0, 0, 0]))):
                return False

        return True


    # Checks is the current piece can legally move right
    def is_legal_right(self, piece):
        for x, y in piece.rotations[piece.rotation_count]:
            if (x >= 9) or not (np.array_equal(self.board_data[x + 1][y], np.array([0, 0, 0]))):
                return False
        return True


    # Checks is the current piece can legally move left
    def is_legal_left(self, piece):
        for x, y in piece.rotations[piece.rotation_count]:
            if (x <= 0) or not (np.array_equal(self.board_data[x - 1][y], np.array([0, 0, 0]))):
                return False
        return True


    def reset_bag(self):
        self.bag = [0, 1, 2, 3, 4, 5, 6]


    def reset_pieces(self):
        self.pieces = [

            # Square
            # Has 4 rotations to simplify piece movements so all pieces have the same amount of rotation
            Piece([[[4, 0], [5, 0], [4, 1], [5, 1]],
                   [[4, 0], [5, 0], [4, 1], [5, 1]],
                   [[4, 0], [5, 0], [4, 1], [5, 1]],
                   [[4, 0], [5, 0], [4, 1], [5, 1]]], color=[255, 255, 0]),

            # Z Shape and all rotations
            Piece([[[3, 0], [4, 0], [4, 1], [5, 1]],
                   [[4, 1], [4, 2], [5, 0], [5, 1]],
                   [[3, 1], [4, 1], [4, 2], [5, 2]],
                   [[3, 1], [3, 2], [4, 0], [4, 1]]], color=(255, 0, 0)),

            # S shape and all rotations
            Piece([[[3, 1], [4, 1], [4, 0], [5, 0]],
                   [[4, 0], [4, 1], [5, 1], [5, 2]],
                   [[3, 2], [4, 2], [4, 1], [5, 1]],
                   [[3, 0], [3, 1], [4, 1], [4, 2]]], color=(0, 255, 0)),

            # T shape piece and all rotations
            Piece([[[3, 1], [4, 0], [4, 1], [5, 1]],
                   [[4, 0], [4, 1], [4, 2], [5, 1]],
                   [[3, 1], [4, 1], [4, 2], [5, 1]],
                   [[3, 1], [4, 0], [4, 1], [4, 2]]], color=(225, 0, 225)),

            # J shape and all rotations
            Piece([[[3, 0], [3, 1], [4, 1], [5, 1]],
                   [[4, 0], [4, 1], [4, 2], [5, 0]],
                   [[3, 1], [4, 1], [5, 1], [5, 2]],
                   [[3, 2], [4, 0], [4, 1], [4, 2]]], color=(0, 0, 255)),

            # L shape and all rotations
            Piece([[[3, 1], [4, 1], [5, 0], [5, 1]],
                   [[4, 0], [4, 1], [4, 2], [5, 2]],
                   [[3, 1], [3, 2], [4, 1], [5, 1]],
                   [[3, 0], [4, 0], [4, 1], [4, 2]]], color=(255, 127, 0)),

            # Line shape and all rotations
            Piece([[[3, 0], [4, 0], [5, 0], [6, 0]],
                   [[5, -1], [5, 0], [5, 1], [5, 2]],
                   [[3, 1], [4, 1], [5, 1], [6, 1]],
                   [[4, -1], [4, 0], [4, 1], [4, 2]]], color=(0, 255, 255)),
        ]


    def is_legal_spawn(self):
        for x, y in self.curr_piece.next.start_pos[0]:
            if not np.array_equal(self.board_data[x][y], np.array([0, 0, 0])):
                return False

        return True
