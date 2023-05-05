from tkinter import *
from tkinter.ttk import *


from Board import Board
from Piece import Piece
import pygame
import random
import time
import math
import copy
import numpy as np

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
    K_SPACE,
    K_c,
)


# Updates the board to show when a falling piece moves
def show_piece_move(board):
    # Blank board
    screen.blit(board.get_board_display(),
                (screen_width / 2 - board.board_width / 2, screen_height / 2 - board.board_height / 2))

    # If the piece cannot move down, store it where it is
    if board.curr_piece.bottom:
        board.update_board_data(board.curr_piece)
        screen.blit(board.get_board_display(),
                    (screen_width / 2 - board.board_width / 2, screen_height / 2 - board.board_height / 2))

    # Else, show a temp display of the pieces current position, and the pieces shadow location
    else:
        screen.blit(board.get_piece_display(board.curr_piece),
                    (screen_width / 2 - board.board_width / 2, screen_height / 2 - board.board_height / 2))
    pygame.display.flip()


def show_board(board):
    # Blank board
    screen.blit(board.get_board_display(),
                (screen_width / 2 - board.board_width / 2, screen_height / 2 - board.board_height / 2))

    screen.blit(board.get_stored_piece_display(), (300, 131))
    screen.blit(board.get_next_piece_display(), (900, 131))
    pygame.display.flip()


def get_piece():
    board.reset_pieces()
    all_pieces = board.pieces

    if len(board.bag) == 0:
        board.reset_bag()

    piece_index = random.randint(0, len(board.bag) - 1)
    piece_id = board.bag[piece_index]
    random_piece = all_pieces[piece_id]
    board.bag.pop(piece_index)
    return random_piece


def play():
    # Get the first piece and the first 3 queued pieces
    board.curr_piece = get_piece()
    board.curr_piece.set_next(get_piece())
    board.curr_piece.next.set_next(get_piece())
    board.curr_piece.next.next.set_next(get_piece())
    board.curr_piece.next.next.next.set_next(get_piece())

    # Blank board display and game variables
    start_time = time.time()
    stored_count = 0
    seconds = 0.9
    time_per_tile = 0.05
    playing = True

    # play the game
    while playing:

        # Time since last gravitational movement
        curr_time = time.time()
        elapsed_time = curr_time - start_time
        held_keys = pygame.key.get_pressed()

        # Gravity
        if elapsed_time > seconds:
            # If the piece can move down legally, move it down one space
            if board.is_legal_down(board.curr_piece):
                board.curr_piece.move_down()

            # Else, store the piece data where it is, and start over with a new random piece
            else:

                # Update the board to store the piece location, and show the piece on the board
                board.update_board_data(board.curr_piece)

                # After the piece has been added to the board, get a new piece and show it in its starting position
                board.curr_piece = board.curr_piece.next
                board.curr_piece.next.next.next.set_next(get_piece())

            # Display the piece falling down by one, and reset the timer
            show_board(board)
            show_piece_move(board)
            start_time = time.time()

        if held_keys[K_DOWN]:
            if elapsed_time > time_per_tile:
                # If the piece can move down legally, move it down one space
                if board.is_legal_down(board.curr_piece):
                    board.curr_piece.move_down()

                # Else, store the piece data where it is, and start over with a new random piece
                else:

                    # Update the board to store the piece location, and show the piece on the board
                    board.update_board_data(board.curr_piece)

                    # After the piece has been added to the board, get a new piece and show it in its starting position
                    board.curr_piece = board.curr_piece.next
                    board.curr_piece.next.next.next.set_next(get_piece())

                # Display the piece falling down by one, and reset the timer
                show_board(board)
                show_piece_move(board)
                start_time = time.time()

        # Runs every time an action is detected from  the user
        for event in pygame.event.get():

            # Terminate the game if they close the popup window
            if event.type == QUIT:
                playing = False

            # Checks if a key was pressed
            elif event.type == KEYDOWN:

                # End the game if the user hits the escape key
                if event.key == K_ESCAPE:
                    playing = False

                elif event.key == K_SPACE:

                    # Move the piece down until one of its tiles collides with something
                    while not board.curr_piece.bottom:
                        # if the space below each tile is open, move down
                        if board.is_legal_down(board.curr_piece):
                            board.curr_piece.move_down()
                        else:
                            board.curr_piece.bottom = True
                            break

                # Store piece
                elif event.key == K_c and stored_count == 0:

                    temp = copy.deepcopy(board.curr_piece)
                    stored_count += 1
                    board.curr_piece.rotations = board.curr_piece.start_pos
                    board.curr_piece = board.store_piece(board.curr_piece)

                    if board.curr_piece is None:
                        board.curr_piece = temp.next
                    else:
                        board.curr_piece.set_next(temp.next)

                    board.curr_piece.next.next.next.set_next(get_piece())

                elif event.key == K_LEFT:
                    if board.is_legal_left(board.curr_piece):
                        board.curr_piece.move_left()

                elif event.key == K_RIGHT:
                    if board.is_legal_right(board.curr_piece):
                        board.curr_piece.move_right()

                # If the user rotates the piece, rotate it to its new position the check if its new spot is legal.
                elif event.key == K_UP:

                    board.curr_piece.rotate()
                    # If the rotation is not legal, check for wall kick possibilities
                    if not board.is_legaL_rotation(board.curr_piece):

                        # Wall kick off left wall
                        if board.is_legal_right(board.curr_piece):
                            board.curr_piece.move_right()

                        # Wall kick off right wall
                        elif board.is_legal_left(board.curr_piece):
                            board.curr_piece.move_left()

                        else:
                            board.curr_piece.rotate_opposite()

            if board.curr_piece.bottom:
                # Update the board to store the piece location, and show the piece on the board
                board.update_board_data(board.curr_piece)
                board.check_for_full_row()

                # After the piece has been added to the board, get a new piece and show it in its starting position
                board.curr_piece = board.curr_piece.next
                board.curr_piece.next.next.next.set_next(get_piece())
                stored_count = 0


            # Update board to show the move on this line
            show_board(board)
            show_piece_move(board)

        board.check_for_full_row()

        # Checks if the game has ended
        if playing:
            playing = board.is_legal_spawn()


screen_width = 1400
screen_height = 1000
screen = pygame.display.set_mode((screen_width, screen_height))
screen.fill((55, 55, 55))
pygame.init()
board = Board()
play()
