import numpy as np


class Piece:

    def __init__(self, rotations: [[[int, int]]], rotation_count: int = 0, color=[0, 0, 0]):
        self.start_pos = np.array(rotations)
        self.rotations = np.array(rotations)
        self.rotation_count = rotation_count
        self.color = color
        self.next = None
        self.bottom = False


    # Shift the current piece to the left by one tile
    def move_left(self):
        for i in range(4):
            for j in range(4):
                self.rotations[j][i][0] -= 1
        return self


    # Shift the current piece to the right by one tile
    def move_right(self):
        for i in range(4):
            for j in range(4):
                self.rotations[j][i][0] += 1
        return self


    # Shift the current piece to the left by one tile
    def move_down(self):
        for i in range(4):
            for j in range(4):
                self.rotations[j][i][1] += 1
        return self

    # Rotate the piece once
    def rotate(self):
        curr_rotation = self.rotation_count
        self.rotation_count = (curr_rotation + 1) % 4

    def rotate_opposite(self):
        curr_rotation = self.rotation_count
        self.rotation_count = (curr_rotation + 3) % 4


    # Set the next piece that will come
    def set_next(self, piece):
        self.next = piece
