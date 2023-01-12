# packages
import math
import numpy as np
import torch
import torch.nn as nn


# visible stamp
class BasicStamp(nn.Module):

    def __init__(
            self,
            n_malicious: int = 0, dba: bool = 0,  # number of users for stamp to be shared by or distributed between
            row_shift: int = 0, col_shift: int = 0,  # offset of image from upper left corner
            row_size: int = 4, col_size: int = 4,  # size of the stamp for EACH user
            row_gap: int = 0, col_gap: int = 0  # gap between placement of EACH user
    ):

        # setup
        super(BasicStamp, self).__init__()
        self.n_malicious, self.dba = n_malicious, dba
        self.row_shift, self.col_shift = row_shift, col_shift
        self.row_size, self.col_size = row_size, col_size

        # find grid to distribute stamp between
        if self.dba:
            assert self.n_malicious > 1, 'dba requested but multiple users are not being requested for n_malicious'
            root = math.isqrt(self.n_malicious)

            # grid distribution of stamp
            if root ** 2 == self.n_malicious:
                self.user_rows, self.user_cols = root, root
            elif root * (root + 1) == self.n_malicious:
                self.user_rows, self.user_cols = root, root + 1
            else:
                raise ValueError(
                    f'Input n_malicious={self.n_malicious} is not easily grid divisible. '
                    'Some suggested values for n_malicious include'
                    f'{root ** 2}, {root * (root + 1)}, and {(root + 1) ** 2}. '
                )

        # if centralized attack with multiple attackers
        else:
            assert row_gap + col_gap == 0, 'gap between users specified, but attack is not distributed among users'

            # adjust stamp size
            self.user_rows, self.user_cols = 1, 1

        # generate stamp
        self.stamp = torch.zeros(
            (self.user_rows * row_size, self.user_cols * col_size)
        )

        # create stamp pattern
        i, j = self.stamp.shape
        i = i // 4
        j = j // 4
        self.stamp[i:(3 * i), :] = 1
        self.stamp[:, j:(3 * j)] = 1

        # self.stamp = self.stamp.uniform_()  # get randomized stamp

        # objects for stamp distribution
        self.stamp_row_starts = row_size * np.arange(self.user_rows)
        self.stamp_col_starts = col_size * np.arange(self.user_cols)

        # save object to indicate stamp placements
        row_move, col_move = row_size + row_gap, col_size + col_gap
        self.input_row_starts = row_move * np.arange(self.user_rows) + self.row_shift
        self.input_col_starts = col_move * np.arange(self.user_cols) + self.col_shift

    def _forward_helper(self, x, user_id):

        # identify user's placement within the grid
        row, col = user_id // self.user_cols, user_id % self.user_cols

        input_row_start, input_col_start = self.input_row_starts[row], self.input_col_starts[col]
        input_row_end, input_col_end = input_row_start + self.row_size, input_col_start + self.col_size

        assert (input_row_end <= x.size(-1) and input_col_end <= x.size(-2))
        to_stamp = self.stamp[input_row_start:input_row_end, input_col_start:input_col_end]
        x[..., input_row_start:input_row_end, input_col_start:input_col_end] = to_stamp

        return x

    def forward(self, x, user_id: int = -1):
        assert user_id >= -1, 'need to specify valid user_id for stamp distribution and location'

        if not self.dba:
            col_stop, row_stop = self.col_shift + self.col_size, self.row_shift + self.row_size
            x[..., self.col_shift:col_stop, self.row_shift:row_stop] = self.stamp

        else:

            if user_id == -1:
                for i in range(self.n_malicious):
                    x = self._forward_helper(x, i)

            else:
                x = self._forward_helper(x, user_id)

        return x
