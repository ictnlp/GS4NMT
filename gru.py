from utils import *
import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

# inheriting from nn.Module
class GRU(nn.Module):

    '''
    Gated Recurrent Unit network with initial state as parameter:

        z_t = sigmoid((x_t dot W_xz + b_xz) + (h_{t-1} dot U_hz + b_hz))
        r_t = sigmoid((x_t dot W_xr + b_xr) + (h_{t-1} dot U_hr + b_xr))

        => zr_t = sigmoid((x_t dot W_xzr + b_xzr) + (h_{t-1} dot U_hzr + b_hzr))
        slice ...

        h_above = tanh(x_t dot W_xh + b_xh + (h_{t-1} dot U_hh + b_hh) * r_t)

        h_t = (1 - z_t) * h_above + z_t * h_{t-1}
        #h_t = (1 - z_t) * h_{t-1} + z_t * h_above

    all parameters are initialized in [-0.01, 0.01]
    '''

    def __init__(self,
                 input_size,
                 hidden_size,
                 enc_hid_size=None,
                 with_ln=False,
                 prefix='GRU', **kwargs):

        # input vector size and hidden vector size
        # calls the init function of nn.Module
        super(GRU, self).__init__()

        self.enc_hid_size = enc_hid_size
        self.with_ln = with_ln
        self.prefix = prefix

        if self.with_ln is not True:

            if self.enc_hid_size is None:

                self.xz = nn.Linear(input_size, hidden_size)
                self.hz = nn.Linear(hidden_size, hidden_size)
                self.xr = nn.Linear(input_size, hidden_size)
                self.hr = nn.Linear(hidden_size, hidden_size)
                self.xh = nn.Linear(input_size, hidden_size)
                self.hh = nn.Linear(hidden_size, hidden_size)
            else:
                self.yz = nn.Linear(input_size, hidden_size)
                self.hz = nn.Linear(hidden_size, hidden_size)
                self.cz = nn.Linear(2 * enc_hid_size, hidden_size)
                self.yr = nn.Linear(input_size, hidden_size)
                self.hr = nn.Linear(hidden_size, hidden_size)
                self.cr = nn.Linear(2 * enc_hid_size, hidden_size)
                self.yh = nn.Linear(input_size, hidden_size)
                self.hh = nn.Linear(hidden_size, hidden_size)
                self.ch = nn.Linear(2 * enc_hid_size, hidden_size)
        else:
            self.xzr = nn.Linear(input_size, 2 * hidden_size)
            self.hzr = nn.Linear(hidden_size, 2 * hidden_size)
            self.xh = nn.Linear(input_size, hidden_size)
            self.hh = nn.Linear(hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
    '''
    def forward(self, x_t, x_m, h_tm1, attend=None):

        if self.enc_hid_size is None:
            z_t = self.sigmoid(self.xz(x_t) + self.hz(h_tm1))
            r_t = self.sigmoid(self.xr(x_t) + self.hr(h_tm1))
            h_t_above = self.tanh(self.xh(x_t) + self.hh(r_t * h_tm1))
        else:
            z_t = self.sigmoid(self.yz(x_t) + self.hz(h_tm1) + self.cz(attend))
            r_t = self.sigmoid(self.yr(x_t) + self.hr(h_tm1) + self.cr(attend))
            h_t_above = self.tanh(self.yh(x_t) + self.hh(r_t * h_tm1) + self.ch(attend))

        h_t = (1. - z_t) * h_tm1 + z_t * h_t_above
        if x_m is not None:
            x_m = x_m.unsqueeze(-1).expand_as(h_t)
            h_t = x_m * h_t + (1. - x_m) * h_tm1

        return h_t

