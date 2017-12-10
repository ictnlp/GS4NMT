import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from tools.utils import *

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
        self.hidden_size = hidden_size
        self.with_ln = with_ln
        self.prefix = prefix

        self.xh = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)
        self.xrz = nn.Linear(input_size, 2 * hidden_size)
        self.hrz = nn.Linear(hidden_size, 2 * hidden_size)

        if self.enc_hid_size is not None:
            self.crz = nn.Linear(enc_hid_size, 2 * hidden_size)
            self.ch = nn.Linear(enc_hid_size, hidden_size)

        #if self.with_ln is not True:

            #self.xz = nn.Linear(input_size, hidden_size)
            #self.hz = nn.Linear(hidden_size, hidden_size)
            #self.xr = nn.Linear(input_size, hidden_size)
            #self.hr = nn.Linear(hidden_size, hidden_size)

            #if self.enc_hid_size is not None:
                #self.cz = nn.Linear(2 * enc_hid_size, hidden_size)
                #self.cr = nn.Linear(2 * enc_hid_size, hidden_size)
                #self.ch = nn.Linear(2 * enc_hid_size, hidden_size)

        if self.with_ln is True:

            self.ln0 = Layer_Norm(2 * hidden_size)
            self.ln1 = Layer_Norm(2 * hidden_size)
            self.ln2 = Layer_Norm(hidden_size)
            self.ln3 = Layer_Norm(hidden_size)

            if self.enc_hid_size is not None:
                self.ln4 = Layer_Norm(2 * hidden_size)
                self.ln5 = Layer_Norm(hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
    '''
    def forward(self, x_t, x_m, h_tm1, attend=None):

        x_rz_t, h_rz_tm1, x_h_t = self.xrz(x_t), self.hrz(h_tm1), self.xh(x_t)

        if self.with_ln is not True:

            if self.enc_hid_size is None:
                #r_t = self.sigmoid(self.xr(x_t) + self.hr(h_tm1))
                #z_t = self.sigmoid(self.xz(x_t) + self.hz(h_tm1))
                #h_t_above = self.tanh(self.xh(x_t) + self.hh(r_t * h_tm1))
                rz_t = x_rz_t + h_rz_tm1
            else:
                #z_t = self.sigmoid(self.xz(x_t) + self.hz(h_tm1) + self.cz(attend))
                #r_t = self.sigmoid(self.xr(x_t) + self.hr(h_tm1) + self.cr(attend))
                #h_t_above = self.tanh(self.xh(x_t) + self.hh(r_t * h_tm1) + self.ch(attend))
                a_rz_t, a_h_t = self.crz(attend), self.ch(attend)
                rz_t = x_rz_t + h_rz_tm1 + a_rz_t

        else:

            x_rz_t, h_rz_tm1, x_h_t = self.ln0(x_rz_t), self.ln1(h_rz_tm1), self.ln2(x_h_t)

            if self.enc_hid_size is None:
                rz_t = x_rz_t + h_rz_tm1
            else:
                a_rz_t, a_h_t = self.crz(attend), self.ch(attend)
                a_rz_t, a_h_t = self.ln4(a_rz_t), self.ln5(a_h_t)
                rz_t = x_rz_t + h_rz_tm1 + a_rz_t

        assert rz_t.dim() == 2
        rz_t = self.sigmoid(rz_t)
        r_t, z_t = rz_t[:, :self.hidden_size], rz_t[:, self.hidden_size:]

        h_h_tm1 = self.hh(r_t * h_tm1)
        if self.with_ln: h_h_tm1 = self.ln3(h_h_tm1)
        #h_h_tm1 = h_h_tm1 * r_t
        if self.enc_hid_size is None: h_h_tm1 = x_h_t + h_h_tm1
        else: h_h_tm1 = x_h_t + h_h_tm1 + a_h_t

        h_t_above = self.tanh(h_h_tm1)

        h_t = (1. - z_t) * h_tm1 + z_t * h_t_above

        if x_m is not None:
            #x_m = x_m.unsqueeze(-1).expand_as(h_t)
            #h_t = x_m * h_t + (1. - x_m) * h_tm1
            h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1

        return h_t

