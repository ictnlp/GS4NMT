import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from tools.utils import *

# inheriting from nn.Module
class RAN(nn.Module):

    '''
    Recurrent Additive Networks

        i_t = Sigmoid((W_ic dot c_{t-1} + b_ic) + (W_ix dot x_t + b_ix))
        f_t = Sigmoid((W_fc dot c_{t-1} + b_fc) + (W_fx dot x_t + b_fx))
        c_t = i_t * x_t + f_t * c_{t-1}
        h_t = tanh(c_t) or h_t = c_t

    all parameters are initialized in [-0.01, 0.01]
    '''

    def __init__(self,
                 x_size,
                 c_size=None,
                 if_tanh=True,
                 residual=False,
                 laycnt=1,
                 with_ln=False,
                 share_weight=True,
                 prefix='RAN', **kwargs):

        # calls the init function of nn.Module
        super(RAN, self).__init__()

        # the units number of input (embsize)
        self.x_size = x_size
        self.residual = residual
        self.laycnt = laycnt

        # the units number of hidden
        self.h_size = c_size if c_size else x_size
        self.if_tanh = if_tanh
        self.with_ln = with_ln
        self.share_weight = share_weight
        self.prefix = prefix

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.complex = (c_size is not None) and (not c_size == x_size)

        if not self.with_ln:

            self.cx1, self.ci1, self.xi1, self.cf1, self.xf1, self.cz1, self.xz1 = \
                    self.init_modules()

            if share_weight is False:
                self.cx2, self.ci2, self.xi2, self.cf2, self.xf2, self.cz2, self.xz2 = \
                        self.init_modules()
        else:
            self.xzr = nn.Linear(self.input_size, 2 * self.hidden_size)
            self.hzr = nn.Linear(self.hidden_size, 2 * self.hidden_size)
            self.xh = nn.Linear(self.input_size, self.hidden_size)
            self.hh = nn.Linear(self.hidden_size, self.hidden_size)

    def init_modules(self):

        cx, cz, xz = None, None, None
        # only serves to allow different input vector and state vector dimensions
        if self.complex:
            cx = nn.Linear(self.x_size, self.h_size)
        ci = nn.Linear(self.h_size, self.h_size)
        xi = nn.Linear(self.x_size, self.h_size)
        cf = nn.Linear(self.h_size, self.h_size)
        xf = nn.Linear(self.x_size, self.h_size)
        if self.residual:
            cz = nn.Linear(self.h_size, self.h_size)
            xz = nn.Linear(self.x_size, self.h_size)

        return cx, ci, xi, cf, xf, cz, xz

    '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
    '''

    def _one_layer(self, x_t, x_m, c_tm1, flag=0):

        cx, ci, xi, cf, xf, cz, xz = \
                self.cx1, self.ci1, self.xi1, self.cf1, self.xf1, self.cz1, self.xz1

        if self.share_weight is False and flag == 1:
            cx, ci, xi, cf, xf, cz, xz = \
                    self.cx2, self.ci2, self.xi2, self.cf2, self.xf2, self.cz2, self.xz2

        c_t_above = cx(x_t) if cx is not None else x_t

        i_t = self.sigmoid(ci(c_tm1) + xi(x_t))
        f_t = self.sigmoid(cf(c_tm1) + xf(x_t))

        if cx is not None:
            c_t = i_t * c_t_above + f_t * c_tm1
        else:
            c_t = i_t * x_t + f_t * c_tm1

        if x_m is not None:
            x_m = tc.unsqueeze(x_m, dim=-1).expand_as(c_t)
            c_t = x_m * c_t + (1 - x_m) * c_tm1

        h_t = self.tanh(c_t) if self.if_tanh else c_t

        if cz is not None and xz is not None:
            z_t = self.sigmoid(cz(c_tm1) + xz(x_t))
            h_t = (1. - z_t) * h_t + z_t * c_t_above

        return c_t, h_t

    def forward(self, x_t, x_m, c_tm1, attent=None):

        for layer in range(self.laycnt):

            if attent is not None:
                x_t = tc.cat([x_t, attent], dim=-1)

            c_tm1[layer], x_t = self._one_layer(x_t, x_m, c_tm1[layer], flag=layer % 2)

        return c_tm1, x_t



