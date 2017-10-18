from utils import param_init, _p, log, _slice, ln, debug
import numpy

class GRU(object):

    '''
    Gated Recurrent Unit network with layer normalization:

        z_t = sigmoid(x_t dot W_xz + h_{t-1} dot U_hz)
        r_t = sigmoid(x_t dot W_xr + h_{t-1} dot U_hr)

        => zr_t = sigmoid(x_t dot W_xzr + h_{t-1} dot U_hzr)

        h_above = tanh(x_t dot W_xh + (h_{t-1} dot U_hh) * r_t)
        #h_above = tanh(x_t dot W_xh + (h_{t-1} * r_t) dot U_hh)

        h_t = (1 - z_t) * h_above + z_t * h_{t-1}
        #h_t = (1 - z_t) * h_{t-1} + z_t * h_above

    all parameters are initialized in [-0.01, 0.01]
    '''

    def __init__(self,
                 input_dims,
                 output_dims,
                 flag_ln=False,
                 prefix='GRU', **kwargs):

        debug('Init ' + prefix +' parameters ... ', False)
        self.input_dims = input_dims  # the units number of input layer (embsize)
        self.output_dims = output_dims  # the units number of hidden layer
        self.flag_ln = flag_ln
        self.prefix = prefix

        debug('{{{} => {}}} '.format(self.input_dims, output_dims), False)

        self.params = []
        self._init_params()
        debug('... Done')

    def _init_params(self):

        f = lambda name: _p(self.prefix, name)  # return 'GRU_' + parameters name

        size_xh = (self.input_dims, self.output_dims)
        size_hh = (self.output_dims, self.output_dims)
        # following three are parameters matrix from input layer to hidden layer:
        # generate numpy.ndarray by normal distribution

        self.W_xzr = param_init().param(size_xh, concat=True, name=f('W_xzr'))
        self.U_hzr = param_init().param(size_hh, concat=True, name=f('U_hzr'))

        # following three are parameters matrix from hidden layer to hidden layer:
        # generate numpy.ndarray by standard normal distribution with qr
        # factorization
        self.W_xh = param_init().param(size_xh, 'normal', name=f('W_xh'))
        self.U_hh = param_init().param(size_hh, 'orth', name=f('U_hh'))

        # following three are bias vector of hidden layer: generate by normal distribution
        self.b_zr = param_init().param((2 * self.output_dims, ), name=f('b_zr'))
        self.b_h = param_init().param((self.output_dims,), name=f('b_h'))

# just put all this parameters matrix (numpy.ndarray) into a list
        self.params = self.params + [self.W_xzr, self.W_xh,
                                     self.U_hzr, self.U_hh,
                                     self.b_zr, self.b_h]

        # default False
        if self.flag_ln is not False:

            mul_scale = 1.0
            add_scale = 0.0

            self.g1 = param_init().param((2 * self.output_dims,),
                                         scale=mul_scale,
                                         name=_p(self.prefix, 'g1'))
            self.b1 = param_init().param((2 * self.output_dims,),
                                         scale=add_scale,
                                         name=_p(self.prefix, 'b1'))

            self.g2 = param_init().param((self.output_dims,),
                                         scale=mul_scale,
                                         name=_p(self.prefix, 'g2'))
            self.b2 = param_init().param((self.output_dims,),
                                         scale=add_scale,
                                         name=_p(self.prefix, 'b2'))

            self.g3 = param_init().param((2 * self.output_dims,),
                                         scale=mul_scale,
                                         name=_p(self.prefix, 'g3'))
            self.b3 = param_init().param((2 * self.output_dims,),
                                         scale=add_scale,
                                         name=_p(self.prefix, 'b3'))

            self.g4 = param_init().param((self.output_dims,),
                                         scale=mul_scale,
                                         name=_p(self.prefix, 'g4'))
            self.b4 = param_init().param((self.output_dims,),
                                         scale=add_scale,
                                         name=_p(self.prefix, 'b4'))

            self.params += [self.g1, self.b1, self.g2, self.b2,
                            self.g3, self.b3, self.g4, self.b4]

    '''
    x_zr_t: input representation dot W_xzr at time t
    x_h_t: input representation dot W_xh at time t
    xmask_t: mask of xseq_repr_t at time t
    h_tm1: previous state
    '''
    def next_state(self, x_zr_t, x_h_t, xmask_t, h_tm1):

        if self.flag_ln is not False:
            x_zr_t = ln(x_zr_t, self.g1, self.b1)
            x_h_t = ln(x_h_t, self.g2, self.b2)

        h_zr_tm1 = T.dot(h_tm1, self.U_hzr)

        if self.flag_ln is not False:
            h_zr_tm1 = ln(h_zr_tm1, self.g3, self.b3)

        rz_t = x_zr_t + h_zr_tm1

        r_t = T.nnet.sigmoid(_slice(rz_t, 0, self.output_dims))
        z_t = T.nnet.sigmoid(_slice(rz_t, 1, self.output_dims))

        h_h_tm1 = T.dot(h_tm1, self.U_hh)

        if self.flag_ln is not False:
            h_h_tm1 = ln(h_h_tm1, self.g4, self.b4)

        '''
        h_h_tm1 = T.dot(h_tm1 * r_t, self.U_hh)
        if self.flag_ln is not False:
            h_h_tm1 = ln(h_h_tm1, self.g4, self.b4)
        '''

        h_h_tm1 = h_h_tm1 * r_t
        h_h_tm1 = h_h_tm1 + x_h_t
        h_t_above = T.tanh(h_h_tm1)

        h_t = (1. - z_t) * h_t_above + z_t * h_tm1
        # just because this, look for the reason for 6 hours,
        # when it is a little different from training, error...
        # h_t = (1. - z_t) * h_tm1 + z_t * h_t_above

        if xmask_t is not None:
            h_t = xmask_t[:, None] * h_t + (1. - xmask_t)[:, None] * h_tm1
        return h_t

    '''
        seq_repr: (src_sent_len,batch_size,embsize)
        seq_mask: (src_sent_len,batch_size) 0-1 matrix (padding)
        return: (src_sent_len, batch_size, output_dims)
    '''

    def forward(self, seq_repr, seq_mask=None, init_state=None):

        n_steps = seq_repr.shape[0]
        if seq_repr.ndim == 3:  # state_below is a 3-d matrix
            batch_size = seq_repr.shape[1]
        else:
            raise NotImplementedError

        seq_repr_zr = T.dot(seq_repr, self.W_xzr) + self.b_zr
        seq_repr_h = T.dot(seq_repr, self.W_xh) + self.b_h

        if seq_mask is not None:
            seqs = [seq_repr_zr, seq_repr_h, seq_mask]
            fn = self.next_state
        else:
            seqs = [seq_repr_zr, seq_repr_h]
            fn = lambda x1, x2: self.next_state(x1, x2, None)

        if init_state is None:
            init_state = T.alloc(numpy.float32(0.), batch_size, self.output_dims)

        rval, updates = theano.scan(fn,
                                    sequences=seqs,
                                    outputs_info=[init_state],
                                    n_steps=n_steps
                                    )
        self.output = rval
        return self.output

