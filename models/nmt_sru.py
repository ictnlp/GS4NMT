import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import wargs
from utils import *

if 'sru' in [wargs.enc_rnn_type, wargs.dec_rnn_type]:
    from sru import check_sru_requirement
    can_use_sru = check_sru_requirement()
    if can_use_sru:
        from sru import SRU

if 'gru' in [wargs.enc_rnn_type, wargs.dec_rnn_type]:
    from gru import GRU
if 'lstm' in [wargs.enc_rnn_type, wargs.dec_rnn_type]:
    pass

class NMT(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size):

        super(NMT, self).__init__()

        self.bidire_enc = True
        self.encoder = Encoder(src_vocab_size, wargs.src_wemb_size, wargs.enc_hid_size,
                               bidirectional=self.bidire_enc, with_ln=wargs.laynorm)

        if wargs.enc_rnn_type == 'gru':
            ctx_size = wargs.enc_hid_size
            self.s_init = nn.Linear(ctx_size, wargs.dec_hid_size)
        elif wargs.enc_rnn_type == 'sru': ctx_size = 2 * wargs.enc_hid_size

        self.ha = nn.Linear(ctx_size, wargs.align_size)
        self.tanh = nn.Tanh()
        self.decoder = Decoder(trg_vocab_size, with_ln=wargs.laynorm)

    def init_state(self, xs_h, xs_mask=None, hidden=None):

        assert xs_h.dim() == 3  # slen, batch_size, enc_size
        if xs_mask is not None:
            xs_h = (xs_h * xs_mask[:, :, None]).sum(0) / xs_mask.sum(0)[:, None]
        else:
            xs_h = xs_h.mean(0)

        return self.tanh(self.s_init(xs_h))

    def init_state_sru(self, hidden):

        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """

        if self.bidire_enc:
            #hidden = tc.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
            hidden = tc.stack([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
            hidden = hidden.max(-2)[0]

        return hidden

    def init(self, xs, xs_mask=None, test=True):

        if test:  # for decoding
            if wargs.gpu_id and not xs.is_cuda: xs = xs.cuda()
            xs = Variable(xs, requires_grad=False, volatile=True)

        if wargs.enc_rnn_type == 'gru':
            xs = self.encoder(xs, xs_mask)
            s0 = self.init_state(xs, xs_mask)
        elif wargs.enc_rnn_type == 'sru':
            xs, hid = self.encoder(xs, xs_mask)
            #print xs.size(), hid.size()
            #s0 = self.init_state(xs, xs_mask)
            s0 = self.init_state_sru(hid)

        uh = self.ha(xs)

        return s0, xs, uh

    def forward(self, srcs, trgs, srcs_m, trgs_m):
        # (max_slen_batch, batch_size, enc_hid_size)
        s0, srcs, uh = self.init(srcs, srcs_m, False)

        return self.decoder(s0, srcs, trgs, uh, srcs_m, trgs_m)


class Encoder(nn.Module):

    '''
        Bi-directional Gated Recurrent Unit network encoder
    '''

    def __init__(self,
                 src_vocab_size,
                 input_size,
                 output_size,
                 bidirectional=False,
                 with_ln=False,
                 prefix='Encoder', **kwargs):

        super(Encoder, self).__init__()

        self.output_size = output_size
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        self.src_lookup_table = nn.Embedding(src_vocab_size, wargs.src_wemb_size, padding_idx=PAD)

        if wargs.enc_rnn_type == 'gru':
            self.forw_gru = GRU(input_size, output_size, with_ln=with_ln, prefix=f('Forw'))
            self.back_gru = GRU(output_size, output_size, with_ln=with_ln, prefix=f('Back'))
        elif wargs.enc_rnn_type == 'sru':
            self.rnn = SRU(
                    input_size=input_size,
                    hidden_size=output_size,
                    num_layers=wargs.enc_layer_cnt,
                    dropout=wargs.drop_rate,
                    bidirectional=bidirectional)

    def forward(self, xs, xs_mask=None, h0=None):

        max_L, b_size = xs.size(0), xs.size(1)
        xs_e = xs if xs.dim() == 3 else self.src_lookup_table(xs)

        if wargs.enc_rnn_type == 'sru': return self.rnn(xs_e, h0)

        right = []
        h = h0 if h0 else Variable(tc.zeros(b_size, self.output_size), requires_grad=False)
        if wargs.gpu_id: h = h.cuda()
        for k in range(max_L):
            # (batch_size, src_wemb_size)
            h = self.forw_gru(xs_e[k], xs_mask[k] if xs_mask is not None else None, h)
            right.append(h)

        left = []
        h = h0 if h0 else Variable(tc.zeros(b_size, self.output_size), requires_grad=False)
        if wargs.gpu_id: h = h.cuda()
        for k in reversed(range(max_L)):
            h = self.back_gru(right[k], xs_mask[k] if xs_mask is not None else None, h)
            left.append(h)

        return tc.stack(left[::-1], dim=0)

class Attention(nn.Module):

    def __init__(self, dec_hid_size, align_size):

        super(Attention, self).__init__()
        self.align_size = align_size
        self.sa = nn.Linear(dec_hid_size, self.align_size)
        self.tanh = nn.Tanh()
        self.a1 = nn.Linear(self.align_size, 1)

    # support for 3-D input
    def forward(self, s_tm1, xs_h, uh, xs_mask=None):

        '''
        d1, d2, d3 = uh.size()
        # (b, dec_hid_size) -> (b, aln) -> (1, b, aln) -> (slen, b, aln) -> (slen, b)
        e_ij = self.a1(
            self.tanh(self.sa(s_tm1)[None, :, :] + uh)).squeeze(2).exp()
        if xs_mask is not None: e_ij = e_ij * xs_mask

        # probability in each column: (slen, b)
        e_ij = e_ij / e_ij.sum(0)[None, :]

        # weighted sum of the h_j: (b, enc_hid_size)
        attend = (e_ij[:, :, None] * xs_h).sum(0)
        '''

        if s_tm1.dim() == 2: s_tm1 = s_tm1[None, :, :]

        # (tL,b,dec_hid_size)->(tL,b,aln)->(1,tL,b,aln)->(sL,tL,b,aln)->(sL,tL,b)
        e_ij = self.a1(
            self.tanh(self.sa(s_tm1)[None,:,:,:] + uh[:,None,:,:])).squeeze(-1).exp()
        #print e_ij.size()
        #print xs_mask.size()
        #print xs_mask[:,None,:].size()
        if xs_mask is not None: e_ij = e_ij * xs_mask[:,None,:]
        # probability in each column: (sL, tL, b)
        e_ij = e_ij / e_ij.sum(0)[None, :, :]
        # weighted sum of the h_j: (tL, b, enc_hid_size)
        attend = (e_ij[:, :, :, None] * xs_h[:,None,:,:]).sum(0)

        e_ij, attend = e_ij.squeeze(1), attend.squeeze(0)

        # (sL, b), (b, enc_hid_size) or (sL, tL, b), (tL, b, enc_hid_size)
        return e_ij, attend

class Decoder(nn.Module):

    def __init__(self, trg_vocab_size, with_ln=False, max_out=True):

        super(Decoder, self).__init__()

        self.max_out = max_out
        self.attention = Attention(wargs.dec_hid_size, wargs.align_size)
        self.trg_lookup_table = nn.Embedding(trg_vocab_size, wargs.trg_wemb_size, padding_idx=PAD)
        self.tanh = nn.Tanh()

        if wargs.dec_rnn_type == 'gru':
            self.gru1 = GRU(wargs.trg_wemb_size, wargs.dec_hid_size, with_ln=with_ln)
            self.gru2 = GRU(wargs.enc_hid_size, wargs.dec_hid_size, with_ln=with_ln)
        elif wargs.dec_rnn_type == 'sru':
            self.gru1 = SRU(input_size=wargs.trg_wemb_size, hidden_size=wargs.dec_hid_size,
                    num_layers=wargs.dec_layer_cnt, dropout=0., bidirectional=False)
            self.gru2 = SRU(input_size=2*wargs.enc_hid_size, hidden_size=wargs.dec_hid_size,
                    num_layers=wargs.dec_layer_cnt, dropout=0., bidirectional=False)

        out_size = 2 * wargs.out_size if max_out else wargs.out_size
        self.ls = nn.Linear(wargs.dec_hid_size, out_size)
        self.ly = nn.Linear(wargs.trg_wemb_size, out_size)
        self.lc = nn.Linear(2*wargs.enc_hid_size, out_size)

    def fwd_gru(self, s_tm1, xs_h, ys, uh, xs_mask=None, ys_mask=None):

        tlen_batch_s, tlen_batch_c = [], []
        y_Lm1, b_size = ys.size(0), ys.size(1)
        # (max_tlen_batch - 1, batch_size, trg_wemb_size)
        ys_e = ys if ys.dim() == 3 else self.trg_lookup_table(ys)
        s_above, hidden_t = self.gru1(ys_e, s_tm1)
        for k in range(y_Lm1):
            attend, s_tm1, _ = self.step(s_tm1, xs_h, uh, ys_e[k],
                                         xs_mask if xs_mask is not None else None,
                                         ys_mask[k] if ys_mask is not None else None)
            tlen_batch_c.append(attend)
            tlen_batch_s.append(s_tm1)

        s = tc.stack(tlen_batch_s, dim=0)
        c = tc.stack(tlen_batch_c, dim=0)
        del tlen_batch_s, tlen_batch_c

        logit = self.step_out(s, ys_e, c)
        if ys_mask is not None: logit = logit * ys_mask[:, :, None]  # !!!!
        del s, c

        return logit

    # s_tm1: (b, dec_hid_size)
    def step_sru(self, s_tm1, xs_h, uh, y_tm1, xs_mask=None, y_mask=None):

        if not isinstance(y_tm1, tc.autograd.variable.Variable):
            if isinstance(y_tm1, int): y_tm1 = tc.Tensor([y_tm1]).long()
            elif isinstance(y_tm1, list): y_tm1 = tc.Tensor(y_tm1).long()
            if wargs.gpu_id: y_tm1 = y_tm1.cuda()
            y_tm1 = Variable(y_tm1, requires_grad=False, volatile=True)
            y_tm1 = self.trg_lookup_table(y_tm1)

        #print 'y_tm1: ', y_tm1.size()
        #s_tm1 = tc.stack(wargs.dec_layer_cnt * [s_tm1], 0)
        #print 's_tm1: ', s_tm1.size()
        #print s_tm1.size()
        # input: (len, batch, n_in), c0: (depth, batch, n_out)
        s_above, hidden_t = self.gru1(y_tm1[None,:,:], s_tm1)
        # context, enc_hidden = 
        #print 's_above: ', s_above.size()
        #print 'hidden_t: ', hidden_t.size()
        #s_above = s_above[0]
        # (slen, batch_size) (batch_size, enc_hid_size)
        alpha_ij, attend = self.attention(s_above, xs_h, uh, xs_mask)
        #print 'attend: ', attend.size()
        #s_above = tc.stack(wargs.dec_layer_cnt * [s_above[-1]], 0)
        #print 's_above: ', s_above.size()
        s_t, hidden_t = self.gru2(attend[None,:,:], hidden_t)
        #print 's_t: ', s_t.size()
        #print 'hidden_t: ', hidden_t.size()

        del alpha_ij
        # s_t: (1, 1, 512)
        return attend, s_t[0], y_tm1, hidden_t

    def forward(self, s_tm1, xs_h, ys, uh, xs_mask=None, ys_mask=None):

        if wargs.dec_rnn_type == 'gru': logit = self.fwd_gru(s_tm1, xs_h, ys, uh, xs_mask, ys_mask)
        elif wargs.dec_rnn_type == 'sru':

            tlen_batch_s, tlen_batch_c = [], []
            y_Lm1, b_size = ys.size(0), ys.size(1)
            # (max_tlen_batch - 1, batch_size, trg_wemb_size)
            ys_e = ys if ys.dim() == 3 else self.trg_lookup_table(ys)
            for k in range(y_Lm1):
                a_t, s_t, _, s_tm1 = self.step_sru(s_tm1, xs_h, uh, ys_e[k],
                                           xs_mask if xs_mask is not None else None,
                                           ys_mask[k] if ys_mask is not None else None)
                tlen_batch_c.append(a_t)
                tlen_batch_s.append(s_t)

            s = tc.stack(tlen_batch_s, dim=0)
            c = tc.stack(tlen_batch_c, dim=0)
            del tlen_batch_s, tlen_batch_c
            logit = self.step_out(s, ys_e, c)
            if ys_mask is not None: logit = logit * ys_mask[:, :, None]  # !!!!

        return logit

    def step_out(self, s, y, c):

        # (max_tlen_batch - 1, batch_size, dec_hid_size)
        logit = self.ls(s) + self.ly(y) + self.lc(c)
        # (max_tlen_batch - 1, batch_size, out_size)

        if logit.dim() == 2:    # for decoding
            logit = logit.view(logit.size(0), logit.size(1)/2, 2)
        elif logit.dim() == 3:
            logit = logit.view(logit.size(0), logit.size(1), logit.size(2)/2, 2)

        return logit.max(-1)[0] if self.max_out else self.tanh(logit)

