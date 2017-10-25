import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import wargs
from ran import RAN
from tools.utils import *

class NMT(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size):

        super(NMT, self).__init__()

        self.encoder = Encoder(src_vocab_size, wargs.src_wemb_size, wargs.enc_hid_size)
        self.s_init = nn.Linear(wargs.enc_hid_size, wargs.dec_hid_size)
        self.tanh = nn.Tanh()
        self.ha = nn.Linear(wargs.enc_hid_size, wargs.align_size)
        self.decoder = Decoder(trg_vocab_size)

    def init_state(self, xs_h, xs_mask=None):

        assert xs_h.dim() == 3  # slen, batch_size, enc_size
        if xs_mask is not None:
            x = (xs_h * xs_mask.unsqueeze(dim=2).expand_as(xs_h)).sum(0)[0]
            mean_enc = x / xs_mask.sum(0)[0].unsqueeze(-1).expand_as(x)
        else:
            mean_enc = xs_h.mean(0)[0]

        return self.tanh(self.s_init(mean_enc))

    def gen_uh(self, xs_h):

        d1, d2, d3 = xs_h.size()
        return self.ha(xs_h.view(-1, d3)).view(d1, d2, wargs.align_size)

    def init(self, xs, xs_mask=None, test=True):

        if test:  # for decoding
            xs = Variable(xs, requires_grad=False)
            if wargs.gpu_id and not xs.is_cuda: xs = xs.cuda()

        xs = self.encoder(xs, xs_mask)
        s0 = self.init_state(xs, xs_mask)
        uh = self.gen_uh(xs)
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
                 with_ln=False,
                 prefix='Encoder', **kwargs):

        super(Encoder, self).__init__()

        self.output_size = output_size
        self.laycnt = wargs.enc_layer_cnt
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        self.src_lookup_table = nn.Embedding(src_vocab_size, wargs.src_wemb_size, padding_idx=PAD)

        self.forw_ran = RAN(input_size, output_size,
                            residual=True, with_ln=with_ln, prefix=f('Forw'))
        self.back_ran = RAN(output_size, output_size,
                            residual=True, with_ln=with_ln, prefix=f('Back'))

    def snaker(self, feed, c=None, xs_mask=None):

        first = []
        max_L = feed.size(0)
        for k in range(max_L):
            # (batch_size, src_wemb_size)
            c, h = self.forw_ran(feed[k], xs_mask[k] if xs_mask else None, c)
            first.append(h)

        second = []
        #c = init_state if init_state else Variable(
        #    tc.zeros(this_batch_size, self.output_size), requires_grad=False)
        #if wargs.gpu_id:
        #    c = c.cuda()
        for k in reversed(range(max_L)):
            c, h = self.back_ran(first[k], xs_mask[k] if xs_mask else None, c)
            second.append(h)

        return tc.stack(tuple(second[::-1]), dim=0), c

    def forward(self, xs, xs_mask=None, h0=None):

        b_size = xs.size(1)
        x_es = xs if xs.dim() == 3 else self.src_lookup_table(xs)

        c0 = h0 if h0 else Variable(tc.zeros(b_size, self.output_size), requires_grad=False)
        if wargs.gpu_id: c0 = c0.cuda()
        c = [c0]
        for layer in range(self.laycnt):
            x_es, c = self.snaker(x_es, c, xs_mask)

        return x_es

class Attention(nn.Module):

    def __init__(self, dec_hid_size, align_size):

        super(Attention, self).__init__()
        self.align_size = align_size
        self.sa = nn.Linear(dec_hid_size, self.align_size)
        self.ha = nn.Linear(dec_hid_size, self.align_size)
        self.tanh = nn.Tanh()
        #self.a1 = nn.Linear(self.align_size, 1)
        self.ran = RAN(wargs.dec_hid_size, wargs.dec_hid_size, residual=False)

    def forward0(self, s_tm1, xs_h, uh, xs_mask=None):

        d1, d2, d3 = uh.size()
        # (b, dec_hid_size) -> (b, aln) -> (1, b, aln) -> (slen, b, aln) -> (slen, b)
        e_ij = self.a1(
            self.tanh(self.sa(s_tm1).unsqueeze(0).expand_as(uh) + uh).view(-1, d3)).view(
                d1, d2, -1).squeeze(2).exp()
        if xs_mask is not None: e_ij = e_ij * xs_mask

        # probability in each column: (slen, b)
        e_ij = e_ij / e_ij.sum(0).expand_as(e_ij)

        # weighted sum of the h_j: (b, enc_hid_size)
        attend = (e_ij.unsqueeze(-1).expand_as(xs_h) * xs_h).sum(0).squeeze(0)

        return e_ij, attend

    def forward(self, s_tm1, xs_h, c_tm1, xs_mask=None, y_mask=None):

        d1, d2, d3 = uh.size()
        # (b, dec_hid_size) -> (b, aln) -> (1, b, aln) -> (slen, b, aln) -> (slen, b)
        ws = self.sa(s_tm1)
        self.ran(s_tm1, y_mask, )
        e_ij = self.a1(
            self.tanh(self.sa(s_tm1).unsqueeze(0).expand_as(uh) + uh).view(-1, d3)).view(
                d1, d2, -1).squeeze(2).exp()
        if xs_mask is not None: e_ij = e_ij * xs_mask

        # probability in each column: (slen, b)
        e_ij = e_ij / e_ij.sum(0).expand_as(e_ij)

        # weighted sum of the h_j: (b, enc_hid_size)
        attend = (e_ij.unsqueeze(-1).expand_as(xs_h) * xs_h).sum(0).squeeze(0)

        return e_ij, attend

class Decoder(nn.Module):

    def __init__(self, trg_vocab_size, max_out=True):

        super(Decoder, self).__init__()

        self.laycnt = wargs.dec_layer_cnt - 1
        self.max_out = max_out
        self.trg_lookup_table = nn.Embedding(trg_vocab_size, wargs.trg_wemb_size, padding_idx=PAD)
        self.attention = Attention(wargs.dec_hid_size, wargs.align_size)

        self.tanh = nn.Tanh()
        self.ran1 = RAN(wargs.trg_wemb_size, wargs.dec_hid_size, residual=False)
        self.ran2 = RAN(wargs.dec_hid_size + wargs.enc_hid_size,
                        wargs.dec_hid_size, residual=True, laycnt=self.laycnt, share_weight=False)

        out_size = 2 * wargs.out_size if max_out else wargs.out_size
        self.ls = nn.Linear(wargs.dec_hid_size, out_size)

        self.dropout = nn.Dropout(wargs.drop_rate)

    def step(self, c_tm1, xs_h, uh, y_tm1, xs_mask=None, y_mask=None):

        if not isinstance(y_tm1, tc.autograd.variable.Variable):
            if isinstance(y_tm1, int):
                y_tm1 = tc.Tensor([y_tm1]).long()
            elif isinstance(y_tm1, list):
                y_tm1 = tc.Tensor(y_tm1).long()
            if wargs.gpu_id: y_tm1 = y_tm1.cuda()
            y_tm1 = Variable(y_tm1, requires_grad=False, volatile=True)
            y_tm1 = self.trg_lookup_table(y_tm1)

        c1 = [c_tm1[0]]
        c1, s_above = self.ran1(y_tm1, y_mask, c1)
        # (slen, batch_size) (batch_size, enc_hid_size)
        alpha_ij, attend = self.attention(s_above, xs_h, uh, xs_mask)
        c2 = c_tm1[1:]
        c2, s_t = self.ran2(s_above, y_mask, c2, attend)

        c_t = c1 + c2
        return c_t, s_t

    def forward(self, s0, xs_h, ys, uh, xs_mask=None, ys_mask=None):

        y_Lm1, b_size = ys.size(0), ys.size(1)
        # (max_tlen_batch - 1, batch_size, trg_wemb_size)
        ys_e = self.trg_lookup_table(ys)
        c = [s0]
        tlen_batch_a, tlen_batch_h = [], []
        for k in range(y_Lm1):
            # (batch_size, trg_wemb_size)
            c, s_above = self.ran1(ys_e[k], ys_mask[k] if ys_mask else None, c)
            # (slen, batch_size) (batch_size, enc_hid_size)
            alpha_ij, attend = self.attention(s_above, xs_h, c, xs_mask)
            tlen_batch_a.append(attend)
            tlen_batch_h.append(s_above)

        c = [s0] * self.laycnt
        tlen_batch_s = []
        for k in range(y_Lm1):
            c, s_t = self.ran2(tlen_batch_h[k],
                               ys_mask[k] if ys_mask else None, c, tlen_batch_a[k])
            tlen_batch_s.append(s_t)

        s = tc.stack(tuple(tlen_batch_s), dim=0)

        logit = self.step_out(s, ys_e, c)
        del tlen_batch_a, tlen_batch_h, tlen_batch_s

        if ys_mask is not None: logit = logit * ys_mask[:, :, None]  # !!!!
        del s, c, s_above, s_t

        return self.dropout(logit)

    def step_out(self, s):
        # (max_tlen_batch - 1, batch_size, dec_hid_size)
        # (batch_size, dec_hid_size), no dropout in decoding
        logit = self.ls(s)
        # (max_tlen_batch - 1, batch_size, out_size)

        if logit.dim() == 2:    # for decoding
            logit = logit.view(logit.size(0), logit.size(1)/2, 2)
        elif logit.dim() == 3:
            logit = logit.view(logit.size(0), logit.size(1), logit.size(2)/2, 2)

        return logit.max(-1)[0] if self.max_out else self.tanh(logit)

