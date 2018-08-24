import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import wargs
from .gru import GRU
from tools.utils import *
from models.losser import *

EOS = 3

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
            xs_h = (xs_h * xs_mask[:, :, None]).sum(0) / xs_mask.sum(0)[:, None]
        else:
            xs_h = xs_h.mean(0)

        return self.tanh(self.s_init(xs_h))

    def init(self, xs, xs_mask=None, test=True):

        if test is True and not isinstance(xs, tc.autograd.variable.Variable):  # for decoding
            if wargs.gpu_id and not xs.is_cuda: xs = xs.cuda()
            xs = Variable(xs, requires_grad=False, volatile=True)

        xs = self.encoder(xs, xs_mask)
        s0 = self.init_state(xs, xs_mask)
        uh = self.ha(xs)
        return s0, xs, uh

    def forward(self, srcs, trgs, srcs_m, trgs_m, isAtt=False, test=False):
        # (max_slen_batch, batch_size, enc_hid_size)
        s0, srcs, uh = self.init(srcs, srcs_m, test)
        if wargs.feed_previous == True:
            return self.decoder.fp_forward(s0, srcs, trgs, uh, srcs_m, trgs_m, isAtt=isAtt)
        else:
            return self.decoder(s0, srcs, trgs, uh, srcs_m, trgs_m, isAtt=isAtt)

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
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        self.src_lookup_table = nn.Embedding(src_vocab_size, wargs.src_wemb_size, padding_idx=PAD)

        self.forw_gru = GRU(input_size, output_size, with_ln=with_ln, prefix=f('Forw'))
        self.back_gru = GRU(output_size, output_size, with_ln=with_ln, prefix=f('Back'))

    def forward(self, xs, xs_mask=None, h0=None):

        max_L, b_size = xs.size(0), xs.size(1)
        xs_e = xs if xs.dim() == 3 else self.src_lookup_table(xs)

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
        self.maskSoftmax = MaskSoftmax()
        self.a1 = nn.Linear(self.align_size, 1)

    def forward(self, s_tm1, xs_h, uh, xs_mask=None):

        d1, d2, d3 = uh.size()
        _check_a1 = self.a1(self.tanh(self.sa(s_tm1)[None, :, :] + uh)).squeeze(2)
        e_ij = self.maskSoftmax(_check_a1, mask=xs_mask, dim=0)
        # weighted sum of the h_j: (b, enc_hid_size)
        attend = (e_ij[:, :, None] * xs_h).sum(0)

        return e_ij, attend

class Decoder(nn.Module):

    def __init__(self, trg_vocab_size, max_out=True):

        super(Decoder, self).__init__()

        self.max_out = max_out
        self.attention = Attention(wargs.dec_hid_size, wargs.align_size)
        self.trg_lookup_table = nn.Embedding(trg_vocab_size, wargs.trg_wemb_size, padding_idx=PAD)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.gru1 = GRU(wargs.trg_wemb_size, wargs.dec_hid_size)
        self.gru2 = GRU(wargs.enc_hid_size, wargs.dec_hid_size)

        out_size = 2 * wargs.out_size if max_out else wargs.out_size
        self.ls = nn.Linear(wargs.dec_hid_size, out_size)
        self.ly = nn.Linear(wargs.trg_wemb_size, out_size)
        self.lc = nn.Linear(wargs.enc_hid_size, out_size)

        self.classifier = Classifier(wargs.out_size, trg_vocab_size,
                                     self.trg_lookup_table if wargs.copy_trg_emb is True else None)

    def step(self, s_tm1, xs_h, uh, y_tm1, xs_mask=None, y_mask=None):

        if not isinstance(y_tm1, tc.autograd.variable.Variable):
            if isinstance(y_tm1, int): y_tm1 = tc.Tensor([y_tm1]).long()
            elif isinstance(y_tm1, list): y_tm1 = tc.Tensor(y_tm1).long()
            if wargs.gpu_id: y_tm1 = y_tm1.cuda()
            y_tm1 = Variable(y_tm1, requires_grad=False, volatile=True)
            y_tm1 = self.trg_lookup_table(y_tm1)

        if xs_mask is not None and not isinstance(xs_mask, tc.autograd.variable.Variable):
            xs_mask = Variable(xs_mask, requires_grad=False, volatile=True)
            if wargs.gpu_id: xs_mask = xs_mask.cuda()

        s_above = self.gru1(y_tm1, y_mask, s_tm1)
        # (slen, batch_size) (batch_size, enc_hid_size)
        alpha_ij, attend = self.attention(s_above, xs_h, uh, xs_mask)
        s_t = self.gru2(attend, y_mask, s_above)

        return attend, s_t, y_tm1, alpha_ij

    def forward(self, s_tm1, xs_h, ys, uh, xs_mask, ys_mask, isAtt=False, ss_eps=1.):

        tlen_batch_s, tlen_batch_y, tlen_batch_c = [], [], []
        y_Lm1, b_size = ys.size(0), ys.size(1)
        assert (xs_mask is not None) and (ys_mask is not None)

        if isAtt is True: attends = []
        # (max_tlen_batch - 1, batch_size, trg_wemb_size)
        ys_e = ys if ys.dim() == 3 else self.trg_lookup_table(ys)

        for k in range(y_Lm1):

            y_tm1 = ys_e[k]
            attend, s_tm1, _, alpha_ij = \
                    self.step(s_tm1, xs_h, uh, y_tm1, xs_mask, ys_mask[k])

            tlen_batch_c.append(attend)
            tlen_batch_y.append(y_tm1)
            tlen_batch_s.append(s_tm1)

            if isAtt is True: attends.append(alpha_ij)

        s = tc.stack(tlen_batch_s, dim=0)
        y = tc.stack(tlen_batch_y, dim=0)
        c = tc.stack(tlen_batch_c, dim=0)
        del tlen_batch_s, tlen_batch_y, tlen_batch_c

        logit = self.step_out(s, y, c)
        logit = logit * ys_mask[:, :, None]  # !!!!
        del s, y, c
        results = (logit, tc.stack(attends, 0)) if isAtt is True else logit

        return results

    def fp_forward(self, s_tm1, xs_h, ys, uh, xs_mask, ys_mask, isAtt=False, ss_eps=1.):

        tlen_batch_s, tlen_batch_y, tlen_batch_c, tlen_batch_m = [], [], [], []
        b_size = ys.size(1)
        assert (xs_mask is not None) and (ys_mask is not None)

        if isAtt is True: attends = []
        # (max_tlen_batch - 1, batch_size, trg_wemb_size)
        ys_e = ys if ys.dim() == 3 else self.trg_lookup_table(ys)
        y_tm1 = ys_e[0]
        y_mask = Variable(tc.FloatTensor([1] * b_size).cuda())
        mask_next_time = [False] * b_size
 
        for k in range(wargs.max_seq_len):

            attend, s_tm1, _, alpha_ij = \
                    self.step(s_tm1, xs_h, uh, y_tm1, xs_mask, y_mask)

            tlen_batch_c.append(attend)
            tlen_batch_y.append(y_tm1)
            tlen_batch_s.append(s_tm1)
            tlen_batch_m.append(y_mask)
            if isAtt is True: attends.append(alpha_ij)

            logit = self.step_out(s_tm1, y_tm1, attend)
            prob = self.classifier.softmax(self.classifier.get_a(logit))
            next_ces = tc.max(prob,1)[1]
            y_tm1 = self.trg_lookup_table(next_ces)

            tmp_y_mask = y_mask.data.tolist()

            for i in range(b_size):
                if (tmp_y_mask[i] > 0.5) and (mask_next_time[i] == False):
                    tmp_y_mask[i] = 1.0
                else:
                    tmp_y_mask[i] = 0.0
            for i in range(b_size):
                if (next_ces.data[i] == EOS):
                    mask_next_time[i] = True
                else:
                    mask_next_time[i] = False
            reach_end = True
            for i in range(b_size):
                if (tmp_y_mask[i] > 0.5):
                    reach_end = False
            y_mask = Variable(tc.FloatTensor(tmp_y_mask).cuda())

            if reach_end == True:
                break


        s = tc.stack(tlen_batch_s, dim=0)
        y = tc.stack(tlen_batch_y, dim=0)
        c = tc.stack(tlen_batch_c, dim=0)
        m = tc.stack(tlen_batch_m, dim=0)
        del tlen_batch_s, tlen_batch_y, tlen_batch_c, tlen_batch_m

        logit = self.step_out(s, y, c)
        logit = logit * m[:, :, None]  # !!!!
        del s, y, c
        results = (logit, tc.stack(attends, 0)) if isAtt is True else logit

        return results, m


    def step_out(self, s, y, c):

        # (max_tlen_batch - 1, batch_size, dec_hid_size)
        logit = self.ls(s) + self.ly(y) + self.lc(c)
        # (max_tlen_batch - 1, batch_size, out_size)

        if logit.dim() == 2:    # for decoding
            logit = logit.view(logit.size(0), logit.size(1)/2, 2)
        elif logit.dim() == 3:
            logit = logit.view(logit.size(0), logit.size(1), logit.size(2)/2, 2)

        return logit.max(-1)[0] if self.max_out else self.tanh(logit)


