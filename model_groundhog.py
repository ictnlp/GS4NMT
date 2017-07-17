from gru import GRU
from utils import *
import torch.nn as nn
import wargs
import const
import torch.nn.functional as F
import torch as tc
from torch.autograd import Variable


class NMT(nn.Module):

    def __init__(self):

        super(NMT, self).__init__()

        self.encoder = Encoder(wargs.src_wemb_size, wargs.enc_hid_size)
        self.s_init = nn.Linear(wargs.enc_hid_size, wargs.dec_hid_size)
        self.tanh = nn.Tanh()
        self.ha = nn.Linear(2 * wargs.enc_hid_size, wargs.align_size)
        self.decoder = Decoder()

    def init_state(self, h0_left):

        return self.tanh(self.s_init(h0_left))

    def gen_uh(self, xs_h):

        d1, d2, d3 = xs_h.size()
        return self.ha(xs_h.view(-1, d3)).view(d1, d2, wargs.align_size)

    def init(self, xs, xs_mask=None, test=True):

        if test:  # for decoding
            if wargs.gpu_id and not xs.is_cuda: xs = xs.cuda()
            xs = Variable(xs, requires_grad=False, volatile=True)

        xs, h0_left = self.encoder(xs, xs_mask)
        s0 = self.init_state(h0_left)
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
                 input_size,
                 output_size,
                 with_ln=False,
                 prefix='Encoder', **kwargs):

        super(Encoder, self).__init__()

        self.output_size = output_size
        f = lambda name: str_cat(prefix, name)  # return 'Encoder_' + parameters name

        self.src_lookup_table = nn.Embedding(wargs.src_dict_size + 4,
                                             wargs.src_wemb_size, padding_idx=const.PAD)

        self.forw_gru = GRU(input_size, output_size, with_ln=with_ln, prefix=f('Forw'))
        self.back_gru = GRU(input_size, output_size, with_ln=with_ln, prefix=f('Back'))

    def forward(self, xs, xs_mask=None, h0=None):

        max_L, b_size = xs.size(0), xs.size(1)
        xs_e = xs if xs.dim() == 3 else self.src_lookup_table(xs)

        right = []
        h = h0 if h0 else Variable(tc.zeros(b_size, self.output_size), requires_grad=False)
        if wargs.gpu_id: h = h.cuda()
        for k in range(max_L):
            # (batch_size, src_wemb_size)
            h = self.forw_gru(xs_e[k], xs_mask[k] if xs_mask else None, h)
            right.append(h)

        left = []
        h = h0 if h0 else Variable(tc.zeros(b_size, self.output_size), requires_grad=False)
        if wargs.gpu_id: h = h.cuda()
        for k in reversed(range(max_L)):
            h = self.back_gru(xs_e[k], xs_mask[k] if xs_mask else None, h)
            left.append(h)

        right = tc.stack(tuple(right))
        reverse_left = tc.stack(tuple(left[::-1]))
        # (slen, batch_size, 2*output_size)
        r1, r2 = tc.cat([right, reverse_left], -1), left[-1]

        del right, left[:], h, reverse_left
        return r1, r2

class Attention(nn.Module):

    def __init__(self, dec_hid_size, align_size):

        super(Attention, self).__init__()
        self.align_size = align_size
        self.sa = nn.Linear(dec_hid_size, self.align_size)
        self.tanh = nn.Tanh()
        self.a1 = nn.Linear(self.align_size, 1)

    def forward(self, s_tm1, xs_h, uh, xs_mask=None):

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

class Decoder(nn.Module):

    def __init__(self, max_out=True):

        super(Decoder, self).__init__()

        self.max_out = max_out
        self.attention = Attention(wargs.dec_hid_size, wargs.align_size)
        self.trg_lookup_table = nn.Embedding(wargs.trg_dict_size + 4,
                                             wargs.trg_wemb_size, padding_idx=const.PAD)
        self.tanh = nn.Tanh()
        self.gru = GRU(wargs.trg_wemb_size, wargs.dec_hid_size, enc_hid_size=wargs.enc_hid_size)

        out_size = 2 * wargs.out_size if max_out else wargs.out_size
        self.ls = nn.Linear(wargs.dec_hid_size, out_size)
        self.ly = nn.Linear(wargs.trg_wemb_size, out_size)
        self.lc = nn.Linear(2 * wargs.enc_hid_size, out_size)

    def step(self, s_tm1, xs_h, uh, y_tm1, xs_mask=None, y_mask=None):

        if not isinstance(y_tm1, tc.autograd.variable.Variable):
            if isinstance(y_tm1, int):
                y_tm1 = tc.Tensor([y_tm1]).long()
            elif isinstance(y_tm1, list):
                y_tm1 = tc.Tensor(y_tm1).long()
            if wargs.gpu_id: y_tm1 = y_tm1.cuda()
            y_tm1 = Variable(y_tm1, requires_grad=False, volatile=True)
            y_tm1 = self.trg_lookup_table(y_tm1)

        # (slen, batch_size), (batch_size, enc_hid_size)
        alpha_ij, attend = self.attention(s_tm1, xs_h, uh, xs_mask)
        s_t = self.gru(y_tm1, y_mask, s_tm1, attend)
        del alpha_ij
        return attend, s_t, y_tm1

    def forward(self, s_tm1, xs_h, ys, uh, xs_mask=None, ys_mask=None):

        tlen_batch_s, tlen_batch_c = [], []
        y_Lm1, b_size = ys.size(0), ys.size(1)
        # (max_tlen_batch - 1, batch_size, trg_wemb_size)
        ys_e = ys if ys.dim() == 3 else self.trg_lookup_table(ys)
        for k in range(y_Lm1):
            attend, s_tm1, _ = self.step(s_tm1, xs_h, uh, ys_e[k],
                                         xs_mask if xs_mask else None,
                                         ys_mask[k] if ys_mask else None)
            tlen_batch_c.append(attend)
            tlen_batch_s.append(s_tm1)

        s = tc.stack(tuple(tlen_batch_s), dim=0)
        c = tc.stack(tuple(tlen_batch_c), dim=0)
        del tlen_batch_s, tlen_batch_c

        # (max_tlen_batch - 1, batch_size, dec_hid_size)
        logit = self.ls(s.view(-1, s.size(-1))) + \
                self.ly(ys_e.view(-1, ys_e.size(-1))) + \
                self.lc(c.view(-1, c.size(-1)))
        logit = logit.view(y_Lm1, b_size, -1)

        # (max_tlen_batch - 1, batch_size, out_size)
        logit = logit.view(y_Lm1, b_size, logit.size(2)/2, 2).max(-1)[0].squeeze(-1) if \
                self.max_out else self.tanh(logit)

        if ys_mask: logit = logit * ys_mask.unsqueeze(-1).expand_as(logit)  # !!!!
        del s, c
        return logit

    def logit(self, s, y, c):
        # (batch_size, dec_hid_size)
        logit = self.ls(s) + self.ly(y) + self.lc(c)

        return logit.view(logit.size(0), logit.size(1)/2, 2).max(-1)[0].squeeze(-1) if \
                self.max_out else self.tanh(logit)

class Classifier(nn.Module):

    def __init__(self, input_size, output_size):

        super(Classifier, self).__init__()

        self.dropout = nn.Dropout(wargs.drop_rate)
        self.map_vocab = nn.Linear(input_size, output_size)
        self.log_prob = nn.LogSoftmax()

        weight = tc.ones(output_size)
        weight[const.PAD] = 0   # do not predict padding
        self.criterion = nn.NLLLoss(weight, size_average=False)
        if wargs.gpu_id: self.criterion.cuda()

    def nll_loss(self, flat_vocab, gold, gold_mask):

        flat_logp = self.log_prob(flat_vocab)
        flat_logp = flat_logp * gold_mask.unsqueeze(-1).expand_as(flat_logp)
        nll = self.criterion(flat_logp, gold)

        return nll

    def forward(self, feed, gold=None, gold_mask=None):

        # no dropout in decoding
        feed = self.dropout(feed) if gold else feed
        # (max_tlen_batch - 1, batch_size, out_size)
        pred_vocab = self.map_vocab(feed.contiguous().view(-1, feed.size(-1)))

        # decoding, if gold is None and gold_mask is None:
        if gold is None: return -self.log_prob(pred_vocab)

        if gold.dim() == 2: gold, gold_mask = gold.view(-1), gold_mask.view(-1)
        # (max_tlen_batch - 1, batch_size, trg_vocab_size)
        pred_correct = (pred_vocab.max(dim=-1)[1].squeeze()).eq(
            gold).masked_select(gold.ne(const.PAD)).sum()

        # negative likelihood log
        nll = self.nll_loss(pred_vocab, gold, gold_mask)

        # total loss,  correct count in one batch
        return nll, pred_correct


