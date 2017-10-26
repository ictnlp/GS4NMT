import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import wargs
from gru import GRU
from tools.utils import *

class Cyknet(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(Cyknet, self).__init__()

        self.output_dim = output_dim
        self.fwz = wargs.filter_window_size
        self.ffs = wargs.filter_feats_size

        for i in range(len(self.fwz)):
            self.l_f1 = nn.Linear(input_dim, output_dim)
            self.l_conv = nn.Conv1d(1, self.ffs[i], kernel_size=output_dim*self.fwz[i], stride=output_dim)
            self.l_f2 = nn.Linear(self.ffs[i], output_dim)
            #self.r_f1 = nn.Linear(input_dim, output_dim)
            #self.r_conv = nn.Conv1d(1, self.ffs[i], kernel_size=output_dim*self.fwz[i], stride=output_dim)
            #self.r_f2 = nn.Linear(self.ffs[i], output_dim)

    def cyk_mask(self, x_maxL, xs_mask=None):

        if xs_mask is None:
            return Variable(tc.triu(tc.ones(x_maxL, x_maxL))[:, :, None], requires_grad=False)
        assert xs_mask.dim() == 2
        x_maxL, B = xs_mask.size(0), xs_mask.size(1)

        cykmask = []
        for bid in range(B):
            #mask = tc.diag(xs_mask.data[:, bid], diagonal=0)
            length = int(xs_mask.data[:, bid].sum())
            ta = tc.triu(tc.ones(x_maxL, x_maxL))
            if length == x_maxL:
                cykmask.append(ta)
                continue
            #print ta
            #print ta.size(), length
            ta[:, length:].copy_(tc.zeros(x_maxL, x_maxL - length))
            cykmask.append(ta)

        # (L, L, B)
        return Variable(tc.stack(cykmask).permute(1, 2, 0), requires_grad=False)

    def forward(self, xs_h, xs_mask=None):

        x_maxL, B = xs_h.size(0), xs_h.size(1)
        assert xs_h.dim() == 3

        # input: length, batch, dim
        # c0: layers*(2), batch, dim
        #self.l_table, self.r_table = [None] * x_maxL, [None] * x_maxL
        #for k in range(1, x_maxL + 1):
            #self.l_table[k - 1], self.r_table[k - 1] = [None] * (x_maxL-k+2), [None] * (x_maxL-k+2)
        #    self.l_table[k - 1], self.r_table[k - 1] = [None] * (x_maxL+1), [None] * (x_maxL+1)
        l_table = Variable(tc.zeros(x_maxL, x_maxL, B, self.output_dim), requires_grad=True)
        #r_table = Variable(tc.zeros(x_maxL, x_maxL, B, self.output_dim), requires_grad=True)

        #print self.l_table
        #print len(self.l_table)
        #print self.l_table[0][2]

        #xs_h_r = tc.FloatTensor(xs_h.data.tolist()[::-1])
        #if wargs.gpu_id and not xs_h_r.is_cuda: xs_h_r = xs_h_r.cuda()
        #xs_h_r = Variable(xs_h_r, requires_grad=True)

        #self.l_table = Variable(tc.zeros(x_maxL, x_maxL+1, B, self.output_dim), requires_grad=True)
        #self.r_table = Variable(tc.zeros(x_maxL, x_maxL+1, B, self.output_dim), requires_grad=True)
        if wargs.gpu_id and not l_table.is_cuda: l_table = l_table.cuda()
        #if wargs.gpu_id and not r_table.is_cuda: r_table = r_table.cuda()
        #if wargs.gpu_id and not self.r_table.is_cuda: self.r_table = self.r_table.cuda()

        for wid in range(0, x_maxL):

            # set elements in diagonal
            #print len(self.l_table[l - 1])
            #self.l_table[l - 1][l] = self.l_f1(xs_h[l - 1])
            # xs_h: (x_maxL, B, input_dim)
            l_table[wid][wid].data.copy_(self.l_f1(xs_h[wid]).data)
            #r_table[wid][wid].data.copy_(self.l_f1(xs_h_r[wid]).data)
            #self.r_table[l - 1][l] = self.r_f1(xs_h_r[l - 1])
            #self.r_table[l - 1][l].data.copy_(self.r_f1(xs_h_r[l - 1]).data)
            if wid < 1: continue
            # wid-1 -> 0
            for j in range(wid - 1, -1, -1):

                l_left_down, r_left_down = [], []
                # from j+1 to l-1
                for k in range(j, wid):
                    #print self.l_table[j][k].size(), self.l_table[k][l].size()
                    #l_rule_comb = tc.stack([self.l_table[j][k], self.l_table[k][l]], dim=0)
                    #r_rule_comb = tc.stack([self.r_table[j][k], self.r_table[k][l]], dim=0)
                    l_rule_comb = tc.cat([l_table[j][k], l_table[k+1][wid]], dim=-1)
                    #r_rule_comb = tc.cat([r_table[j][k], r_table[k+1][wid]], dim=-1)
                    #print l_rule_comb.size()
                    #l_rule_comb = l_rule_comb.unsqueeze(1).permute(2, 1, 0 ,3)
                    #print l_rule_comb.size()
                    #print r_rule_comb.size()
                    #r_rule_comb = r_rule_comb.unsqueeze(1).permute(2, 1, 0 ,3)
                    #print r_rule_comb.size()
                    #print self.l_conv(l_rule_comb).size()
                    l_rule_comb = l_rule_comb[:, None, :]
                    #r_rule_comb = r_rule_comb[:, None, :]
                    l_left_down.append(self.l_conv(l_rule_comb).squeeze(-1).squeeze(-1))
                    #r_left_down.append(self.r_conv(r_rule_comb).squeeze(-1).squeeze(-1))

                #print tc.stack(l_left_down, dim=0).size()
                #print tc.stack(r_left_down, dim=0).size()
                # max pooling for rule comb candidates
                l_node = tc.stack(l_left_down, dim=0).max(0)[0]   # (B, filter_feats_size)
                #r_node = tc.stack(r_left_down, dim=0).max(0)[0]   # (B, filter_feats_size)
                l_table[j][wid].data.copy_(self.l_f2(l_node).data)
                #r_table[j][wid].data.copy_(self.l_f2(r_node).data)
                #self.r_table[j][l] = self.r_f2(r_node)
                del l_left_down[:], l_node, l_rule_comb
                #del l_left_down[:], l_node, l_rule_comb, r_left_down[:], r_node, r_rule_comb
                #self.l_table[j][l].data.copy_(self.l_f2(l_node).data)
                #self.r_table[j][l].data.copy_(self.r_f2(r_node).data)

        #outputs = []
        #self.r_table = [a[::-1] for a in self.r_table]
        #r_table.data.copy_(tc.Tensor(r_table.data.permute(1, 0, 2, 3).tolist()[::-1]).permute(1, 0, 2, 3))
        #r_table = Variable(tc.Tensor(r_table.data.permute(1, 0, 2, 3).tolist()[::-1]).permute(1, 0, 2, 3)
        #                  , requires_grad=True).cuda()
        #print self.l_table
        #self.l_table = tc.stack([tc.stack(a) for a in self.l_table])
        #self.r_table = tc.stack([tc.stack(a) for a in self.r_table])
        #print self.l_table.size()
        #print self.r_table.size()
        #self.l_table = self.l_table.sum(1)
        #self.r_table = self.r_table.sum(1)
        #for wid in range(0, x_maxL):

            # from l-2 to 0
            #for j in range(wid - 1, -1, -1):
                #print tc.stack([self.l_table[x][l] for x in range(0, l)]).size()
                #for x in range(0, l): print self.l_table[x][l].size()
                #l_col_sum = tc.stack([self.l_table[x][l] for x in range(0, l)]).sum(0)
                #for x in range(0, l): print self.r_table[x][l]
                #r_col_sum = tc.stack([self.r_table[x][l-1] for x in range(0, x_maxL - l + 1)]).sum(0)
                #print l_col_sum.size(), r_col_sum.size()
                #outputs.append(tc.stack([l_col_sum, r_col_sum]).mean(0))
                #if self.l_table[j][l] is not None: outputs.append(self.l_table[j][l])

        #print tc.stack(outputs, 0).size()
        #print xs_mask[:, :, None].size()
        #outputs = tc.stack(outputs, 0) * xs_mask[:, :, None] if xs_mask is not None else tc.stack(outputs, 0)
        # (x_maxL, x_maxL, B, output_dim)
        cykmask = self.cyk_mask(x_maxL, xs_mask)
        if wargs.gpu_id and not cykmask.is_cuda: cykmask = cykmask.cuda()

        #return l_table, cykmask
        #print type(l_table), type(r_table), l_table.is_cuda, r_table.is_cuda
        #print l_table.size(), r_table.size()
        return l_table, cykmask
        #return tc.cat((l_table, r_table), dim=-1), cykmask
        #return (self.l_table + self.r_table) * xs_mask[:, :, None]

class NMT(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size):

        super(NMT, self).__init__()

        self.src_lookup_table = nn.Embedding(src_vocab_size, wargs.src_wemb_size, padding_idx=PAD)

        self.encoder = Encoder(wargs.src_wemb_size, wargs.enc_hid_size, with_ln=wargs.laynorm)
        self.cyknet = Cyknet(wargs.enc_hid_size, wargs.enc_hid_size)
        #self.cyknet = Cyknet(wargs.src_wemb_size, wargs.enc_hid_size)
        self.s_init = nn.Linear(wargs.enc_hid_size, wargs.dec_hid_size)
        self.s_init_cyk = nn.Linear(wargs.enc_hid_size, wargs.dec_hid_size)
        self.tanh = nn.Tanh()
        self.ha = nn.Linear(wargs.enc_hid_size, wargs.align_size)
        self.ha_cyk = nn.Linear(wargs.enc_hid_size, wargs.align_size)
        self.decoder = Decoder(trg_vocab_size, with_ln=wargs.laynorm)

    def init_state(self, xs_h, xs_mask=None, cyk_table=None, cykmask=None):

        assert xs_h.dim() == 3  # slen, batch_size, enc_size
        if xs_mask is not None:
            xs_h = (xs_h * xs_mask[:, :, None]).sum(0) / xs_mask.sum(0)[:, None]
        else:
            xs_h = xs_h.mean(0)

        assert cyk_table.dim() == 4   # L, L, B, enc_size
        if cykmask is not None and cykmask is not None:
            #print cyk_table.size()
            #print cykmask[:, :, :, None].size()
            #print (cyk_table * cykmask[:, :, :, None]).sum(0).sum(0).size()
            cyk_table = (cyk_table * cykmask[:, :, :, None]).sum(0).sum(0) / cykmask.sum(0).sum(0)[:, None]
        else:
            cyk_table = cyk_table.mean(0).mean(0)

        return self.tanh(self.s_init(xs_h) + self.s_init_cyk(cyk_table))

    def init_cyk_state(self, cyk_table, cykmask=None):

        assert cyk_table.dim() == 4   # L, L, B, enc_size
        if cykmask is not None:
            #print cyk_table.size()
            #print cykmask[:, :, :, None].size()
            #print (cyk_table * cykmask[:, :, :, None]).sum(0).sum(0).size()
            cyk_table = (cyk_table * cykmask[:, :, :, None]).sum(0).sum(0) / cykmask.sum(0).sum(0)[:, None]
        else:
            cyk_table = cyk_table.mean(0).mean(0)

        return self.tanh(self.s_init(cyk_table))

    def init(self, xs, xs_mask=None, test=True):

        if test:  # for decoding
            if wargs.gpu_id and not xs.is_cuda: xs = xs.cuda()
            xs = Variable(xs, requires_grad=False, volatile=True)

        xs = xs if xs.dim() == 3 else self.src_lookup_table(xs)
        #print '-------------------------------'
        #print xs.size()
        xs = self.encoder(xs, xs_mask)
        #print xs.size()
        uh = self.ha(xs)
        #s0 = self.init_state(xs, xs_mask)
        xs_cyk, cykmask = self.cyknet(xs, xs_mask)
        #print xs.size(), cykmask.size()
        #s0 = self.init_cyk_state(xs, cykmask)
        s0 = self.init_state(xs, xs_mask, xs_cyk, cykmask)
        uh_cyk = self.ha_cyk(xs_cyk)
        return s0, xs, uh, xs_cyk, uh_cyk, cykmask

    def forward(self, srcs, trgs, srcs_m, trgs_m):
        # (max_slen_batch, batch_size, enc_hid_size)
        s0, srcs, uh, xs_cyk, uh_cyk, cykmask = self.init(srcs, srcs_m, False)

        return self.decoder(s0, srcs, trgs, uh, xs_cyk, uh_cyk, srcs_m, trgs_m, cykmask)


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

        self.forw_gru = GRU(input_size, output_size, with_ln=with_ln, prefix=f('Forw'))
        self.back_gru = GRU(output_size, output_size, with_ln=with_ln, prefix=f('Back'))

    def forward(self, xs, xs_mask=None, h0=None):

        max_L, b_size = xs.size(0), xs.size(1)
        assert xs.dim() == 3

        right = []
        h = h0 if h0 else Variable(tc.zeros(b_size, self.output_size), requires_grad=False)
        if wargs.gpu_id: h = h.cuda()
        for k in range(max_L):
            # (batch_size, src_wemb_size)
            h = self.forw_gru(xs[k], xs_mask[k] if xs_mask is not None else None, h)
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

    def forward0(self, s_tm1, xs_h, uh, xs_mask=None):

        d1, d2, d3 = uh.size()
        # (b, dec_hid_size) -> (b, aln) -> (1, b, aln) -> (slen, b, aln) -> (slen, b)
        e_ij = self.a1(
            self.tanh(self.sa(s_tm1)[None, :, :] + uh)).squeeze(2).exp()
        if xs_mask is not None: e_ij = e_ij * xs_mask

        # probability in each column: (slen, b)
        e_ij = e_ij / e_ij.sum(0)[None, :]

        # weighted sum of the h_j: (b, enc_hid_size)
        attend = (e_ij[:, :, None] * xs_h).sum(0)

        return e_ij, attend

    # attention on 2-d dimension
    def forward(self, s_tm1, xs_h, uh, cykmask=None):

        _, L, B, A = uh.size()

        #p.view(-1)[aaa.view(-1).nonzero().view(-1)]
        # (b, dec_hid_size) -> (b, aln) -> (1, 1, b, aln) -> (L, L, b, aln) -> (L, L, b)
        e_ij = self.a1(
            self.tanh(self.sa(s_tm1)[None, None, :, :] + uh)).squeeze(3).exp()
        if cykmask is not None: e_ij = e_ij * cykmask

        # probability in each column: (L, L, b)
        e_ij = e_ij / e_ij.sum(0).sum(0)[None, None, :]

        # weighted sum of the h_j: (b, enc_hid_size)
        attend = (e_ij[:, :, :, None] * xs_h).sum(0).sum(0)

        return e_ij, attend

class Decoder(nn.Module):

    def __init__(self, trg_vocab_size, with_ln=False, max_out=True):

        super(Decoder, self).__init__()

        self.max_out = max_out
        self.attention = Attention(wargs.dec_hid_size, wargs.align_size)
        self.trg_lookup_table = nn.Embedding(trg_vocab_size, wargs.trg_wemb_size, padding_idx=PAD)
        self.tanh = nn.Tanh()
        self.gru1 = GRU(wargs.trg_wemb_size, wargs.dec_hid_size, with_ln=with_ln)
        self.gru2 = GRU(wargs.enc_hid_size, wargs.dec_hid_size, with_ln=with_ln)

        self.att_seq = nn.Linear(wargs.enc_hid_size, wargs.enc_hid_size)
        self.att_cyk = nn.Linear(wargs.enc_hid_size, wargs.enc_hid_size)

        out_size = 2 * wargs.out_size if max_out else wargs.out_size
        self.ls = nn.Linear(wargs.dec_hid_size, out_size)
        self.ly = nn.Linear(wargs.trg_wemb_size, out_size)
        self.lc = nn.Linear(wargs.enc_hid_size, out_size)

    def step(self, s_tm1, xs_h, uh, y_tm1, xs_cyk, uh_cyk, xs_mask=None, y_mask=None, cykmask=None):

        if not isinstance(y_tm1, tc.autograd.variable.Variable):
            if isinstance(y_tm1, int): y_tm1 = tc.Tensor([y_tm1]).long()
            elif isinstance(y_tm1, list): y_tm1 = tc.Tensor(y_tm1).long()
            if wargs.gpu_id: y_tm1 = y_tm1.cuda()
            y_tm1 = Variable(y_tm1, requires_grad=False, volatile=True)
            y_tm1 = self.trg_lookup_table(y_tm1)

        s_above = self.gru1(y_tm1, y_mask, s_tm1)
        # (slen, batch_size) (batch_size, enc_hid_size)
        alpha_ij_seq, attend_seq = self.attention.forward0(s_above, xs_h, uh, xs_mask)
        alpha_ij_cyk, attend_cyk = self.attention(s_above, xs_cyk, uh_cyk, cykmask)
        attend = self.att_seq(attend_seq) + self.att_cyk(attend_cyk)
        s_t = self.gru2(attend, y_mask, s_above)

        return attend, s_t, y_tm1, alpha_ij_seq

    def forward(self, s_tm1, xs_h, ys, uh, xs_cyk, uh_cyk, xs_mask=None, ys_mask=None, cykmask=None):

        tlen_batch_s, tlen_batch_c = [], []
        y_Lm1, b_size = ys.size(0), ys.size(1)
        # (max_tlen_batch - 1, batch_size, trg_wemb_size)
        ys_e = ys if ys.dim() == 3 else self.trg_lookup_table(ys)
        for k in range(y_Lm1):
            attend, s_tm1, _, _ = self.step(s_tm1, xs_h, uh, ys_e[k], xs_cyk, uh_cyk,
                                            xs_mask if xs_mask is not None else None,
                                            ys_mask[k] if ys_mask is not None else None, cykmask)
            tlen_batch_c.append(attend)
            tlen_batch_s.append(s_tm1)

        s = tc.stack(tlen_batch_s, dim=0)
        c = tc.stack(tlen_batch_c, dim=0)
        del tlen_batch_s, tlen_batch_c

        logit = self.step_out(s, ys_e, c)
        if ys_mask is not None: logit = logit * ys_mask[:, :, None]  # !!!!
        del s, c

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

