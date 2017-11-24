import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import wargs
from gru import GRU
from tools.utils import *

class NMT(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size):

        super(NMT, self).__init__()

        self.encoder = Encoder(src_vocab_size, wargs.src_wemb_size, wargs.enc_hid_size)
        self.s_init = nn.Linear(wargs.enc_hid_size, wargs.dec_hid_size)
        self.tanh = nn.Tanh()
        #if wargs.dynamic_cyk_decoding is False: self.ha = nn.Linear(wargs.enc_hid_size, wargs.align_size)
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

        if test:  # for decoding
            if wargs.gpu_id and not xs.is_cuda: xs = xs.cuda()
            xs = Variable(xs, requires_grad=False, volatile=True)

        xs = self.encoder(xs, xs_mask)
        s0 = self.init_state(xs, xs_mask)
        #uh = self.ha(xs) if wargs.dynamic_cyk_decoding is False else None
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
        self.a1 = nn.Linear(self.align_size, 1)

    def forward(self, s_tm1, xs_h, uh, xs_mask=None):

        d1, d2, d3 = uh.size()
        # (b, dec_hid_size) -> (b, aln) -> (1, b, aln) -> (slen, b, aln) -> (slen, b)
        #print self.sa(s_tm1)[None, :, :].size(), uh.size()
        #print 'xs_mask: ', xs_mask
        #print 'uh: ', uh
        #print 's_tm1: ', s_tm1
        #print self.sa(s_tm1)[None, :, :]
        #print 'tanh: ', self.tanh(self.sa(s_tm1)[None, :, :] + uh)
        #print 'no exp: ', self.a1(self.tanh(self.sa(s_tm1)[None, :, :] + uh)).squeeze(2)
        #print self.a1.weight

        #e_ij = self.a1(self.tanh(self.sa(s_tm1)[None, :, :] + uh)).squeeze(2).exp()

        # better softmax version with max for numerical stability
        e_ij = self.a1(self.tanh(self.sa(s_tm1)[None, :, :] + uh)).squeeze(2)
        #print 'e_ij: ', e_ij
        #print 'e_ij - max: ', e_ij - e_ij.max(0)[0]
        e_ij = (e_ij - e_ij.max(0)[0]).exp()
        #print 'exp e_ij - max: ', e_ij

        if xs_mask is not None: e_ij = e_ij * xs_mask
        #print 'mask exp e_ij: ', e_ij

        # probability in each column: (slen, b)
        e_ij = e_ij / e_ij.sum(0)[None, :]

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
        self.gru1 = GRU(wargs.trg_wemb_size, wargs.dec_hid_size)
        self.gru2 = GRU(wargs.enc_hid_size, wargs.dec_hid_size)

        out_size = 2 * wargs.out_size if max_out else wargs.out_size
        self.ls = nn.Linear(wargs.dec_hid_size, out_size)
        self.ly = nn.Linear(wargs.trg_wemb_size, out_size)
        self.lc = nn.Linear(wargs.enc_hid_size, out_size)

        '''
        if wargs.dynamic_cyk_decoding is True:
            self.fwz = wargs.filter_window_size
            self.ffs = wargs.filter_feats_size
            self.ha = nn.Linear(wargs.enc_hid_size, wargs.align_size)
            for i in range(len(self.fwz)):
                self.l_f1 = nn.Linear(wargs.enc_hid_size, wargs.enc_hid_size)
                self.l_conv = nn.Sequential(
                    nn.Conv1d(1, self.ffs[i], kernel_size=wargs.enc_hid_size*self.fwz[i],
                                        stride=wargs.enc_hid_size),
                    nn.ReLU(),
                    #nn.BatchNorm2d(self.ffs[i])
                )
                #self.l_f2 = nn.Linear(self.ffs[i], wargs.enc_hid_size)
                self.l_f2 = nn.Linear(2 * wargs.enc_hid_size, wargs.enc_hid_size)
        '''

    '''
        record the source idx of last translation for each sentence in a batch, if this idx is
        next to last one, we use cnn to combine them and update the xs_mask to do next attention
        add batch_adj_list parameter: [[sent_0], [sent_1], ..., [sent_(batch_size-1)]]
    '''
    def update_src_btg_tree(self, xs_h, xs_mask,
                            batch_adj_list, p_attend_sidx=None, c_attend_sidx=None, prevb_id=None):
        if p_attend_sidx is None or p_attend_sidx[0] is None: return
        else:
            # batch
            for bidx, (p, c) in enumerate(zip(p_attend_sidx, c_attend_sidx)):
                #_adj_list = batch_adj_list[bidx] if prevb_id is None else batch_adj_list[prevb_id[bidx]]
                _adj_list = batch_adj_list[bidx]
                #while c not in _adj_list: c = c + 1    # for some error
                #p = self.p_attend_sidx[bidx]
                debug('Batch id {}: {} {} {}'.format(bidx, p, c, _adj_list))
                assert (p in _adj_list) and (c in _adj_list)
                if abs(_adj_list.index(p) - _adj_list.index(c)) == 1:
                    # change source mask for next attention step
                    print 'merge################', bidx, p, c, xs_mask.size()
                    #if prevb_id is None: xs_mask[p][bidx].data.copy_(tc.zeros(1))
                    #else: xs_mask[p][_idx].data.copy_(tc.zeros(1))
                    #if prevb_id is None: batch_adj_list[bidx].remove(p)  # update the adjacency list
                    #else: batch_adj_list[_idx][bidx].remove(p)  # update the adjacency list
                    #if prevb_id is None: xs_mask[p][bidx].data.copy_(tc.zeros(1))
                    #else: xs_mask[p][prevb_id[bidx]].data.copy_(tc.zeros(1))
                    xs_mask[p][bidx].data.copy_(tc.zeros(1))
                    batch_adj_list[bidx].remove(p)  # update the adjacency list
                    #print '1****batch_adj_list[{}]: '.format(bidx), batch_adj_list[bidx]
                    #print 'remove****batch_adj_list: ', batch_adj_list
                    #adj_reduce = tc.cat([self.l_f1(xs_h[p][bidx]),
                    #                     self.l_f1(xs_h[c][bidx])], dim=-1)[None, None, :]
                    # (1, 1, 2*enc_hid_size) -> (1, self.ffs[i], 1) -> (1, enc_hid_size)
                    # update the expression of reduced neighbours
                    #xs_h[c][bidx].data.copy_(self.tanh(
                    #    self.l_f2(self.l_conv(adj_reduce).squeeze(-1))).squeeze().data)
                    #adj_reduce = tc.cat([self.l_f1(xs_h[p][bidx]),
                    #                     self.l_f1(xs_h[c][bidx])], dim=-1)[None, :]
                    #xs_h[c][bidx].data.copy_(self.tanh(self.l_f2(adj_reduce)).data)
                    #print xs_h[c][bidx].size()
                    #adj_reduce = tc.cat([xs_h[p][bidx][None, :], xs_h[c][bidx][None, :]], dim=0)
                    #xs_h[c][bidx].data.copy_(adj_reduce.mean(0).data)
                    #print adj_reduce.size()

                #if bidx not in delete_idx: self.p_attend_sidx[bidx] == c
                #if prevb_id is None: self.p_attend_sidx[bidx] = c
                #else:
                #    print prevb_id, delete_idx
                #    self.p_attend_sidx[prevb_id[bidx]] = c
                    #assert len(delete_idx) + len(c_attend_sidx) == len(self.p_attend_sidx)
            #self.p_attend_sidx = rm_elems_byid(self.p_attend_sidx, delete_idx)
            #self.p_attend_sidx = c_attend_sidx
            #self.p_attend_sidx = [None] * len(c_attend_sidx)
            #for bidx, c in enumerate(c_attend_sidx):
                # update the previous attent id to parent
                #if prevb_id is None: self.p_attend_sidx[bidx] = c
                #else: self.p_attend_sidx[prevb_id[bidx]] = c
                #self.p_attend_sidx[bidx] = c

            return

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

    def forward(self, s_tm1, xs_h, ys, uh, xs_mask=None, ys_mask=None):

        tlen_batch_s, tlen_batch_c = [], []
        y_Lm1, b_size = ys.size(0), ys.size(1)

        if wargs.dynamic_cyk_decoding is True:
            batch_adj_list = []
            for idx in xs_mask.sum(0): batch_adj_list.append(range(int(idx.data[0])))
            p_attend_sidx = None

        # (max_tlen_batch - 1, batch_size, trg_wemb_size)
        ys_e = ys if ys.dim() == 3 else self.trg_lookup_table(ys)
        for k in range(y_Lm1):

            #if wargs.dynamic_cyk_decoding is True: uh = self.ha(xs_h)

            attend, s_tm1, _, alpha_ij = \
                    self.step(s_tm1, xs_h, uh, ys_e[k],
                              xs_mask if xs_mask is not None else None,
                              ys_mask[k] if ys_mask is not None else None)

            if wargs.dynamic_cyk_decoding is True:
                #print alpha_ij
                # (slen, batch_size)
                #print 'alpha_ij: ', alpha_ij
                c_attend_sidx = alpha_ij.data.max(0)[1].tolist()
                #print 'c_attend_sidx:', c_attend_sidx
                self.update_src_btg_tree(xs_h, xs_mask, batch_adj_list, p_attend_sidx, c_attend_sidx)
                p_attend_sidx = c_attend_sidx

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


