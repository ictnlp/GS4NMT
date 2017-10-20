import torch as tc
import torch.nn as nn

import wargs
from utils import *

class Classifier(nn.Module):

    def __init__(self, input_size, output_size, trg_lookup_table=None):

        super(Classifier, self).__init__()

        self.dropout = nn.Dropout(wargs.drop_rate)
        self.map_vocab = nn.Linear(input_size, output_size)

        if trg_lookup_table is not None:
            assert input_size == wargs.trg_wemb_size
            wlog('Copying weight of trg_lookup_table into classifier')
            self.map_vocab.weight = trg_lookup_table.weight
        self.log_prob = nn.LogSoftmax()

        weight = tc.ones(output_size)
        weight[PAD] = 0   # do not predict padding, same with ingore_index
        self.criterion = nn.NLLLoss(weight, size_average=False, ignore_index=PAD)

        self.output_size = output_size
        self.softmax = nn.Softmax()

    def get_a(self, logit, noise=False):

        logit = self.map_vocab(logit)

        if noise is True:
            g = get_gumbel(logit.size(0), logit.size(1))
            if wargs.gpu_id and not g.is_cuda: g = g.cuda()
            logit = (logit + g * 0.05) / 1.

        return logit

    def logit_to_prob(self, logit, gumbel=None, tao=None):

        # (L, B)
        d1, d2, _ = logit.size()
        logit = self.get_a(logit)
        if gumbel is None:
            p = self.softmax(logit)
        else:
            #print 'logit ..............'
            #print tc.max((logit < 1e+10) == False)
            #print 'gumbel ..............'
            #print tc.max((gumbel < 1e+10) == False)
            #print 'aaa ..............'
            #aaa = (gumbel.add(logit)) / tao
            #print tc.max((aaa < 1e+10) == False)
            p = self.softmax((gumbel.add(logit)) / tao)
        p = p.view(d1, d2, self.output_size)

        return p

    def nll_loss(self, pred, gold, gold_mask):

        if pred.dim() == 3: pred = pred.view(-1, pred.size(-1))
        pred = self.log_prob(pred)
        pred = pred * gold_mask[:, None]

        return self.criterion(pred, gold)

    def forward(self, feed, gold=None, gold_mask=None, noise=False):

        # no dropout in decoding
        feed = self.dropout(feed) if gold is not None else feed
        # (max_tlen_batch - 1, batch_size, out_size)
        pred = self.get_a(feed, noise)

        # decoding, if gold is None and gold_mask is None:
        if gold is None: return -self.log_prob(pred)

        if gold.dim() == 2: gold, gold_mask = gold.view(-1), gold_mask.view(-1)
        # negative likelihood log
        nll = self.nll_loss(pred, gold, gold_mask)

        # (max_tlen_batch - 1, batch_size, trg_vocab_size)
        pred_correct = (pred.max(dim=-1)[1]).eq(gold).masked_select(gold.ne(PAD)).sum()

        # total loss,  correct count in one batch
        return nll, pred_correct


