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

    #   outputs: the predict outputs from the model.
    #   gold: correct target sentences in current batch 
    def snip_back_prop(self, outputs, gold, gold_mask, shard_size=100):

        """
        Compute the loss in shards for efficiency.
        """
        batch_loss, batch_correct_num = 0, 0
        cur_batch_count = outputs.size(1)
        shard_state = { "feed": outputs, "gold": gold, 'gold_mask': gold_mask }

        for shard in shards(shard_state, shard_size):
            loss, pred_correct = self(**shard)
            batch_loss += loss.data.clone()[0]
            batch_correct_num += pred_correct.data.clone()[0]
            loss.div(cur_batch_count).backward()

        return batch_loss, batch_correct_num

def filter_shard_state(state):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=True, volatile=False)
            yield k, v

def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute.make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.
    Yields:
        Each yielded shard is a dict.
    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield state
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, tc.split(v, shard_size))
                             for k, v in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            # each slice: return (('feed', 'gold', ...), (feed0, gold0, ...))
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = ((state[k], v.grad.data) for k, v in non_none.items()
                     if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        tc.autograd.backward(inputs, grads)


