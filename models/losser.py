import pdb
import torch as tc
import torch.nn as nn

import wargs
from tools.utils import *
from collections import Counter
import time

class MyLogSoftmax(nn.Module):

    def __init__(self, self_norm_alpha=None):

        super(MyLogSoftmax, self).__init__()
        self.sna = self_norm_alpha

    def forward(self, x):

        # input torch tensor or variable
        x_max = tc.max(x, dim=-1, keepdim=True)[0]  # take max for numerical stability
        log_norm = tc.log( tc.sum( tc.exp( x - x_max ), dim=-1, keepdim=True ) + epsilon ) + x_max
        # get log softmax
        x = x - log_norm

        # Sum_( log(P(xi)) - alpha * square( log(Z(xi)) ) )
        if self.sna is not None: x = x - self.sna * tc.pow(log_norm, 2)

        return log_norm, x

class Classifier(nn.Module):

    def __init__(self, input_size, output_size, trg_lookup_table=None):

        super(Classifier, self).__init__()

        self.dropout = nn.Dropout(wargs.drop_rate)
        self.map_vocab = nn.Linear(input_size, output_size)

        if trg_lookup_table is not None:
            assert input_size == wargs.trg_wemb_size
            wlog('Copying weight of trg_lookup_table into classifier')
            self.map_vocab.weight = trg_lookup_table.weight
        #self.log_prob = nn.LogSoftmax()
        self.log_prob = MyLogSoftmax(wargs.self_norm_alpha)

        weight = tc.ones(output_size)
        weight[PAD] = 0   # do not predict padding, same with ingore_index
        self.criterion = nn.NLLLoss(weight, size_average=False, ignore_index=PAD)

        self.output_size = output_size
        self.softmax = nn.Softmax()

        self.losses = {"nll_loss": self.nll_loss,  "mixed_loss": self.mixed_loss
        , "sen_p2_loss" : self.sen_p2_loss, "sen_gleu_loss" : self.sen_gleu_loss, 
        "sen_bleu_loss" : self.sen_bleu_loss}
        self.loss = self.losses[wargs.loss]
        self.t1 = 0.0
        self.t2 = 0.0
    def get_a(self, logit, noise=False):

        if not logit.dim() == 2: logit = logit.contiguous().view(-1, logit.size(-1))
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
        log_norm, pred = self.log_prob(pred)
        pred = pred * gold_mask[:, None]

        return self.criterion(pred, gold), log_norm * gold_mask[:, None]


    def sen_gleu_loss(self, batch_count, pred_mask, pred, gold, gold_mask):

        pred = self.softmax(pred)
        pred_mask = tc.transpose(pred_mask,0,1)
        pred, gold, gold_mask = self.unsqueeze_transpose(batch_count, pred, gold, gold_mask)
        pred = pred * pred_mask[:,:, None]
        gold, gold_mask = gold.data.cpu().numpy().tolist(), gold_mask.data.cpu().numpy().tolist()
        pred_mask = pred_mask.data.cpu().numpy().tolist()

        losses = []
        for i in range(pred.size(0)):
            match_sum = []
            gold_sum = []
            gold_ngram, match_ngram = self.first_ngram(pred_mask[i], pred[i], gold[i], gold_mask[i])
            for j in range(4):
                match_sum.append(match_ngram[j])
                gold_sum.append(gold_ngram[j])
            gleu_1 = sum(match_ngram) / sum(gold_ngram)
            losses.append(-1 * gleu_1)


        loss = sum(losses)/batch_count
        return loss



    def sen_bleu_loss(self, batch_count, pred_mask, pred, gold, gold_mask):

        pred = self.softmax(pred)
        pred_mask = tc.transpose(pred_mask,0,1)
        pred, gold, gold_mask = self.unsqueeze_transpose(batch_count, pred, gold, gold_mask)
        pred = pred * pred_mask[:,:, None]
        gold, gold_mask = gold.data.cpu().numpy().tolist(), gold_mask.data.cpu().numpy().tolist()
        pred_mask = pred_mask.data.cpu().numpy().tolist()

        losses = []
        for i in range(pred.size(0)):
            if(len(gold[i]) < 5):
                continue
            match_4 = []
            gold_ngram, match_ngram = self.first_ngram(pred_mask[i], pred[i], gold[i], gold_mask[i])
            for j in range(4):
                match_4.append(match_ngram[j]/gold_ngram[j])
            bleu = tc.rsqrt(tc.rsqrt(match_4[0] * match_4[1] * match_4[2] * match_4[3]))

            losses.append(-1 * bleu)
            loss = sum(losses)/batch_count
            return loss



    def sen_p2_loss(self, batch_count, pred_mask, pred, gold, gold_mask):

        pred = self.softmax(pred)
        pred_mask = tc.transpose(pred_mask,0,1)
        pred, gold, gold_mask = self.unsqueeze_transpose(batch_count, pred, gold, gold_mask)
        pred = pred * pred_mask[:,:, None]
        gold, gold_mask = gold.data.cpu().numpy().tolist(), gold_mask.data.cpu().numpy().tolist()
        pred_mask = pred_mask.data.cpu().numpy().tolist()

        losses = []
        for i in range(pred.size(0)):
            gold_ngram, match_ngram = self.first_ngram(pred_mask[i], pred[i], gold[i], gold_mask[i])
            p2 = match_ngram[1] / gold_ngram[1]
            losses.append(-1 * p2)
        loss = sum(losses)/batch_count
        return loss

    


    def mixed_loss(self, sen_loss, alpha, batch_count, pred, gold, gold_mask):
        nll_loss, _ = self.nll_loss(pred, gold, gold_mask)
        senlevel_loss = sen_loss(batch_count, pred, gold, gold_mask)
        return alpha * senlevel_loss + (1-alpha) * nll_loss

    def unsqueeze_transpose(self, batch_count, pred, gold, gold_mask):

        pred = pred.view(-1, batch_count, pred.size(-1))
        gold = gold.view(-1, batch_count)
        gold_mask = gold_mask.view(-1, batch_count)
        pred, gold, gold_mask = tc.transpose(pred,0,1), tc.transpose(gold,0,1), tc.transpose(gold_mask,0,1)
        return pred, gold, gold_mask

    def first_ngram(self, pred_mask, pred, gold, gold_mask):

        pred_mask = [int(mask) for mask in pred_mask]
        gold_mask = [int(mask) for mask in gold_mask]
        (prob, index) = tc.max(pred, dim = 1)
        index = index.data.tolist()
        one_gram = Counter()
        for j in range(len(gold_mask)):
            if gold_mask[j] == 1:
                one_gram[(gold[j])] += 1

        two_gram = Counter()
        for j in range(len(gold_mask) - 1):
            if gold_mask[j+1] == 1:
                two_gram[(gold[j],gold[j+1])] += 1

        three_gram = Counter()
        for j in range(len(gold_mask) - 2):
            if gold_mask[j+2] == 1:
                three_gram[(gold[j],gold[j+1],gold[j+2])] += 1

        four_gram = Counter()
        for j in range(len(gold_mask) - 3):
            if gold_mask[j+3] == 1:
                four_gram[(gold[j],gold[j+1],gold[j+2],gold[j+3])] += 1

        sum_one_gram = Variable(tc.zeros(1).cuda())
        sum_two_gram = Variable(tc.zeros(1).cuda())
        sum_three_gram = Variable(tc.zeros(1).cuda())
        sum_four_gram = Variable(tc.zeros(1).cuda())
        pred_one_gram = Counter()
        for j in range(len(pred_mask)):
            if pred_mask[j] == 1:
                pred_one_gram[(index[j])] += prob[j]
                sum_one_gram += prob[j]
        pred_two_gram = Counter()
        for j in range(len(pred_mask) - 1):
            if pred_mask[j+1] == 1:
                pred_two_gram[(index[j],index[j+1])] += prob[j] * prob[j+1]
                sum_two_gram += prob[j] * prob[j+1]
        pred_three_gram = Counter()
        for j in range(len(pred_mask) - 2):
            if pred_mask[j+2] == 1:
                pred_three_gram[(index[j],index[j+1],index[j+2])] += prob[j] * prob[j+1] * prob[j+2]
                sum_three_gram += prob[j] * prob[j+1] * prob[j+2]

        pred_four_gram = Counter()
        for j in range(len(pred_mask) - 3):
            if pred_mask[j+3] == 1:
                pred_four_gram[(index[j],index[j+1],index[j+2],index[j+3])] += prob[j] * prob[j+1] * prob[j+2] * prob[j+3]
                sum_four_gram += prob[j] * prob[j+1] * prob[j+2] * prob[j+3]

        match_one_gram = Variable(tc.zeros(1).cuda())
        match_two_gram = Variable(tc.zeros(1).cuda())
        match_three_gram = Variable(tc.zeros(1).cuda())
        match_four_gram = Variable(tc.zeros(1).cuda())
        
        for gram in one_gram:
            if gram in pred_one_gram:
                if one_gram[gram] > pred_one_gram[gram].data[0]:
                    match_one_gram += pred_one_gram[gram]
                else:
                    match_one_gram += one_gram[gram]
        for gram in two_gram:
            if gram in pred_two_gram:
                if two_gram[gram] > pred_two_gram[gram].data[0]:
                    match_two_gram += pred_two_gram[gram]
                else:
                    match_two_gram += two_gram[gram]
        for gram in three_gram:
            if gram in pred_three_gram:
                if three_gram[gram] > pred_three_gram[gram].data[0]:
                    match_three_gram += pred_three_gram[gram]
                else:
                    match_three_gram += three_gram[gram]
        for gram in four_gram:
            if gram in pred_four_gram:
                if four_gram[gram] > pred_four_gram[gram].data[0]:
                    match_four_gram += pred_four_gram[gram]
                else:
                    match_four_gram += four_gram[gram]

        return [sum_one_gram, sum_two_gram, sum_three_gram, sum_four_gram], [match_one_gram, 
            match_two_gram,match_three_gram,match_four_gram]





    def forward(self, feed, pred_mask=None, gold=None, gold_mask=None, noise=False):

        # no dropout in decoding
        feed = self.dropout(feed) if gold is not None else feed
        # (max_tlen_batch - 1, batch_size, out_size)
        pred = self.get_a(feed, noise)

        # decoding, if gold is None and gold_mask is None:
        if gold is None: return -self.log_prob(pred)[-1] if wargs.self_norm_alpha is None else -pred


        assert(gold.dim() == 2)
        batch_count = gold.size(-1)
        if gold.dim() == 2: gold, gold_mask = gold.view(-1), gold_mask.view(-1)

        # negative likelihood log
        if wargs.loss == "nll_loss":
            loss, log_norm = self.nll_loss(pred, gold, gold_mask)
        elif wargs.loss == "mixed_loss":
            loss = self.loss(self.losses[wargs.mix_loss], wargs.alpha, batch_count, pred, gold, gold_mask)
        else:
            loss = wargs.loss_learning_rate * self.loss(batch_count, pred_mask, pred, gold, gold_mask)

        # (max_tlen_batch - 1, batch_size, trg_vocab_size)

        # total loss,  correct count in one batch
        return loss

    #   outputs: the predict outputs from the model.
    #   gold: correct target sentences in current batch 
    def snip_back_prop(self, outputs, pred_mask, gold, gold_mask, shard_size=100):

        """
        Compute the loss in shards for efficiency.
        """
        cur_batch_count = outputs.size(1)
        loss = self(outputs, pred_mask, gold, gold_mask)
        batch_loss = loss.data.clone()[0]
        loss.div(cur_batch_count).backward()

        return batch_loss

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


