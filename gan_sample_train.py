import wargs
import const
import torch as tc
import math
from translate import Translator
from utils import *
from torch.autograd import Variable
from train import memory_efficient
from optimizer import Optim


class Trainer:

    # src: (max_slen_batch, batch_size, emb)
    # gold: (max_tlen_batch, batch_size, emb)
    def __init__(self, nmtModel, sv, tv, optim, trg_dict_size, n_critic=1):

        self.nmtModel = nmtModel
        self.sv = sv
        self.tv = tv
        self.optim = optim
        self.trg_dict_size = trg_dict_size

        self.n_critic = n_critic

        self.translator = Translator(nmtModel, sv, tv, beam=10)

        self.optim_G = Optim(
            'adadelta', 1.0, wargs.max_grad_norm,
            learning_rate_decay=wargs.learning_rate_decay,
            start_decay_from=wargs.start_decay_from,
            last_valid_bleu=wargs.last_valid_bleu
        )



    # p1: (max_tlen_batch, batch_size, vocab_size)
    def distance(self, p1, p2, y_masks, type='JS', y_gold=None):

        if p2.size(0) > p1.size(0):

            p2 = p2[:(p1.size(0) + 1)]

        if type == 'JS':

            #D_kl = tc.mean(tc.sum((tc.log(p1) - tc.log(p2)) * p1, dim=-1).squeeze(), dim=0)
            M = (p1 + p2) / 2.
            D_kl1 = tc.sum((tc.log(p1) - tc.log(M)) * p1, dim=-1).squeeze()
            D_kl2 = tc.sum((tc.log(p2) - tc.log(M)) * p2, dim=-1).squeeze()
            JS = 0.5 * D_kl1 + 0.5 * D_kl2

            dist = tc.sum(JS * y_masks)
            del JS

        elif type == 'KL':

            KL = tc.sum((tc.log(p1) - tc.log(p2)) * p1, dim=-1).squeeze()
            dist = tc.sum(KL * y_masks)

        elif type == 'KL-sent':

            p1 = tc.gather(p1, 2, y_gold.unsqueeze(2).expand_as(p1))[:, :, 0]
            p2 = tc.gather(p2, 2, y_gold.unsqueeze(2).expand_as(p2))[:, :, 0]
            # p1 (max_tlen_batch, batch_size)

            dist = tc.sum((y_masks * tc.log(p1) - y_masks * tc.log(p2)) * p1).squeeze()
            # KL: (1, batch_size)

        return dist / y_masks.size(1)

    def save_model(self, eid, bid):

        model_state_dict = self.nmtModel.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'classifier' not in k}
        class_state_dict = self.nmtModel.classifier.state_dict()
        model_dict = {
            'model': model_state_dict,
            'class': class_state_dict,
            'epoch': eid,
            'batch': bid,
            'optim': self.optim
        }

        if wargs.save_one_model:
            model_file = '{}.pt'.format(wargs.model_prefix)
        else:
            model_file = '{}_e{}_upd{}.pt'.format(wargs.model_prefix, eid, bid)
        tc.save(model_dict, model_file)

    def try_trans(self, src, ref):

        # (len, 1)
        #src = sent_filter(list(src[:, bid].data))
        x_filter = sent_filter(list(src))
        y_filter = sent_filter(list(ref))
        #wlog('\n[{:3}] {}'.format('Src', idx2sent(x_filter, self.sv)))
        #wlog('[{:3}] {}'.format('Ref', idx2sent(y_filter, self.tv)))

        onebest, onebest_ids = self.translator.trans_onesent(x_filter)

        #wlog('[{:3}] {}'.format('Out', onebest))

        return onebest_ids


    def batch_hyp(self, src, x_masks, ref, y_maxL):

        # y_masks: (trg_max_len, batch_size)
        batch_size = src.size(1)
        maxL = y_maxL
        hyps, hyps_L, hyps_ = [], [], []

        for bid in range(batch_size):

            onebest_ids = self.try_trans(src[:, bid].data, ref[:, bid].data)

            if len(onebest_ids) == 0 or onebest_ids[0] != const.BOS:
                onebest_ids = [const.BOS] + onebest_ids
            if onebest_ids[-1] == const.EOS: onebest_ids = onebest_ids[:-1]

            onebest_ids = tc.Tensor(onebest_ids).long()
            hyps.append(onebest_ids)
            hyp_L = onebest_ids.size(0)
            if hyp_L > maxL: maxL = hyp_L

            hyps_L.append(hyp_L)
            hyps_.append(None)

        for bid in range(batch_size):

            hyps[bid] = tc.cat(tuple([hyps[bid],
                                      const.PAD * tc.ones(maxL - hyps_L[bid]).long()]), 0)

        hyps = tc.stack(tuple(hyps), dim=1)
        if wargs.gpu_id and not hyps.is_cuda: hyps = hyps.cuda()
        hyps = Variable(hyps, requires_grad=False)

        hyps_mask = hyps.ne(const.PAD).float()
        # (len, batch_size)
        o2 = self.nmtModel(src, hyps, x_masks, hyps_mask)
        p_y_hpy = self.nmtModel.classifier.logit_to_prob(o2)

        #print maxL, y_maxL
        maxL = maxL if maxL > y_maxL else y_maxL
        #print maxL, y_maxL
        for bid in range(batch_size):

            hyp_L = hyps_L[bid]
            #print maxL, hyps_L[bid]
            #print '------------'
            one_p_y_hpy = p_y_hpy[:, bid, :]

            if hyp_L < maxL:
                pad = tc.ones(maxL - hyp_L) / self.trg_dict_size
                pad = pad.unsqueeze(-1).expand((pad.size(0), one_p_y_hpy.size(-1)))
                if wargs.gpu_id and not pad.is_cuda: pad = pad.cuda()
                #print one_p_y_hpy.size(0), pad.size(0)
                one_p_y_hpy.data[hyp_L:] = pad

            hyps_[bid] = one_p_y_hpy

        #for h in hyps_:
        #    print h.size()
        hyps_ = tc.stack(tuple(hyps_), dim=1)
        if hyps_.size(0) > y_maxL:
            hyps_ = hyps_[:y_maxL]
            hyps_mask = hyps_mask[:y_maxL]
        return hyps_, hyps_mask

    def train(self, train_data):

        loss_val = 0.
        batch_count = len(train_data)

        for eid in range(wargs.start_epoch, wargs.max_epochs):

            for bid in range(batch_count):

                _, srcs, trgs, slens, srcs_m, trgs_m = train_data[bid]
                gold_feed, gold_feed_mask = trgs[:-1], trgs_m[:-1]
                B = srcs.size(1)
                N = trgs[1:].data.ne(const.PAD).sum()

                wlog('Train Discrimitor .......... ')
                for j in range(self.n_critic):

                    self.optim.init_optimizer(self.nmtModel.parameters())
                    self.nmtModel.zero_grad()
                    #self.optim.zero_grad()

                    o1 = self.nmtModel(srcs, gold_feed, srcs_m, gold_feed_mask)
                    p_y_gold = self.nmtModel.classifier.logit_to_prob(o1)
                    # p_y_gold: (gold_max_len - 1, batch_size, trg_dict_size)

                    p_y_hpy1, hyps_mask = self.batch_hyp(srcs, srcs_m,
                                                         gold_feed, gold_feed_mask.size(0))
                    # p_y_hpy1: ()

                    #print 'aaaaaaaaaaaaaaaaaaaaaaaaa'
                    #print p_y_gold.size()
                    #print p_y_hpy1.size()
                    #print hyps_mask.size()
                    #loss_D = -self.distance(p_y_gold, p_y_hpy1, hyps_mask, type='KL')
                    loss_D = self.distance(p_y_gold, p_y_hpy1, hyps_mask,
                                           type='KL-sent', y_gold=trgs[1:])
                    wlog('Discrimitor KL distance {}'.format(loss_D.data[0]))

                    #loss_D.div(batch_size).backward(retain_variables=True)
                    (1 * loss_D).div(B).backward()
                    self.optim.step()
                    del o1, p_y_gold, p_y_hpy1, hyps_mask

                wlog('Train generator .......... ')
                #o1 = self.nmtModel(srcs, gold_feed, srcs_m, gold_feed_mask)
                #p_y_gold = self.nmtModel.classifier.prob(o1)

                self.optim_G.init_optimizer(self.nmtModel.parameters())
                self.nmtModel.zero_grad()
                #self.optim.zero_grad()
                #p_y_hpy2, hyps_mask = self.batch_hyp(srcs, srcs_m,
                #                                     gold_feed, gold_feed_mask.size(0))
                #print 'aaaaaaaaaaaaaaaaaaaaaaaaa'
                #print p_y_gold.size()
                #print p_y_hpy2.size()
                #print hyps_mask.size()
                # padding hyps
                #loss_G = self.distance(p_y_gold, p_y_hpy2, hyps_mask, type='KL')
                #loss_G.div(batch_size).backward(retain_variables=True)
                #loss_G.div(batch_size).backward()
                for i in range(5):
                    outputs = self.nmtModel(srcs, gold_feed, srcs_m, gold_feed_mask)
                    batch_loss, grad_output, batch_correct_num = memory_efficient(
                        outputs, trgs[1:], trgs_m[1:], self.nmtModel.classifier)
                    outputs.backward(grad_output)

                    #loss, correct_num = self.nmtModel.classifier(o1, trgs[1:], trgs_m[1:])
                    wlog('W-MLE:{:4.2f}, W-ppl:{:4.2f}, S-MLE:{:4.2f}'.format(
                        batch_loss/N, math.exp(batch_loss/N), batch_loss/B))
                    #loss.div(batch_size).backward()
                    #(loss_G + loss).div(batch_size).backward()

                    self.optim_G.step()
                    #del loss, correct_num
                    #del o1, p_y_gold, p_y_hpy2, hyps_mask
                    del outputs, batch_loss, batch_correct_num

                #for k in range(batch_size):
                #    onebest_ids = self.try_trans(srcs[:, k].data, gold_feed[:, k].data)
                
