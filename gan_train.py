import wargs
import const
import torch as tc
import math
from cp_sample import Translator
from utils import *
from torch.autograd import Variable
from train import memory_efficient
from optimizer import Optim

from bleu import *


class Trainer:

    # src: (max_slen_batch, batch_size, emb)
    # gold: (max_tlen_batch, batch_size, emb)
    def __init__(self, nmtModel, sv, tv, optim, trg_dict_size, n_critic=1):

        self.nmtModel = nmtModel
        self.sv = sv
        self.tv = tv
        self.optim = optim
        self.trg_dict_size = trg_dict_size

        self.n_critic = 1#n_critic

        self.translator = Translator(nmtModel, sv, tv, beam=10)

        self.optim_G = Optim(
            'adadelta', 1.0, wargs.max_grad_norm,
            learning_rate_decay=wargs.learning_rate_decay,
            start_decay_from=wargs.start_decay_from,
            last_valid_bleu=wargs.last_valid_bleu
        )

        self.tao = 1

        self.optim.init_optimizer(self.nmtModel.parameters())
        self.optim_G.init_optimizer(self.nmtModel.parameters())

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
            # (L, B)
            dist = tc.sum(KL * y_masks)

            W_KL = KL / y_masks.sum(0).expand_as(KL)
            W_dist = tc.sum(W_KL * y_masks)

        elif type == 'KL-sent':

            #print p1[0]
            #print p2[0]
            #print '-----------------------------'
            p1 = tc.gather(p1, 2, y_gold.unsqueeze(2).expand_as(p1))[:, :, 0]
            p2 = tc.gather(p2, 2, y_gold.unsqueeze(2).expand_as(p2))[:, :, 0]
            # p1 (max_tlen_batch, batch_size)
            #print (p2 < 1) == False

            dist = tc.sum((y_masks * tc.log(p1) - y_masks * tc.log(p2)) * p1).squeeze()
            # KL: (1, batch_size)

        return dist / y_masks.size(1), W_dist / y_masks.size(1)

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

    def hyps_padding_dist(self, B, hyps_L, y_maxL, p_y_hpy):

        hyps_dist = [None] * B
        for bid in range(B):

            hyp_L = hyps_L[bid]
            one_p_y_hpy = p_y_hpy[:, bid, :]

            if hyp_L < y_maxL:
                pad = tc.ones(y_maxL - hyp_L) / self.trg_dict_size
                pad = pad.unsqueeze(-1).expand((pad.size(0), one_p_y_hpy.size(-1)))
                if wargs.gpu_id and not pad.is_cuda: pad = pad.cuda()
                #print one_p_y_hpy.size(0), pad.size(0)
                one_p_y_hpy.data[hyp_L:] = pad

            hyps_dist[bid] = one_p_y_hpy

        hyps_dist = tc.stack(tuple(hyps_dist), dim=1)

        return hyps_dist

    def get_g(self, LB, V, eps=1e-30):

        vg = Variable(
            -tc.log(-tc.log(tc.Tensor(LB, V).uniform_(0, 1) + eps) + eps), requires_grad=False)

        return vg * 0

    def gumbel_sampling(self, B, y_maxL, output):

        if output.is_cuda: output = output.cpu()
        # output (L * B, V)
        if output.dim() == 3: output = output.view(-1, output.size(-1))
        g = self.get_g(output.size(0), self.trg_dict_size)
        hyps = tc.max(g + output, 1)[1]
        # hyps (L*B, 1)
        hyps = hyps.view(y_maxL, B)
        hyps[0] = const.BOS * tc.ones(B).long()   # first words are <s>
        # hyps (L, B)
        c1 = tc.clamp((hyps.data - const.EOS), min=0, max=self.trg_dict_size)
        c2 = tc.clamp((const.EOS - hyps.data), min=0, max=self.trg_dict_size)
        _hyps = c1 + c2
        _hyps = tc.cat([_hyps, tc.zeros(B).long().unsqueeze(0)], 0)
        _hyps = tc.min(_hyps, 0)[1]
        #_hyps = tc.max(0 - _hyps, 0)[1]
        # idx: (1, B)
        hyps_L = _hyps.view(-1).tolist()
        hyps_mask = tc.zeros(y_maxL, B)
        for bid in range(B): hyps_mask[:, bid][:hyps_L[bid]] = 1.
        hyps_mask = Variable(hyps_mask, requires_grad=False)

        if wargs.gpu_id and not hyps_mask.is_cuda: hyps_mask = hyps_mask.cuda()
        if wargs.gpu_id and not hyps.is_cuda: hyps = hyps.cuda()
        if wargs.gpu_id and not g.is_cuda: g = g.cuda()

        return g, hyps, hyps_mask, hyps_L

    def try_trans(self, srcs, ref):

        # (len, 1)
        #src = sent_filter(list(srcs[:, bid].data))
        x_filter = sent_filter(list(srcs))
        y_filter = sent_filter(list(ref))
        #wlog('\n[{:3}] {}'.format('Src', idx2sent(x_filter, self.sv)))
        #wlog('[{:3}] {}'.format('Ref', idx2sent(y_filter, self.tv)))

        onebest, onebest_ids = self.translator.trans_onesent(x_filter)

        #wlog('[{:3}] {}'.format('Out', onebest))

        return onebest_ids


    def beamsearch_sampling(self, srcs, x_masks, ref, y_maxL):

        # y_masks: (trg_max_len, batch_size)
        B = srcs.size(1)
        hyps, hyps_L = [None] * B, [None] * B
        for bid in range(B):

            onebest_ids = self.try_trans(srcs[:, bid].data, ref[:, bid].data)

            if len(onebest_ids) == 0 or onebest_ids[0] != const.BOS:
                onebest_ids = [const.BOS] + onebest_ids
            if onebest_ids[-1] == const.EOS: onebest_ids = onebest_ids[:-1]

            hyp_L = len(onebest_ids)
            hyps_L[bid] = hyp_L

            onebest_ids = tc.Tensor(onebest_ids).long()

            if hyp_L < y_maxL:
                hyps[bid] = tc.cat(
                    tuple([onebest_ids, const.PAD * tc.ones(y_maxL - hyps_L[bid]).long()]), 0)
            else:
                hyps[bid] = onebest_ids[:y_maxL]

        hyps = tc.stack(tuple(hyps), dim=1)

        if wargs.gpu_id and not hyps.is_cuda: hyps = hyps.cuda()
        hyps = Variable(hyps, requires_grad=False)
        hyps_mask = hyps.ne(const.PAD).float()

        return hyps, hyps_mask, hyps_L

    def train(self, train_data, name='default'):

        loss_val = 0.
        batch_count = len(train_data)
        self.nmtModel.train()

        for eid in range(wargs.start_epoch, wargs.max_epochs + 1):

            for bid in range(batch_count):

                _, srcs, trgs, slens, srcs_m, trgs_m = train_data[bid]
                gold_feed, gold_feed_mask = trgs[:-1], trgs_m[:-1]
                B, y_maxL = srcs.size(1), gold_feed.size(0)
                N = trgs[1:].data.ne(const.PAD).sum()

                trgs_list = numpy_to_str(gold_feed.cpu(),
                                         gold_feed_mask.cpu().data.numpy().sum(0).tolist())

                wlog('Train Discrimitor .......... {}'.format(name))
                for j in range(self.n_critic):

                    self.nmtModel.zero_grad()
                    #self.optim.zero_grad()

                    o1 = self.nmtModel(srcs, gold_feed, srcs_m, gold_feed_mask)
                    p_y_gold = self.nmtModel.classifier.logit_to_prob(o1)
                    # p_y_gold: (gold_max_len - 1, B, trg_dict_size)

                    #logit = self.nmtModel.classifier.get_a(o1)
                    #g, hyps, hyps_mask, hyps_L = self.gumbel_sampling(B, y_maxL, logit)
                    hyps, hyps_mask, hyps_L = self.beamsearch_sampling(srcs, srcs_m, trgs, y_maxL)

                    #print hyps
                    o_hyps = self.nmtModel(srcs, hyps, srcs_m, hyps_mask)
                    #print o_hyps
                    #p_y_hpy = self.nmtModel.classifier.logit_to_prob(o_hyps, g, self.tao)
                    p_y_hpy = self.nmtModel.classifier.logit_to_prob(o_hyps)
                    #print p_y_hpy
                    p_y_hpy = self.hyps_padding_dist(B, hyps_L, y_maxL, p_y_hpy)
                    #print 'aaaaaaaaaaaaaaaaaaaaaaaaa'
                    #print p_y_gold.size()
                    #print p_y_hpy.size()
                    #print hyps_mask.size()
                    #loss_D = -self.distance(p_y_gold, p_y_hpy, hyps_mask, type='KL')
                    loss_D, w_loss_D = self.distance(p_y_gold, p_y_hpy, hyps_mask,
                                                     type='KL', y_gold=trgs[1:])
                    wlog('Discrimitor KL distance {}'.format(w_loss_D.data[0]))

                    #loss_D.div(B).backward(retain_variables=True)
                    (1 * loss_D).div(B).backward()
                    self.optim.step()
                    del o1, p_y_gold, p_y_hpy, hyps_mask

                wlog('Train generator .......... ')
                #o1 = self.nmtModel(srcs, gold_feed, srcs_m, gold_feed_mask)
                #p_y_gold = self.nmtModel.classifier.prob(o1)

                #p_y_hpy2, hyps_mask = self.beamsearch_sampling(srcs, srcs_m,
                #                                     gold_feed, gold_feed_mask.size(0))
                #print 'aaaaaaaaaaaaaaaaaaaaaaaaa'
                #print p_y_gold.size()
                #print p_y_hpy2.size()
                #print hyps_mask.size()
                # padding hyps
                #loss_G = self.distance(p_y_gold, p_y_hpy2, hyps_mask, type='KL')
                #loss_G.div(batch_size).backward(retain_variables=True)
                #loss_G.div(batch_size).backward()

                wlog('MLE ... feed gold .... train ...')
                for i in range(5):

                    self.nmtModel.zero_grad()
                    outputs = self.nmtModel(srcs, gold_feed, srcs_m, gold_feed_mask)
                    #print outputs
                    batch_loss, grad_output, batch_correct_num = memory_efficient(
                        outputs, trgs[1:], trgs_m[1:], self.nmtModel.classifier)
                    #print batch_loss
                    #print grad_output
                    outputs.backward(grad_output)

                    #loss, correct_num = self.nmtModel.classifier(o1, trgs[1:], trgs_m[1:])
                    wlog('Epo:{:>2}/{:>2}, Bat:[{}/{}], W-MLE:{:4.2f}, W-ppl:{:4.2f}, '
                         'S-MLE:{:4.2f}'.format(eid, wargs.max_epochs, bid, batch_count,
                                                batch_loss/N, math.exp(batch_loss/N), batch_loss/B))

                    #loss.div(batch_size).backward()
                    #(loss_G + loss).div(batch_size).backward()

                    self.optim_G.step()
                    #del loss, correct_num
                    #del o1, p_y_gold, p_y_hpy2, hyps_mask
                    del outputs, batch_loss, batch_correct_num

                wlog('MLE ... feed onebest ... train ...')
                for i in range(5):

                    self.nmtModel.zero_grad()
                    hyps, hyps_mask, hyps_L = self.beamsearch_sampling(srcs, srcs_m, trgs, y_maxL)
                    outputs = self.nmtModel(srcs, hyps, srcs_m, hyps_mask)
                    #print outputs
                    batch_loss, grad_output, batch_correct_num = memory_efficient(
                        outputs, trgs[1:], trgs_m[1:], self.nmtModel.classifier)
                    #print batch_loss
                    #print grad_output
                    outputs.backward(grad_output)
                    #loss, correct_num = self.nmtModel.classifier(o1, trgs[1:], trgs_m[1:])
                    wlog('Epo:{:>2}/{:>2}, Bat:[{}/{}], W-MLE:{:4.2f}, W-ppl:{:4.2f}, '
                         'S-MLE:{:4.2f}'.format(eid, wargs.max_epochs, bid, batch_count,
                                                batch_loss/N, math.exp(batch_loss/N), batch_loss/B))
                    self.optim_G.step()
                    del outputs, batch_loss, batch_correct_num


                #for k in range(batch_size):
                #    onebest_ids = self.try_trans(srcs[:, k].data, gold_feed[:, k].data)
                hyps, hyps_mask, hyps_L = self.beamsearch_sampling(srcs, srcs_m, trgs, y_maxL)
                hyps_list = numpy_to_str(hyps.cpu(), hyps_L)
                bleus = []
                for hyp, ref in zip(hyps_list, trgs_list):
                    #print bleu(hypo_c="today weather very good", refs_c=["today weather good", "would rain"],n=4)
                    bleus.append(bleu(hyp, [ref]))
                wlog()




