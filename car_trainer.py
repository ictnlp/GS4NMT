import wargs
import torch as tc
import math
from translate import Translator
from utils import *
from torch.autograd import Variable
from train import memory_efficient
from optimizer import Optim

from bleu import *
from train import mt_eval

class Trainer:

    def __init__(self, nmtModel, sv, tv, optim, trg_dict_size, n_critic=1):

        self.lamda = 5
        self.eps = 1e-20
        self.beta_KL = 0.005
        self.beta_RLGen = 0.1
        self.clip_rate = 0.2
        self.beta_RLBatch = 0.2
        self.gumbeling = False

        self.nmtModel = nmtModel
        self.sv = sv
        self.tv = tv
        self.optim = optim
        self.trg_dict_size = trg_dict_size

        self.n_critic = 1#n_critic

        self.translator_sample = Translator(self.nmtModel, sv, tv, k=1, noise=False)
        #self.translator = Translator(nmtModel, sv, tv, k=10)

        self.optim_G = Optim(
            'adam', 10e-05, wargs.max_grad_norm,
            learning_rate_decay=wargs.learning_rate_decay,
            start_decay_from=wargs.start_decay_from,
            last_valid_bleu=wargs.last_valid_bleu
        )

        self.optim_RL = Optim(
            'adadelta', 1.0, wargs.max_grad_norm,
            learning_rate_decay=wargs.learning_rate_decay,
            start_decay_from=wargs.start_decay_from,
            last_valid_bleu=wargs.last_valid_bleu
        )

        self.softmax = tc.nn.Softmax()

        #self.optim_G.init_optimizer(self.nmtModel.parameters())
        #self.optim_RL.init_optimizer(self.nmtModel.parameters())

    # p1: (max_tlen_batch, batch_size, vocab_size)
    def distance(self, p1, p2, y_masks, type='JS', y_gold=None):

        B = y_masks.size(1)
        hypo_N = y_masks.data.sum()

        if p2.size(0) > p1.size(0):

            p2 = p2[:(p1.size(0) + 1)]

        if type == 'JS':

            #D_kl = tc.mean(tc.sum((tc.log(p1) - tc.log(p2)) * p1, dim=-1).squeeze(), dim=0)
            M = (p1 + p2) / 2.
            D_kl1 = tc.sum((tc.log(p1) - tc.log(M)) * p1, dim=-1).squeeze()
            D_kl2 = tc.sum((tc.log(p2) - tc.log(M)) * p2, dim=-1).squeeze()
            Js = 0.5 * D_kl1 + 0.5 * D_kl2

            sent_batch_dist = tc.sum(Js * y_masks) / B

            Js = Js / y_masks.sum(0)[None, :]
            word_level_dist = tc.sum(Js * y_masks) / B
            del M, D_kl1, D_kl2, Js

        elif type == 'KL':

            KL = tc.sum((tc.log(p1 + self.eps) - tc.log(p2 + self.eps)) * p1, dim=-1)
            #KL = p1 + self.eps
            #KL = KL / (p2 + self.eps)
            #KL = KL.log()
            #KL = KL * p1
            #KL = KL.sum(-1)
            # (L, B)
            sent_batch_dist = tc.sum(KL * y_masks) / B
            word_level_dist0 = tc.sum(KL * y_masks) / hypo_N

            KL = KL / y_masks.sum(0)[None, :]
            #print W_KL.data
            word_level_dist1 = tc.sum(KL * y_masks) / B
            #print W_dist.data[0], y_masks.size(1)

            del KL

        elif type == 'KL-sent':

            #print p1[0]
            #print p2[0]
            #print '-----------------------------'
            p1 = tc.gather(p1, 2, y_gold[:, :, None])[:, :, 0]
            p2 = tc.gather(p2, 2, y_gold[:, :, None])[:, :, 0]
            # p1 (max_tlen_batch, batch_size)
            #print (p2 < 1) == False

            KL = (y_masks * (tc.log(p1) - tc.log(p2))) * p1

            sent_batch_dist = tc.sum(KL) / B

            KL = KL / y_masks.sum(0)[None, :]
            word_level_dist = tc.sum(KL * y_masks) / B
            # KL: (1, batch_size)
            del p1, p2, KL

        return sent_batch_dist, word_level_dist0, word_level_dist1

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

    def hyps_padding_dist(self, B, hyps_L, y_maxL, p_y_hyp):

        hyps_dist = [None] * B
        for bid in range(B):

            hyp_L = hyps_L[bid]
            one_p_y_hyp = p_y_hyp[:, bid, :]

            if hyp_L < y_maxL:
                pad = tc.ones(y_maxL - hyp_L) / self.trg_dict_size
                pad = pad[:, None].expand((pad.size(0), one_p_y_hyp.size(-1)))
                if wargs.gpu_id and not pad.is_cuda: pad = pad.cuda()
                #print one_p_y_hyp.size(0), pad.size(0)
                one_p_y_hyp.data[hyp_L:] = pad

            hyps_dist[bid] = one_p_y_hyp

        hyps_dist = tc.stack(tuple(hyps_dist), dim=1)

        return hyps_dist

    def gumbel_sampling(self, B, y_maxL, feed_gold_out, noise=False):

        # feed_gold_out (L * B, V)
        logit = self.nmtModel.classifier.get_a(feed_gold_out, noise=noise)

        if logit.is_cuda: logit = logit.cpu()
        hyps = tc.max(logit, 1)[1]
        # hyps (L*B, 1)
        hyps = hyps.view(y_maxL, B)
        hyps[0] = BOS * tc.ones(B).long()   # first words are <s>
        # hyps (L, B)
        c1 = tc.clamp((hyps.data - EOS), min=0, max=self.trg_dict_size)
        c2 = tc.clamp((EOS - hyps.data), min=0, max=self.trg_dict_size)
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

        return hyps, hyps_mask, hyps_L

    def try_trans(self, srcs, ref):

        # (len, 1)
        #src = sent_filter(list(srcs[:, bid].data))
        x_filter = sent_filter(list(srcs))
        y_filter = sent_filter(list(ref))
        #wlog('\n[{:3}] {}'.format('Src', idx2sent(x_filter, self.sv)))
        #wlog('[{:3}] {}'.format('Ref', idx2sent(y_filter, self.tv)))

        onebest, onebest_ids, _ = self.translator_sample.trans_onesent(x_filter)

        #wlog('[{:3}] {}'.format('Out', onebest))

        # no EOS and BOS
        return onebest_ids


    def beamsearch_sampling(self, srcs, x_masks, ref, maxL, eos=True):

        # y_masks: (trg_max_len, batch_size)
        B = srcs.size(1)
        hyps, hyps_L = [None] * B, [None] * B
        for bid in range(B):

            onebest_ids = self.try_trans(srcs[:, bid].data, ref[:, bid].data)

            if len(onebest_ids) == 0 or onebest_ids[0] != BOS:
                onebest_ids = [BOS] + onebest_ids
            if eos is True:
                if not onebest_ids[-1] == EOS:
                    onebest_ids = onebest_ids + [EOS]

            hyp_L = len(onebest_ids)
            hyps_L[bid] = hyp_L

            onebest_ids = tc.Tensor(onebest_ids).long()

            if hyp_L <= maxL:
                hyps[bid] = tc.cat(
                    tuple([onebest_ids, PAD * tc.ones(maxL - hyps_L[bid]).long()]), 0)
            else:
                hyps[bid] = onebest_ids[:maxL]

        hyps = tc.stack(tuple(hyps), dim=1)

        if wargs.gpu_id and not hyps.is_cuda: hyps = hyps.cuda()
        hyps = Variable(hyps, requires_grad=False)
        hyps_mask = hyps.ne(PAD).float()

        return hyps, hyps_mask, hyps_L

    def train(self, dh, train_data, k, valid_data=None, tests_data=None,
              merge=False, name='default', percentage=0.1):

        #if (k + 1) % 1 == 0 and valid_data and tests_data:
        #    wlog('Evaluation on dev ... ')
        #    mt_eval(valid_data, self.nmtModel, self.sv, self.tv,
        #            0, 0, [self.optim, self.optim_RL, self.optim_G], tests_data)

        batch_count = len(train_data)
        self.nmtModel.train()

        self.optim_G.init_optimizer(self.nmtModel.parameters())
        self.optim_RL.init_optimizer(self.nmtModel.parameters())

        for eid in range(wargs.start_epoch, wargs.max_epochs + 1):

            #self.optim_G.init_optimizer(self.nmtModel.parameters())
            #self.optim_RL.init_optimizer(self.nmtModel.parameters())

            size = int(percentage * batch_count)
            shuffled_batch_idx = tc.randperm(batch_count)

            wlog('{}, Epo:{:>2}/{:>2} start, random {}/{}({:.2%}) calc BLEU '.format(
                name, eid, wargs.max_epochs, size, batch_count, percentage), False)
            wlog('-' * 20)
            param_1, param_2, param_3, param_4, param_5, param_6 = [], [], [], [], [], []
            for k in range(size):
                bid, half_size = shuffled_batch_idx[k], wargs.batch_size

                # srcs: (max_sLen_batch, batch_size, emb), trgs: (max_tLen_batch, batch_size, emb)
                if merge is False: _, srcs, trgs, slens, srcs_m, trgs_m = train_data[bid]
                else: _, srcs, trgs, slens, srcs_m, trgs_m = dh.merge_batch(train_data[bid])[0]

                hyps, hyps_mask, hyps_L = self.beamsearch_sampling(srcs, srcs_m, trgs, 100)

                param_1.append(LBtensor_to_Str(hyps[1:].cpu(), [l-1 for l in hyps_L]))
                param_2.append(LBtensor_to_Str(trgs[1:].cpu(),
                                               trgs_m[1:].cpu().data.numpy().sum(0).tolist()))

                param_3.append(LBtensor_to_Str(hyps[1:, :half_size].cpu(),
                                               [l-1 for l in hyps_L[:half_size]]))
                param_4.append(LBtensor_to_Str(trgs[1:, :half_size].cpu(),
                                               trgs_m[1:, :half_size].cpu().data.numpy().sum(0).tolist()))
                param_5.append(LBtensor_to_Str(hyps[1:, half_size:].cpu(),
                                               [l-1 for l in hyps_L[half_size:]]))
                param_6.append(LBtensor_to_Str(trgs[1:, half_size:].cpu(),
                                               trgs_m[1:, half_size:].cpu().data.numpy().sum(0).tolist()))

            start_bat_bleu_hist = bleu('\n'.join(param_3), ['\n'.join(param_4)])
            start_bat_bleu_new = bleu('\n'.join(param_5), ['\n'.join(param_6)])
            start_bat_bleu = bleu('\n'.join(param_1), ['\n'.join(param_2)])
            wlog('Random BLEU on history {}, new {}, mix {}'.format(
                start_bat_bleu_hist, start_bat_bleu_new, start_bat_bleu))

            wlog('Model selection and testing ... ')
            mt_eval(valid_data, self.nmtModel, self.sv, self.tv,
                    eid, 0, [self.optim, self.optim_RL, self.optim_G], tests_data)
            if start_bat_bleu > 0.9:
                wlog('Better BLEU ... go to next data history ...')
                return

            s_kl_seen, w_kl_seen0, w_kl_seen1, rl_gen_seen, rl_rho_seen, rl_bat_seen, w_mle_seen, s_mle_seen, \
                    ppl_seen = 0., 0., 0., 0., 0., 0., 0., 0., 0.
            for bid in range(batch_count):

                if merge is False: _, srcs, trgs, slens, srcs_m, trgs_m = train_data[bid]
                else: _, srcs, trgs, slens, srcs_m, trgs_m = dh.merge_batch(train_data[bid])[0]
                gold_feed, gold_feed_mask = trgs[:-1], trgs_m[:-1]
                gold, gold_mask = trgs[1:], trgs_m[1:]
                B, y_maxL = srcs.size(1), gold_feed.size(0)
                N = gold.data.ne(PAD).sum()
                wlog('{} {} {}'.format(B, y_maxL, N))

                trgs_list = LBtensor_to_StrList(trgs.cpu(), trgs_m.cpu().data.numpy().sum(0).tolist())

                ###################################################################################
                debug('Optimizing KL distance ................................ {}'.format(name))
                #self.nmtModel.zero_grad()
                self.optim.zero_grad()

                feed_gold_out = self.nmtModel(srcs, gold_feed, srcs_m, gold_feed_mask)
                p_y_gold = self.nmtModel.classifier.logit_to_prob(feed_gold_out)
                # p_y_gold: (gold_max_len - 1, B, trg_dict_size)

                if self.gumbeling is True:
                    hyps, hyps_mask, hyps_L = self.gumbel_sampling(B, y_maxL, feed_gold_out, True)
                else:
                    hyps, hyps_mask, hyps_L = self.beamsearch_sampling(srcs, srcs_m, trgs, y_maxL + 1)

                o_hyps = self.nmtModel(srcs, hyps[:-1], srcs_m, hyps_mask[:-1])
                p_y_hyp = self.nmtModel.classifier.logit_to_prob(o_hyps)
                p_y_hyp0 = self.hyps_padding_dist(B, hyps_L, y_maxL, p_y_hyp)
                #B_KL_loss = self.distance(p_y_gold, p_y_hyp0, hyps_mask[1:], type='KL', y_gold=gold)
                S_KL_loss, W_KL_loss0, W_KL_loss1 = self.distance(p_y_gold, p_y_hyp0, hyps_mask[1:], type='KL', y_gold=gold)
                wlog('KL distance between D(Hypo) and D(Gold): Sent-level {}, Word0-level {}, Word1-level {}'.format(
                    S_KL_loss.data[0], W_KL_loss0.data[0], W_KL_loss1.data[0]))
                s_kl_seen += S_KL_loss.data[0]
                w_kl_seen0 += W_KL_loss0.data[0]
                w_kl_seen1 += W_KL_loss1.data[0]

                del p_y_hyp

                ###################################################################################
                debug('Optimizing RL(Gen) .......... {}'.format(name))
                p_y_gold = p_y_gold.gather(2, gold[:, :, None])[:, :, 0]
                p_y_gold = ((p_y_gold + self.eps).log() * gold_mask).sum(0) / gold_mask.sum(0)

                hyps_list = LBtensor_to_StrList(hyps.cpu(), hyps_L)
                bleus_sampling = []
                for hyp, ref in zip(hyps_list, trgs_list): bleus_sampling.append(bleu(hyp, [ref]))
                bleus_sampling = to_Var(bleus_sampling)

                p_y_hyp = p_y_hyp0.gather(2, hyps[1:][:, :, None])[:, :, 0]
                p_y_hyp = ((p_y_hyp + self.eps).log() * hyps_mask[1:]).sum(0) / hyps_mask[1:].sum(0)

                r_theta = p_y_hyp / p_y_gold
                A = 1. - bleus_sampling
                RL_Gen_loss = tc.min(r_theta * A, clip(r_theta, self.clip_rate) * A).sum()
                RL_Gen_loss = (RL_Gen_loss).div(B)

                wlog('...... RL(Gen) cliped loss {}'.format(RL_Gen_loss.data[0]))
                rl_gen_seen += RL_Gen_loss.data[0]
                del p_y_gold, o_hyps, p_y_hyp

                ###################################################################################
                debug('Optimizing RL(Batch) -> Gap of MLE and BLEU ... rho ... feed onebest .... ')
                param_1 = LBtensor_to_Str(hyps[1:].cpu(), [l-1 for l in hyps_L])
                param_2 = LBtensor_to_Str(trgs[1:].cpu(),
                                          trgs_m[1:].cpu().data.numpy().sum(0).tolist())
                rl_bat_bleu = bleu(param_1, [param_2])

                p_y_hyp = p_y_hyp0.gather(2, hyps[1:][:, :, None])[:, :, 0]
                p_y_hyp = ((p_y_hyp + self.eps).log() * hyps_mask[1:]).sum(0) / hyps_mask[1:].sum(0)

                rl_avg_bleu = tc.mean(bleus_sampling).data[0]

                rl_rho = cor_coef(p_y_hyp, bleus_sampling, eps=self.eps)
                rl_rho_seen += rl_rho.data[0]   # must use data, accumulating Variable needs more memory

                #p_y_hyp = p_y_hyp.exp()
                #p_y_hyp = (p_y_hyp * self.lamda / 3).exp()
                #p_y_hyp = self.softmax(p_y_hyp)
                p_y_hyp = p_y_hyp[None, :]
                p_y_hyp_T = p_y_hyp.t().expand(B, B)
                p_y_hyp = p_y_hyp.expand(B, B)
                p_y_hyp_sum = p_y_hyp_T + p_y_hyp + self.eps

                #bleus_sampling = bleus_sampling[None, :].exp()
                bleus_sampling = self.softmax(self.lamda * bleus_sampling[None, :])
                bleus_T = bleus_sampling.t().expand(B, B)
                bleus = bleus_sampling.expand(B, B)
                bleus_sum = bleus_T + bleus + self.eps
                #print 'p_y_hyp_sum......................'
                #print p_y_hyp_sum.data
                RL_Batch_loss = p_y_hyp / p_y_hyp_sum * tc.log(bleus_T / bleus_sum) + \
                        p_y_hyp_T / p_y_hyp_sum * tc.log(bleus / bleus_sum)

                #RL_Batch_loss = tc.sum(-RL_Batch_loss * to_Var(1 - tc.eye(B))).div(B)
                RL_Batch_loss = tc.sum(-RL_Batch_loss * to_Var(1 - tc.eye(B)))

                wlog('RL(Batch) Mean BLEU: {}, rl_batch_loss: {}, rl_rho: {}, Bat BLEU: {}'.format(
                    rl_avg_bleu, RL_Batch_loss.data[0], rl_rho.data[0], rl_bat_bleu))
                rl_bat_seen += RL_Batch_loss.data[0]
                del hyps, hyps_mask, p_y_hyp, bleus_sampling, bleus, \
                        rl_rho, p_y_hyp_T, p_y_hyp_sum, bleus_T, bleus_sum

                (self.beta_KL * S_KL_loss + self.beta_RLGen * RL_Gen_loss + \
                        self.beta_RLBatch * RL_Batch_loss).backward(retain_graph=True)

                ###################################################################################
                mle_loss, grad_output, _ = memory_efficient(
                    feed_gold_out, gold, gold_mask, self.nmtModel.classifier)
                feed_gold_out.backward(grad_output)

                '''
                mle_loss, _ = self.nmtModel.classifier(feed_gold_out, gold, gold_mask)
                mle_loss = mle_loss.div(B)

                (self.beta_KL * KL_loss + self.beta_RLGen * RL_Gen_loss + \
                        self.beta_RLBatch * RL_Batch_loss + mle_loss).backward()
                '''

                w_mle_seen += mle_loss / N
                s_mle_seen += mle_loss / B
                ppl_seen += math.exp(mle_loss/N)
                wlog('Epo:{:>2}/{:>2}, Bat:[{}/{}], W-MLE:{:4.2f}, W-ppl:{:4.2f}, '
                     'S-MLE:{:4.2f}'.format(eid, wargs.max_epochs, bid, batch_count,
                                            mle_loss/N, math.exp(mle_loss/N), mle_loss/B))

                self.optim_G.step()

                del S_KL_loss, W_KL_loss0, W_KL_loss1, RL_Gen_loss, RL_Batch_loss, feed_gold_out

            wlog('End epoch: S-KL {}, W0-KL {}, W1-KL {}, S-RLGen {}, B-rho {}, B-RLBat {}, W-MLE {}, S-MLE {}, W-ppl {}'.format(
                s_kl_seen/batch_count, w_kl_seen0/batch_count, w_kl_seen1/batch_count, rl_gen_seen/batch_count,
                rl_rho_seen/batch_count, rl_bat_seen/batch_count, w_mle_seen/batch_count,
                s_mle_seen/batch_count, ppl_seen/batch_count))

