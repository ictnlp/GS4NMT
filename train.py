from __future__ import division

import os
import sys
import math
import time
import numpy as np
import torch as tc
from torch.autograd import Variable
import wargs
import const
import subprocess
from utils import wlog
from cp_sample import Translator


#save model and evaluate the bleu on validation set
#test_data: {'nist03': Input, ...}
def mt_eval(valid_data, model, sv, tv, eid, bid, optim, tests_data):

    model_state_dict = model.state_dict()
    model_state_dict = {k: v for k, v in model_state_dict.items() if 'classifier' not in k}
    class_state_dict = model.classifier.state_dict()
    model_dict = {
        'model': model_state_dict,
        'class': class_state_dict,
        'epoch': eid,
        'batch': bid,
        'optim': optim
    }

    if wargs.save_one_model:
        model_file = '{}.pt'.format(wargs.model_prefix)
    else:
        model_file = '{}_e{}_upd{}.pt'.format(wargs.model_prefix, eid, bid)

    tc.save(model_dict, model_file)

    tor = Translator(model, sv, tv)

    return tor.trans_eval(valid_data, eid, bid, model_file, tests_data)

'''
def memory_efficient(outputs, gold, gold_mask, classifier):

    batch_loss, batch_correct_num = 0, 0
    outputs = Variable(outputs.data, requires_grad=True, volatile=False)
    cur_batch_count = outputs.size(1)

    os_split = tc.split(outputs, wargs.max_gen_batches)
    gs_split = tc.split(gold, wargs.max_gen_batches)
    ms_split = tc.split(gold_mask, wargs.max_gen_batches)

    for i, (o_split, g_split, m_split) in enumerate(zip(os_split, gs_split, ms_split)):

        g_split, m_split = g_split.view(-1), m_split.view(-1)
        o_split_flat = o_split.view(-1, o_split.size(2))
        scores_flat = classifier.map_vocab(classifier.dropout(o_split_flat))
        #scores_flat = classifier.map1(o_split_flat)
        #scores_flat = classifier.map2(scores_flat)
        # negative likelihood log
        scores_flat = classifier.log_prob(scores_flat)

        # accuracy
        pred_correct = (scores_flat.max(dim=-1)[1].squeeze()).eq(
            g_split).masked_select(g_split.ne(const.PAD)).sum()
        batch_correct_num += pred_correct.data[0]

        scores_flat = scores_flat * m_split.unsqueeze(-1).expand_as(scores_flat)
        loss = classifier.criterion(scores_flat, g_split)
        batch_loss += loss.data[0]
        loss.div(cur_batch_count).backward()
        del o_split_flat, scores_flat, loss, pred_correct

    grad_output = None if outputs.grad is None else outputs.grad.data
    return batch_loss, grad_output, batch_correct_num
'''

def memory_efficient(outputs, gold, gold_mask, classifier):

    batch_loss, batch_correct_num = 0, 0
    outputs = Variable(outputs.data, requires_grad=True, volatile=False)
    cur_batch_count = outputs.size(1)

    os_split = tc.split(outputs, wargs.max_gen_batches)
    gs_split = tc.split(gold, wargs.max_gen_batches)
    ms_split = tc.split(gold_mask, wargs.max_gen_batches)

    for i, (o_split, g_split, m_split) in enumerate(zip(os_split, gs_split, ms_split)):

        loss, correct_num = classifier(o_split, g_split, m_split)
        batch_loss += loss.data[0]
        batch_correct_num += correct_num.data[0]
        loss.div(cur_batch_count).backward()
        del loss, correct_num

    grad_output = None if outputs.grad is None else outputs.grad.data
    return batch_loss, grad_output, batch_correct_num

def train(model, train_data, valid_data, tests_data, vocab_data, optim):

    wlog('Start training ... ')
    assert wargs.sample_size < wargs.batch_size, 'Batch size < sample count'
    # [low, high)
    batch_count = len(train_data)
    batch_start_sample = tc.randperm(batch_count)[0]
    wlog('Randomly select {} samples in the {}th/{} batch'.format(
        wargs.sample_size, batch_start_sample, batch_count))

    bidx, eval_cnt = 0, [0]

    model.train()

    sv = vocab_data['src'].idx2key
    tv = vocab_data['trg'].idx2key

    train_start = time.time()
    wlog('')
    wlog('#####################################################################################')
    wlog('############################################################ Start training #########')
    wlog('#####################################################################################')

    for epoch in range(wargs.start_epoch, wargs.max_epochs + 1):

        epoch_start = time.time()

        # train for one epoch on the training data
        wlog('\n#########################################################', False)
        wlog(' Epoch [{}/{}] '.format(epoch, wargs.max_epochs), False)
        wlog('#########################################################')

        if wargs.epoch_shuffle and epoch > wargs.epoch_shuffle_minibatch:
            train_data.shuffle()

        # shuffle the original batch
        shuffled_batch_idx = tc.randperm(batch_count)

        sample_size = wargs.sample_size
        epoch_loss, epoch_trg_words, epoch_num_correct = 0, 0, 0
        show_loss, show_src_words, show_trg_words, show_correct_num = 0, 0, 0, 0
        sample_spend, eval_spend, epoch_bidx = 0, 0, 0
        show_start = time.time()

        for k in range(batch_count):

            bidx += 1
            epoch_bidx = k + 1
            batch_idx = shuffled_batch_idx[k] if epoch >= wargs.epoch_shuffle_minibatch else k

            # (max_slen_batch, batch_size)
            _, srcs, trgs, slens, srcs_m, trgs_m = train_data[batch_idx]

            # (max_slen_batch, batch_size, src_wemb_size)
            #srcs, trgs = model.src_lookup_table(srcs.cpu()), model.trg_lookup_table(trgs.cpu())
            #srcs, trgs = srcs.cuda(), trgs.cuda()
            # (max_tlen_batch, batch_size, trg_wemb_size)

            model.zero_grad()
            # (max_tlen_batch - 1, batch_size, out_size)
            outputs = model(srcs, trgs[:-1], srcs_m, trgs_m[:-1])
            this_bnum = outputs.size(1)

            batch_loss, grad_output, batch_correct_num = memory_efficient(
                outputs, trgs[1:], trgs_m[1:], model.classifier)

            outputs.backward(grad_output)
            optim.step()
            del outputs, grad_output

            #print model.decoder.trg_lookup_table.weight
            #print model.classifier.map2.weight

            batch_src_words = srcs.data.ne(const.PAD).sum()
            assert batch_src_words == slens.data.sum()
            batch_trg_words = trgs[1:].data.ne(const.PAD).sum()

            show_loss += batch_loss
            show_correct_num += batch_correct_num
            epoch_loss += batch_loss
            epoch_num_correct += batch_correct_num
            show_src_words += batch_src_words
            show_trg_words += batch_trg_words
            epoch_trg_words += batch_trg_words

            if epoch_bidx % wargs.display_freq == 0:
                #print show_correct_num, show_loss, show_trg_words, show_loss/show_trg_words
                ud = time.time() - show_start - sample_spend - eval_spend
                wlog(
                    'Epo:{:>2}/{:>2} |[{:^5} {:^5} {:^5}k] |acc:{:5.2f}% |ppl:{:4.2f} '
                    '|stok/s:{:>4}/{:>2}={:>2} |ttok/s:{:>2} '
                    '|stok/sec:{:6.2f} |ttok/sec:{:6.2f} |elapsed:{:4.2f}/{:4.2f}m'.format(
                        epoch, wargs.max_epochs, epoch_bidx, batch_idx, bidx/1000,
                        (show_correct_num / show_trg_words) * 100,
                        math.exp(show_loss / show_trg_words),
                        batch_src_words, this_bnum, int(batch_src_words / this_bnum),
                        int(batch_trg_words / this_bnum),
                        show_src_words / ud, show_trg_words / ud, ud,
                        (time.time() - train_start) / 60.)
                )
                show_loss, show_src_words, show_trg_words, show_correct_num = 0, 0, 0, 0
                sample_spend, eval_spend = 0, 0
                show_start = time.time()

            if epoch_bidx % wargs.sampling_freq == 0:

                sample_start = time.time()
                tor = Translator(model, sv, tv)

                # (max_len_batch, batch_size)
                sample_src_tensor = srcs.t()[:sample_size]
                sample_trg_tensor = trgs.t()[:sample_size]
                tor.trans_samples(sample_src_tensor, sample_trg_tensor)
                sample_spend = time.time() - sample_start

            # Just watch the translation of some source sentences in training data
            if wargs.if_fixed_sampling and bidx == batch_start_sample:
                # randomly select sample_size sample from current batch
                rand_rows = np.random.choice(this_bnum, sample_size, replace=False)
                sample_src_tensor = tc.Tensor(sample_size, srcs.size(0)).long()
                sample_src_tensor.fill_(const.PAD)
                sample_trg_tensor = tc.Tensor(sample_size, trgs.size(0)).long()
                sample_trg_tensor.fill_(const.PAD)

                for id in xrange(sample_size):
                    sample_src_tensor[id, :] = srcs.t()[rand_rows[id], :]
                    sample_trg_tensor[id, :] = trgs.t()[rand_rows[id], :]

                if wargs.with_mv:
                    fix_npv = npv
                    fix_npv_true = npv_true

            if wargs.epoch_eval is not True and epoch_bidx > wargs.eval_valid_from and \
               epoch_bidx % wargs.eval_valid_freq == 0:

                eval_start = time.time()
                eval_cnt[0] += 1
                wlog('Among epoch, batch [{}], [{}] eval save model ...'.format(
                    epoch_bidx, eval_cnt[0]))

                mt_eval(valid_data, model, sv, tv, epoch, epoch_bidx, optim, tests_data)

                eval_spend = time.time() - eval_start

        avg_epoch_loss = epoch_loss / epoch_trg_words
        avg_epoch_acc = epoch_num_correct / epoch_trg_words
        wlog('\nEnd epoch [{}]'.format(epoch))
        wlog('Train accuracy {:4.2f}%'.format(avg_epoch_acc * 100))
        wlog('Average loss {:4.2f}'.format(avg_epoch_loss))
        wlog('Train perplexity: {0:4.2f}'.format(math.exp(avg_epoch_loss)))

        wlog('End epoch, batch [{}], [{}] eval save model ...'.format(epoch_bidx, eval_cnt[0]))
        mteval_bleu = mt_eval(valid_data, model,
                              sv, tv, epoch, epoch_bidx, optim, tests_data)
        optim.update_learning_rate(mteval_bleu, epoch)

        epoch_time_consume = time.time() - epoch_start
        wlog('Consuming: {:4.2f}s'.format(epoch_time_consume))

    wlog('Finish training, comsuming {:6.2} hours'.format((time.time() - train_start) / 3600))
    wlog('Congratulations!')

