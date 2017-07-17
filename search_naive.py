from __future__ import division

from utils import *
import numpy
import copy

from search_bs import Func


class ORI(Func):

    def __init__(self, tvcb_i2w=None, k=10, ptv=None):

        self.lqc = [0] * 10
        super(ORI, self).__init__(self.lqc)

        self.tvcb_i2w = tvcb_i2w
        self.k = k
        self.ptv = ptv

    def original_trans(self, x):

        x = x[0] if self.ifvalid else x  # numpy ndarray
        # subdict set [0,2,6,29999, 333]
        self.ptv = numpy.asarray(x[1], dtype='int32') if self.ifvalid else None

        # k is the beam size we have
        x = numpy.asarray(x, dtype='int64')
        if x.ndim == 1:
            x = x[None, :]
        src_sent_len = x.shape[1]
        maxlen = src_sent_len * 2
        x = x.T

        sample = []
        sample_score = []

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = []

        # get initial state of decoder rnn and encoder context
        s_im1, ctx0, c_x0 = self.fn_init(x)
        y_im1 = [-1]  # indicator for the first target word (bos target)

        for ii in xrange(maxlen):
            # (src_sent_len, 1, 2*src_nhids) -> (src_sent_len, live_k, 2*src_nhids)
            ctx = numpy.tile(ctx0, [live_k, 1])
            c_x = numpy.tile(c_x0, [live_k, 1])
            yemb_im1, hi = self.fn_nh(y_im1, s_im1)
            pi, ai = self.fn_na(ctx, c_x, hi)
            si = self.fn_ns(hi, ai)  # note, s_im1 should be updated!
            mo = self.fn_mo(yemb_im1, ai, si)
            next_scores = self.fn_pws(mo, self.ptv)  # the larger the better

            next_ces = -next_scores if self.ifscore else self.fn_ce(next_scores)
            #cand_scores = hyp_scores[:, None] - numpy.log(next_scores)
            cand_scores = hyp_scores[:, None] + next_ces
            # print ii, ' ==============================================='
            # print next_scores
            # print ii, ' ==============================================='
            # print cand_scores
            cand_flat = cand_scores.flatten()
            # ranks_flat = cand_flat.argsort()[:(k-dead_k)]
            # we do not need to generate k candidate here, because we just need to generate k-dead_k
            # more candidates ending with eos, so for each previous candidate we just need to expand
            # k-dead_k candidates
            ranks_flat = part_sort(cand_flat, self.k - dead_k)
            # print ranks_flat, cand_flat[ranks_flat[1]], cand_flat[ranks_flat[8]]

            voc_size = next_scores.shape[1]
            trans_indices = ranks_flat // voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(self.k - dead_k).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(s_im1[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            # current beam, if the hyposise ends with eos, we do not
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == self.eos_id:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    # print new_hyp_scores[idx], new_hyp_samples[idx]
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k
            # print ii, '====================================================='
            # print 'hyp_scores:'
            # print hyp_scores
            # print 'hyp_samples:'
            # for hyp_sample in hyp_samples:
            #    print hyp_sample

            if new_live_k < 1:
                break
            if dead_k >= self.k:
                break

            y_im1 = numpy.array([w[-1] for w in hyp_samples])
            s_im1 = numpy.array(hyp_states)

        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

        if self.ifnorm:
            lengths = numpy.array([len(s) for s in sample])
            avg_sample_score = sample_score / lengths
        else:
            avg_sample_score = sample_score
        sidx = numpy.argmin(avg_sample_score)

        best_sum_loss = sample_score[sidx]
        best_avg_loss = avg_sample_score[sidx]
        best_trans = sample[sidx]

        log('@source length[{}], translation length(with eos)[{}], maxlen[{}], avg loss'
            '[{}]={}/{}'.format(src_sent_len, len(best_trans), maxlen, avg_sample_score[sidx],
                                sample_score[sidx], lengths[sidx]))
        log('init[{}] nh[{}] na[{}] ns[{}] mo[{}] ws[{}] ps[{}] p[{}]'.format(*self.lqc))
        return _filter_reidx(best_trans, self.tvcb_i2w, self.ifmv, self.ptv)
