from __future__ import division

import time
import sys
import numpy

from utils import *
from collections import OrderedDict
import heapq
from itertools import count
import copy


class Wcp(object):

    def __init__(self, model, tvcb_i2w=None, k=10, thresh=100.0, lm=None, ngram=3, ptv=None):

        self.model = model
        self.tvcb_i2w = tvcb_i2w
        self.k = k
        self.thresh = thresh
        self.lm = lm
        self.ngram = ngram
        self.ptv = ptv

    def cube_prune_trans(self, s_list):

        self.lqc = [0] * 10
        self.cnt = count()
        self.locrt = [0] * 2
        self.beam = []
        self.translations = []

        self.maxlen = 2 * len(s_list)

        # (srcL, 1)
        s_tensor = tc.Tensor(s_list).long().unsqueeze(-1)

        # s_tensor: (len, 1), beamsize==1
        s_init, enc_src0, uh0 = self.model.init(s_tensor, test=True)
        # s_init: (1, trg_nhids), enc_src0: (srcL, 1, src_nhids*2), uh0: (srcL, 1, align_size)
        slen, enc_size, align_size = enc_src0.size(0), enc_src0.size(2), uh0.size(2)

        init_beam(self.beam, cnt=self.maxlen, s0=s_init)

        best_trans, best_loss = self.cube_pruning()

        log('@source[{}], translation(without eos)[{}], maxlen[{}], loss[{}]'.format(
            src_sent_len, len(best_trans), self.maxlen, best_loss))
        log('init[{}] nh[{}] na[{}] ns[{}] mo[{}] ws[{}] ps[{}] ce[{}]'.format(*self.lqc[0:8]))

        avg_merges = format(self.lqc[9] / self.lqc[8], '0.3f')
        log('average merge count[{}/{}={}]'.format(self.lqc[9],
                                                   self.lqc[8], avg_merges))

        return _filter_reidx(self.bos_id, self.eos_id, best_trans, self.tvcb_i2w,
                             self.ifmv, self.ptv)

    ##################################################################

    # NOTE: merge all candidates in previous beam by euclidean distance of two state vector or KL
    # distance of alignment probability

    ##################################################################

    def merge(self, bidx, eq_classes):

        prevb = self.beam[bidx - 1]
        len_prevb = len(prevb)
        used = []
        key = 0

        _memory = [None] * len_prevb
        _mem_p = [None] * len_prevb
        for j in range(len_prevb):  # index of each item in last beam
            if j in used:
                continue

            tmp = []
            if _memory[j]:
                _needed = _memory[j]
                score_im1_1, s_im1_1, y_im1_1, y_im2_1, y_im3_1, nj = _needed
                if self.ifwatch_adist and _mem_p[j]:
                    pi_1 = _mem_p[j]
                assert(j == nj)
            else:
                # calculation
                score_im1_1, s_im1_1, y_im1_1, bp_im1_1 = prevb[j]
                (y_im2_1, bp_im2_1) = (-1, -
                                       1) if bidx < 2 else self.beam[bidx - 2][bp_im1_1][-2:]
                y_im3_1 = -1 if bidx < 3 else self.beam[bidx - 3][bp_im2_1][-2]

                if self.ifwatch_adist:
                    hi_1 = self.fn_nh(y_im1_1, s_im1_1)
                    pi_1, ai_1 = self.fn_na(ctx0, hi_1)
                    _mem_p[j] = pi_1

                _needed = _memory[j] = (
                    score_im1_1, s_im1_1, y_im1_1, y_im2_1, y_im3_1, j)

            tmp.append(_needed)

            for jj in range(j + 1, len_prevb):

                if _memory[jj]:
                    _needed = _memory[jj]
                    score_im1_2, s_im1_2, y_im1_2, y_im2_2, y_im3_2, njj = _needed
                    if self.ifwatch_adist and _mem_p[jj]:
                        pi_2 = _mem_p[jj]
                    assert(jj == njj)
                else:   # calculation
                    score_im1_2, s_im1_2, y_im1_2, bp_im1_2 = prevb[jj]
                    (y_im2_2, bp_im2_2) = (-1, -1) if bidx < 2 else \
                        self.beam[bidx - 2][bp_im1_2][-2:]
                    y_im3_2 = -1 if bidx < 3 else self.beam[bidx - 3][bp_im2_2][-2]

                    if self.ifwatch_adist:
                        hi_2 = self.fn_nh(y_im1_2, s_im1_2)
                        pi_2, ai_2 = self.fn_na(ctx0, hi_2)
                        _mem_p[jj] = pi_2

                    _needed = _memory[jj] = (
                        score_im1_2, s_im1_2, y_im1_2, y_im2_2, y_im3_2, jj)

                if self.merge_way == 'Him1':

                    distance = euclidean(s_im1_2, s_im1_1)

                    print self.ngram
                    if self.ngram == 2:
                        debug('y11 y12 {} {}, {} {}'.format(y_im1_1, y_im1_2, distance,
                                                            self.thresh))
                        ifmerge = ((y_im1_2 == y_im1_1)
                                   and (distance < self.thresh))
                    elif self.ngram == 3:
                        debug('y21 y22 {} {}, y11 y12 {} {}, {} {}'.format(
                            y_im2_1, y_im2_2, y_im1_1, y_im1_2, distance, self.thresh))
                        ifmerge = ((y_im2_2 == y_im2_1) and (
                            y_im1_2 == y_im1_1) and (distance < self.thresh))
                    elif self.ngram == 4:
                        debug('y31 y32 {} {}, y21 y22 {} {}, y11 y12 {} {}, {} {}'.format(
                            y_im3_1, y_im3_2, y_im2_1, y_im2_2, y_im1_1, y_im1_2, distance,
                            self.thresh))
                        ifmerge = ((y_im3_2 == y_im3_1) and (y_im2_2 == y_im2_1)
                                   and (y_im1_2 == y_im1_1) and (distance < self.thresh))
                    else:
                        raise NotImplementedError

                elif self.merge_way == 'Hi':
                    raise NotImplementedError
                    ifmerge = (y_im1_2 == y_im1_1 and euclidean(
                        hi_2, hi_1) < self.thresh)
                elif self.merge_way == 'AiKL':
                    raise NotImplementedError
                    dist = kl_dist(pi_2, pi_1)
                    debug('attention prob kl distance: {}'.format(dist))
                    ifmerge = (y_im1_2 == y_im1_1 and dist < self.thresh)

                if ifmerge:
                    tmp.append(_needed)
                    used.append(jj)

                if self.ifwatch_adist:
                    dist = kl_dist(pi_2, pi_1)
                    debug('{} {} {}'.format(j, jj, dist))

            eq_classes[key] = tmp
            key += 1

    ##################################################################

    # NOTE: (Wen Zhang) create cube by sort row dimension

    ##################################################################

    #@exeTime
    def create_cube(self, bidx, eq_classes):
        # eq_classes: (score_im1, y_im1, hi, ai, loc_in_prevb) NEW
        cube = []
        cnt_transed = len(self.translations)
        for whichsubcub, leq_class in eq_classes.iteritems():   # sub cube

            each_subcube_rowsz = len(leq_class)
            score_im1_r0, s_im1_r0, y_im1, y_im2, y_im3, _ = leq_class[0]
            subcube = []
            _avg_si, _avg_hi, _avg_ai, _avg_scores_i = None, None, None, None

            if self.lm is not None and bidx >= 5:
                # TODO sort the row dimension by the distribution of next words
                # based on language model
                debug('-3 -2 -1 => {} {} {}'.format(y_im3, y_im2, y_im1))
                if self.ngram == 2:
                    gram = [y_im1]
                elif self.ngram == 3:
                    gram = [y_im2, y_im1]
                elif self.ngram == 4:
                    gram = [y_im3, y_im2, y_im1]
                else:
                    raise NotImplementedError

                lm_next_logps, next_wids = vocab_prob_given_ngram(
                    self.lm, gram, self.tvcb, self.tvcb_i2w)
                np_next_logps = numpy.asarray(lm_next_logps)
                np_next_wids = numpy.asarray(next_wids)

                np_next_neg_logps = -np_next_logps
                next_krank_ids = part_sort(
                    np_next_neg_logps, self.k - cnt_transed)
                next_krank_ces = np_next_neg_logps[next_krank_ids]
                next_krank_wids = np_next_wids[next_krank_ids]

                for idx in gram:
                    log(self.tvcb_i2w[idx] + ' ', nl=False)
                log('=> ', nl=False)
                for wid in next_krank_wids:
                    log(self.tvcb_i2w[wid] + ' ', nl=False)
                log('')

                if self.ifavg_att:

                    if each_subcube_rowsz == 1:
                        _avg_sim1 = s_im1_r0
                    else:
                        merged_sim1 = [tup[1] for tup in leq_class]
                        _avg_sim1 = numpy.mean(
                            numpy.array(merged_sim1), axis=0)
                        # for tup in leq_class: watch the attention prob pi
                        # dist here ....

                    _avg_hi = self.fn_nh(y_im1, _avg_sim1)
                    _avg_pi, _avg_ai = self.fn_na(self.context, _avg_hi)

                row_ksorted_ces = next_krank_ces

            else:
                # TODO sort the row dimension by average scores
                if not each_subcube_rowsz == 1:
                    merged_sim1 = [tup[1] for tup in leq_class]
                    _avg_sim1 = numpy.mean(numpy.array(merged_sim1), axis=0)
                else:
                    _avg_sim1 = s_im1_r0

                _avg_hi = self.fn_nh(y_im1, _avg_sim1)
                _, _avg_ai = self.fn_na(self.context, _avg_hi)
                _avg_si = self.fn_ns(_avg_hi, _avg_ai)
                _avg_moi = self.fn_mo(y_im1, _avg_ai, _avg_hi)
                _avg_scores_i = self.fn_pws(_avg_moi, self.ptv)  # the larger the better

                if self.ifscore:
                    _next_ces_flat = -_avg_scores_i.flatten()    # (1,vocsize) -> (vocsize,)
                else:
                    debug('create cube => f_p')
                    _next_ces = self.fn_ce(_avg_scores_i)
                    _next_ces_flat = _next_ces.flatten()    # (1,vocsize) -> (vocsize,)

                next_krank_ids = part_sort(
                    _next_ces_flat, self.k - cnt_transed)

                row_ksorted_ces = _next_ces_flat[next_krank_ids]

            # add cnt for error The truth value of an array with more than one
            # element is ambiguous
            for i, tup in enumerate(leq_class):
                subcube.append([tup + (_avg_hi, _avg_ai, _avg_si, row_ksorted_ces[j],
                                       wid, i, j, whichsubcub, each_subcube_rowsz) for j, wid in
                                enumerate(next_krank_ids)])

            cube.append(subcube)

        # print created cube before generating current beam for debug ...
        debug('\n************************************************')
        nsubcube = len(cube)
        for subcube_id in xrange(nsubcube):
            subcube = cube[subcube_id]
            nmergings = len(subcube)
            debug('group: {} contains {} mergings:'.format(subcube_id, nmergings))
            for mergeid in xrange(nmergings):
                line_in_subcube = subcube[mergeid]
                score_im1 = line_in_subcube[0][0]
                debug('{: >7} => '.format(format(score_im1, '0.3f')), nl=False)
                for cubetup in line_in_subcube:
                    debug('{: >7}, '.format(
                        format(cubetup[-6], '0.3f')), nl=False)
                debug(' => ', nl=False)
                for cubetup in line_in_subcube:
                    wid = cubetup[-5]
                    debug('{}|{}, '.format(wid, self.tvcb_i2w[wid]), nl=False)
                debug('')
        debug('************************************************\n')

        return cube

    ##################################################################

    # NOTE: (Wen Zhang) Given cube, we calculate true score,
    # computation-expensive here

    ##################################################################

    def Push_heap(self, heap, bidx, citem):
        score_im1, s_im1, y_im1, y_im2, y_im3, bp, \
            _avg_hi, _avg_ai, _avg_si, _avg_ces_i, yi, iexp, jexp, which, rsz = citem
        assert((_avg_hi is None) == (_avg_ai is None))
        hi = self.fn_nh(y_im1, s_im1)
        if self.lm is not None and bidx >= 5:
            ai = _avg_ai if self.ifavg_att else self.fn_na(self.context, hi)[1]
            true_si = self.fn_ns(hi, ai)
            moi = self.fn_mo(y_im1, ai, hi)
            if self.ifscore:
                _score_ith = self.fn_ps(moi, yi)
                true_sci = score_im1 - _score_ith.flatten()[0]
            else:
                sci = self.fn_pws(moi, self.ptv)
                cei = self.fn_ce(sci)
                true_sci = score_im1 + cei.flatten()[yi]
        else:
            if rsz == 1:
                true_si = _avg_si
                true_sci = score_im1 + _avg_ces_i
            else:
                ai = _avg_ai if self.ifavg_att else self.fn_na(self.context, hi)[
                    1]
                true_si = self.fn_ns(hi, ai)
                moi = self.fn_mo(y_im1, ai, true_si)
                if self.ifscore:
                    score_ith = self.fn_ps(moi, yi)
                    true_sci = score_im1 - score_ith.flatten()[0]
                else:
                    sci = self.fn_pws(moi, self.ptv)
                    cei = self.fn_ce(sci)

                    true_sci = score_im1 + cei.flatten()[yi]
                    debug('| {}={}+{}'.format(format(true_sci, '0.3f'),
                                              format(score_im1, '0.3f'),
                                              format(cei.flatten()[yi], '0.3f')))

        heapq.heappush(heap, (true_sci, next(self.cnt), true_si, yi, bp, iexp, jexp, which))

    ##################################################################

    # NOTE: (Wen Zhang) cube pruning

    ##################################################################

    def cube_prune(self, bidx, cube):
        # search in cube
        # cube (matrix(mergings) or vector(no mergings))
        nsubcube = len(cube)
        each_subcube_colsz, each_subcube_rowsz = [], []
        cube_size, counter = 0, 0
        extheap, wavetag, buf_state_merge = [], [], []
        self.lqc[8] += nsubcube   # count of total sub-cubes
        for whichsubcube in xrange(nsubcube):
            subcube = cube[whichsubcube]
            rowsz = len(subcube)
            each_subcube_rowsz.append(rowsz)
            each_subcube_colsz.append(len(subcube[0]))
            # print bidx, rowsz
            self.lqc[9] += rowsz   # count of total lines in sub-cubes
            # initial heap, starting from the left-top corner (best word) of each subcube
            # real score here ... may adding language model here ...
            # we should calculate the real score in current beam when pushing
            # into heap
            self.Push_heap(extheap, bidx, subcube[0][0])
            buf_state_merge.append([])

        cnt_transed = len(self.translations)
        while len(extheap) > 0 and counter < self.k - cnt_transed:
            true_sci, _, true_si, yi, bp, iexp, jexp, which = heapq.heappop(extheap)
            if yi == self.eos_id:
                # beam items count decrease 1
                if self.ifnorm:
                    self.translations.append(
                        ((true_sci / bidx), true_sci, yi, bp, bidx))
                else:
                    self.translations.append(true_sci, yi, bp, bidx)
                debug('add sample {}'.format(self.translations[-1]))
                if len(self.translations) == self.k:
                    # last beam created and finish cube pruning
                    return True
            else:
                # generate one item in current beam
                self.locrt[0] += (bp + 1)
                self.locrt[1] += 1
                self.beam[bidx].append((true_sci, true_si, yi, bp))

            whichsubcub = cube[which]
            # make sure we do not add repeadedly
            if jexp + 1 < each_subcube_colsz[which]:
                right = whichsubcub[iexp][jexp + 1]
                self.Push_heap(extheap, bidx, right)
            if iexp + 1 < each_subcube_rowsz[which]:
                down = whichsubcub[iexp + 1][jexp]
                self.Push_heap(extheap, bidx, down)
            counter += 1
        return False

    def cube_pruning(self):

        for bidx in range(1, self.maxlen + 1):

            eq_classes = OrderedDict()
            self.merge(bidx, eq_classes)

            # create cube and generate next beam from cube
            cube = self.create_cube(bidx, eq_classes)

            if self.cube_prune(bidx, cube):

                debug('Early stop! see {} samples ending with EOS.'.format(self.k))
                avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
                debug('average location of back pointers [{}/{}={}]'.format(
                    self.locrt[0], self.locrt[1], avg_bp))
                sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
                best_sample = sorted_samples[0]
                debug('Translation length(with EOS) [{}]'.format(best_sample[-1]))
                for sample in sorted_samples:  # tuples
                    debug('{}'.format(sample))

                return back_tracking(self.beam, best_sample)

            self.beam[bidx] = sorted(self.beam[bidx], key=lambda tup: tup[0])
            debug('beam {} ----------------------------'.format(bidx))
            for b in self.beam[bidx]:
                debug('{}'.format(b[0:1] + b[2:]))
            # because of the the estimation of P(f|abcd) as P(f|cd), so the generated beam by
            # cube pruning may out of order by loss, so we need to sort it again here
            # losss from low to high

        # no early stop, back tracking
        avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
        debug('Average location of back pointers [{}/{}={}]'.format(
            self.locrt[0], self.locrt[1], avg_bp))
        if len(self.translations) == 0:
            debug('No early stop, no candidates ends with EOS, selecting from '
                 'len {} candidates, may not end with EOS.'.format(maxlen))
            best_sample = (self.beam[maxlen][0][0],) + \
                self.beam[maxlen][0][2:] + (maxlen, )
            debug('Translation length(with EOS) [{}]'.format(best_sample[-1]))
            return back_tracking(self.beam, best_sample)
        else:
            debug('No early stop, not enough {} candidates end with EOS, selecting the best '
                 'sample ending with EOS from {} samples.'.format(self.k, len(self.translations)))
            sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
            best_sample = sorted_samples[0]
            debug('Translation length(with EOS) [{}]'.format(best_sample[-1]))
            for sample in sorted_samples:  # tuples
                debug('{}'.format(sample))
            return back_tracking(self.beam, best_sample)

