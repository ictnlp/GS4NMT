from __future__ import division

import time
import sys
import numpy

from utils import *
from utils import _filter_reidx
from collections import OrderedDict
import heapq
from itertools import count
import copy
import const

class Wcp(object):

    def __init__(self, model, tvcb_i2w=None, k=10, thresh=100.0, lm=None, ngram=3, ptv=None):

        self.model = model
        self.decoder = model.decoder

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
        s_init, self.enc_src0, self.uh0 = self.model.init(s_tensor, test=True)
        # s_init: (1, trg_nhids), enc_src0: (srcL, 1, src_nhids*2), uh0: (srcL, 1, align_size)

        init_beam(self.beam, cnt=self.maxlen, s0=s_init)

        best_trans, best_loss = self.cube_pruning()

        wlog('@source[{}], translation(without eos)[{}], maxlen[{}], loss[{}]'.format(
            len(s_list), len(best_trans), self.maxlen, best_loss))
        wlog('init[{}] nh[{}] na[{}] ns[{}] mo[{}] ws[{}] ps[{}] ce[{}]'.format(*self.lqc[0:8]))

        avg_merges = format(self.lqc[9] / self.lqc[8], '0.3f')
        wlog('average merge count[{}/{}={}]'.format(self.lqc[9], self.lqc[8], avg_merges))

        return _filter_reidx(best_trans, self.tvcb_i2w)

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
        for j in range(len_prevb):  # index of each item in last beam

            if j in used: continue

            tmp = []
            if _memory[j]:
                _needed = _memory[j]
                score_im1_1, s_im1_1, y_im1_1, nj = _needed
                assert(j == nj)
            else:
                score_im1_1, s_im1_1, y_im1_1, _ = prevb[j]
                _needed = _memory[j] = (score_im1_1, s_im1_1, y_im1_1, j)

            tmp.append(_needed)

            for jj in range(j + 1, len_prevb):

                if _memory[jj]:
                    _needed = _memory[jj]
                    score_im1_2, s_im1_2, y_im1_2, njj = _needed
                    assert(jj == njj)
                else:
                    score_im1_2, s_im1_2, y_im1_2, _ = prevb[jj]
                    _needed = _memory[jj] = (score_im1_2, s_im1_2, y_im1_2, jj)

                ifmerge = False
                if wargs.merge_way == 'Y': ifmerge = (y_im1_2 == y_im1_1)
                if ifmerge:
                    tmp.append(_needed)
                    used.append(jj)

            eq_classes[key] = tmp
            key += 1

    ##################################################################

    # NOTE: (Wen Zhang) create cube by sort row dimension

    ##################################################################

    #@exeTime
    def create_cube(self, bidx, eq_classes):
        # eq_classes: 0: [(score_im1, s_im1, y_im1, bp), ... ], 1:
        cube = []
        cnt_transed = len(self.translations)

        # for each equivalence class
        for whichsubcub, leq_class in eq_classes.iteritems():

            each_subcube_rowsz = len(leq_class)
            score_im1_r0, _s_im1, y_im1, _ = leq_class[0]
            subcube = []
            _si, _ai, _next = None, None, None

            # TODO sort the row dimension by average scores
            if not each_subcube_rowsz == 1:
                _s_im1 = [tup[1] for tup in leq_class]
                _s_im1 = tc.mean(tc.stack(_s_im1, dim=0), dim=0)

            _ai, _si, y_im1 = self.decoder.step(_s_im1, self.enc_src0, self.uh0, y_im1)
            _logit = self.decoder.step_out(_si, y_im1, _ai)

            if wargs.vocab_norm:
                _next = self.model.classifier(_logit)
                _next = _next.cpu().data.numpy()
                debug('create cube => f_p')
            else:
                _next = -self.model.classifier.get_a(logit)
            _next = _next.flatten()    # (1,vocsize) -> (vocsize,)

            next_krank_ids = part_sort(_next, self.k - cnt_transed)

            row_ksorted_ces = _next[next_krank_ids]

            # add cnt for error The truth value of an array with more than one
            # element is ambiguous
            for i, tup in enumerate(leq_class):
                subcube.append([tup + (_ai, _si, row_ksorted_ces[j],
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

        score_im1, s_im1, y_im1, bp, _ai, _si, _next, yi, iexp, jexp, which, rsz = citem

        if rsz == 1:
            true_si = _si
            true_sci = score_im1 + _next
        else:
            a_i, true_si, y_im1 = self.decoder.step(s_im1, self.enc_src0, self.uh0, y_im1)
            logit = self.decoder.step_out(true_si, y_im1, a_i)

            if wargs.vocab_norm:
                _next = self.model.classifier(logit)
                _next = _next.cpu().data.numpy()
                debug('create cube => f_p')
            else:
                _next = -self.model.classifier.get_a(logit)
                _next = _next.cpu().data.numpy()
            _next = _next.flatten()    # (1,vocsize) -> (vocsize,)

            true_sci = score_im1 + _next[yi]
            debug('| {}={}+{}'.format(format(true_sci, '0.3f'),
                                      format(score_im1, '0.3f'),
                                      format(_next[yi], '0.3f')))

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
            if yi == const.EOS:
                # beam items count decrease 1
                if wargs.len_norm:
                    self.translations.append(((true_sci / bidx), true_sci, yi, bp, bidx))
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

