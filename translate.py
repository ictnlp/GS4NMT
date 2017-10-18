from __future__ import division
from __future__ import absolute_import

import numpy
import copy
import os
import sys
import re
import time
import subprocess
from shutil import copyfile
from utils import *
import collections
from multiprocessing import Process, Queue
import wargs

if wargs.model == 2: from searchs.nbs_ia import *
elif wargs.model == 3: from searchs.nbs_layers import *
elif wargs.model == 5: from searchs.nbs_sru import *
else: from searchs.nbs import *

from bleu import bleu_file

class Translator(object):

    def __init__(self, model, svcb_i2w=None, tvcb_i2w=None, search_mode=None,
                 thresh=None, lm=None, ngram=None, ptv=None, k=None, noise=False):

        self.svcb_i2w = svcb_i2w
        self.tvcb_i2w = tvcb_i2w
        self.search_mode = search_mode if search_mode else wargs.search_mode
        self.thresh = thresh
        self.lm = lm
        self.ngram = ngram
        self.ptv = ptv
        self.k = k if k else wargs.beam_size
        self.noise = noise

        if self.search_mode == 0: self.greedy = Greedy(self.tvcb_i1w)
        elif self.search_mode == 1: self.nbs = Nbs(model, self.tvcb_i2w, k=self.k,
                                                   noise=self.noise)
        elif self.search_mode == 2: self.wcp = Wcp(model, self.tvcb_i2w, k=self.k)

    def trans_onesent(self, s):

        trans_start = time.time()

        if self.search_mode == 0: trans = self.greedy.greedy_trans(s)
        elif self.search_mode == 1: (trans, ids), loss = self.nbs.beam_search_trans(s)
        elif self.search_mode == 2: (trans, ids), loss = self.wcp.cube_prune_trans(s)

        #spend = time.time() - trans_start
        #wlog('Word-Level spend: {} / {} = {}'.format(
        #    format_time(spend), len(ids), format_time(spend / len(ids))))

        return trans, ids

    def trans_samples(self, srcs, trgs):

        if isinstance(srcs, tc.autograd.variable.Variable): srcs = srcs.data
        if isinstance(trgs, tc.autograd.variable.Variable): trgs = trgs.data

        # srcs: (sample_size, max_sLen)
        for idx in range(len(srcs)):

            s_filter = sent_filter(list(srcs[idx]))
            wlog('\n[{:3}] {}'.format('Src', idx2sent(s_filter, self.svcb_i2w)))
            t_filter = sent_filter(list(trgs[idx]))
            wlog('[{:3}] {}'.format('Ref', idx2sent(t_filter, self.tvcb_i2w)))

            trans, _ = self.trans_onesent(s_filter)

            if wargs.with_bpe is True:
                wlog('[{:3}] {}'.format('Bpe', trans))
                #trans = trans.replace('@@ ', '')
                trans = re.sub('(@@ )|(@@ ?$)', '', trans)

            wlog('[{:3}] {}'.format('Out', trans))

    def single_trans_file(self, src_input_data, src_labels_fname=None):

        batch_count = len(src_input_data)   # batch size 1 for valid
        point_every, number_every = int(math.ceil(batch_count/100)), int(math.ceil(batch_count/10))
        total_trans = []
        sent_no, words_cnt = 0, 0

        trans_start = time.time()
        for bid in range(batch_count):
            batch_srcs_LB = src_input_data[bid][1] # (dxs, tsrcs, lengths, src_mask)
            #batch_srcs_LB = batch_srcs_LB.squeeze()
            for no in range(batch_srcs_LB.size(1)): # batch size, 1 for valid
                s_filter = sent_filter(list(batch_srcs_LB[:,no].data))

                if src_labels_fname:
                    # split by segment labels file
                    segs = self.segment_src(s_filter, labels[bid].strip().split(' '))
                    trans = []
                    for seg in segs:
                        seg_trans, ids = self.trans_onesent(seg)
                        words_cnt += len(ids)
                        trans.append(seg_trans)
                    # merge by order
                    trans = ' '.join(trans)
                else:
                    trans, ids = self.trans_onesent(s_filter)
                    words_cnt += len(ids)

                total_trans.append(trans)
                if numpy.mod(sent_no + 1, point_every) == 0: wlog('.', False)
                if numpy.mod(sent_no + 1, number_every) == 0: wlog('{}'.format(sent_no + 1), False)

                sent_no += 1
        wlog('')

        if self.search_mode == 1:
            C = self.nbs.C
            wlog('Average location of bp [{}/{}={:6.4f}]'.format(C[1], C[0], C[1] / C[0]))
            wlog('Step[{}] stepout[{}]'.format(*C[2:]))

        if self.search_mode == 2:
            C = self.wcp.C
            wlog('Average Merging Rate [{}/{}={:6.4f}]'.format(C[1], C[0], C[1] / C[0]))
            wlog('Average location of bp [{}/{}={:6.4f}]'.format(C[3], C[2], C[3] / C[2]))
            wlog('Step[{}] stepout[{}]'.format(*C[4:]))

        spend = time.time() - trans_start
        wlog('Word-Level spend: {} / {} = {}'.format(
            format_time(spend), words_cnt, format_time(spend / words_cnt)))

        wlog('Done ...')
        return '\n'.join(total_trans) + '\n'

    def segment_src(self, src_list, labels_list):

        #print len(src_list), len(labels_list)
        assert len(src_list) == len(labels_list)
        segments, seg = [], []
        for i in range(len(src_list)):
            c, l = src_list[i], labels_list[i]
            if l == 'S':
                segments.append([c])
            elif l == 'E':
                seg.append(c)
                segments.append(seg)
                seg = []
            elif l == 'B':
                if len(seg) > 0: segments.append(seg)
                seg = []
                seg.append(c)
            else:
                seg.append(c)

        return segments

    def write_file_eval(self, out_fname, trans, data_prefix):

        fout = open(out_fname, 'w')    # valids/trans
        fout.writelines(trans)
        fout.close()

        if wargs.with_bpe is True:
            os.system('cp {} {}.bpe'.format(out_fname, out_fname))
            wlog('cp {} {}.bpe'.format(out_fname, out_fname))
            os.system("sed -r 's/(@@ )|(@@ ?$)//g' {}.bpe > {}".format(out_fname, out_fname))
            wlog("sed -r 's/(@@ )|(@@ ?$)//g' {}.bpe > {}".format(out_fname, out_fname))

        '''
        os.system('cp {} {}.bpe'.format(out_fname, out_fname))
	wlog('cp {} {}.bpe'.format(out_fname, out_fname))
        os.system("sed -r 's/(@@ )|(@@ ?$)//g' {}.bpe > {}".format(out_fname, out_fname))
	wlog("sed -r 's/(@@ )|(@@ ?$)//g' {}.bpe > {}".format(out_fname, out_fname))
        src_sgm = '{}/{}.src.sgm'.format(wargs.val_tst_dir, data_prefix)
        os.system('./scripts/wrap_xml.pl zh {} {} < {} > {}.sgm'.format(src_sgm, data_prefix, out_fname, out_fname))
	wlog('./scripts/wrap_xml.pl zh {} {} < {} > {}.sgm'.format(src_sgm, data_prefix, out_fname, out_fname))
        os.system('./scripts/chi_char_segment.pl -t xml < {}.sgm > {}.seg.sgm'.format(out_fname, out_fname))
	wlog('./scripts/chi_char_segment.pl -t xml < {}.sgm > {}.seg.sgm'.format(out_fname, out_fname))
        os.system('./scripts/de-xml.pl < {}.seg.sgm > {}.seg.plain'.format(out_fname, out_fname))
	wlog('./scripts/de-xml.pl < {}.seg.sgm > {}.seg.plain'.format(out_fname, out_fname))
        '''

        ref_fpaths = []
        # *.ref
        ref_fpath = '{}{}.{}'.format(wargs.val_tst_dir, data_prefix, wargs.val_ref_suffix)
        if os.path.exists(ref_fpath): ref_fpaths.append(ref_fpath)
        for idx in range(wargs.ref_cnt):
            # *.ref0, *.ref1, ...
            ref_fpath = '{}{}.{}{}'.format(wargs.val_tst_dir, data_prefix, wargs.val_ref_suffix, idx)
            if not os.path.exists(ref_fpath): continue
            ref_fpaths.append(ref_fpath)

        mteval_bleu = bleu_file(out_fname, ref_fpaths)
        #mteval_bleu = bleu_file(out_fname + '.seg.plain', ref_fpaths)
        os.rename(out_fname, "{}_{}.txt".format(out_fname, mteval_bleu))

        return mteval_bleu

    def ai_write_file_eval(self, out_fname, trans, data_prefix):

        fout = open(out_fname, 'w')    # valids/trans
        fout.writelines(trans)
        fout.close()

        src_sgm = '{}/{}.src.sgm'.format(wargs.val_tst_dir, data_prefix)
        os.system('./scripts/wrap_xml.pl zh {} {} < {} > {}.sgm'.format(src_sgm, data_prefix, out_fname, out_fname))
        os.system('./scripts/chi_char_segment.pl -t xml < {}.sgm > {}.seg.sgm'.format(out_fname, out_fname))

        #de_xml_cmd = './scripts/de-xml.pl < {}.seg.xml > {}.seg.plain'.format(out_fname, out_fname)
        #os.system(de_xml_cmd)

        # calc BLEU/NIST score
        bleu_cmd = './scripts/mteval-v11b.pl -s {} -r {}/{}.ref.seg.sgm -t {}.seg.sgm -c > {}.bleu'.format(
            src_sgm, wargs.val_tst_dir, data_prefix, out_fname, out_fname)
        os.system(bleu_cmd)

        # get bleu: NIST score = 5.5073  BLEU score = 0.2902 for system "DemoSystem"
        try:
            raw_bleu = ' '.join(open(out_fname + '.bleu', 'r').readlines()).replace('\n', ' ')
            gps = re.search( r'NIST score = (?P<NIST>[\d\.]+)  BLEU score = (?P<BLEU>[\d\.]+) ' , raw_bleu )
            if gps:
                bleu_score = gps.group('BLEU')
            else:
                print >> sys.stderr, "ERROR: unable to get bleu and nist score"
                sys.exit(1)
        except:
            print >> sys.stderr, "ERROR: exception during calculating bleu score"
        os.rename(out_fname, "{}_{}.txt".format(out_fname, bleu_score))

        return bleu_score

    def trans_tests(self, tests_data, eid, bid):

        for _, test_prefix in zip(tests_data, wargs.tests_prefix):

            wlog('Translating testing dataset {}'.format(test_prefix))
            label_fname = '{}{}/{}.label'.format(wargs.val_tst_dir, wargs.seg_val_tst_dir,
                                                 test_prefix) if wargs.segments else None
            trans = self.single_trans_file(tests_data[test_prefix], label_fname)

            outprefix = wargs.dir_tests + '/' + test_prefix + '/trans'
            test_out = "{}_e{}_upd{}_b{}m{}_bch{}".format(
                outprefix, eid, bid, self.k, self.search_mode, wargs.with_batch)

            _ = self.write_file_eval(test_out, trans, test_prefix)

    def trans_eval(self, valid_data, eid, bid, model_file, tests_data):

        wlog('Translating validation dataset {}{}.{}'.format(wargs.val_tst_dir, wargs.val_prefix, wargs.val_src_suffix))
        label_fname = '{}{}/{}.label'.format(wargs.val_tst_dir, wargs.seg_val_tst_dir,
                                             wargs.val_prefix) if wargs.segments else None
        trans = self.single_trans_file(valid_data, label_fname)

        outprefix = wargs.dir_valid + '/trans'
        valid_out = "{}_e{}_upd{}_b{}m{}_bch{}".format(
            outprefix, eid, bid, self.k, self.search_mode, wargs.with_batch)

        mteval_bleu = self.write_file_eval(valid_out, trans, wargs.val_prefix)

        bleu_scores_fname = '{}/train_bleu.log'.format(wargs.dir_valid)
        bleu_scores = [0.]
        if os.path.exists(bleu_scores_fname):
            with open(bleu_scores_fname) as f:
                for line in f:
                    s_bleu = line.split(':')[-1].strip()
                    bleu_scores.append(float(s_bleu))

        wlog('current [{}] - best history [{}]'.format(mteval_bleu, max(bleu_scores)))
        if mteval_bleu > max(bleu_scores):   # better than history
            copyfile(model_file, wargs.best_model)
            wlog('cp {} {}'.format(model_file, wargs.best_model))
            bleu_content = 'epoch [{}], batch[{}], BLEU score*: {}'.format(eid, bid, mteval_bleu)
            if wargs.final_test is False and tests_data is not None: self.trans_tests(tests_data, eid, bid)
        else:
            bleu_content = 'epoch [{}], batch[{}], BLEU score : {}'.format(eid, bid, mteval_bleu)

        append_file(bleu_scores_fname, bleu_content)

        sfig = '{}.{}'.format(outprefix, 'sfig')
        sfig_content = ('{} {} {} {} {}').format(eid, bid, self.search_mode, self.k, mteval_bleu)
        append_file(sfig, sfig_content)

        if wargs.save_one_model:
            os.remove(model_file)
            wlog('delete {}'.format(model_file))

        return mteval_bleu


if __name__ == "__main__":
    import sys
    res = valid_bleu(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    wlog(res)
