import numpy
import copy
import os
import sys
import subprocess
import re
from shutil import copyfile
from utils import *
import collections
from multiprocessing import Process, Queue
import wargs

from search_greedy import Greedy
from search_obs import Obs
from search_nbs import Nbs
#from search_bs_ia import NBS
#from search_bs_layers import NBS
from search_cp import Wcp

from bleu import bleu_file

class Translator(object):

    def __init__(self, model, svcb_i2w=None, tvcb_i2w=None, search_mode=None,
                 thresh=None, lm=None, ngram=None, ptv=None, k=None, noise=False):

        self.model = model
        self.svcb_i2w = svcb_i2w
        self.tvcb_i2w = tvcb_i2w
        self.search_mode = search_mode if search_mode else wargs.search_mode
        self.thresh = thresh
        self.lm = lm
        self.ngram = ngram
        self.ptv = ptv
        self.k = k if k else wargs.beam_size
        self.noise = noise

    def trans_onesent(self, s):

        if self.search_mode == 0:
            self.greedy = Greedy(self.tvcb_i2w)
            trans = self.greedy.greedy_trans(s)

        elif self.search_mode == 1:
            self.ori = Obs(self.tvcb_i2w, k=self.k)
            trans = self.ori.original_trans(s)

        elif self.search_mode == 2:
            self.nbs = Nbs(self.model, self.tvcb_i2w, k=self.k, noise=self.noise)
            trans, ids = self.nbs.beam_search_trans(s)

        elif self.search_mode == 3:
            self.wcp = Wcp(self.model, self.tvcb_i2w, k=self.k, thresh=self.thresh)
            trans = self.wcp.cube_prune_trans(s)

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

            wlog('[{:3}] {}'.format('Out', trans))

    def single_trans_file(self, src_input_data):

        batch_count = len(src_input_data)   # batch size 1 for valid
        total_trans = []
        sent_no = 0
        for bid in range(batch_count):
            batch_srcs_LB = src_input_data[bid][1]
            #batch_srcs_LB = batch_srcs_LB.squeeze()
            for no in range(batch_srcs_LB.size(1)):
                s_filter = sent_filter(list(batch_srcs_LB[:,no].data))
                trans, ids = self.trans_onesent(s_filter)
                total_trans.append(trans)
                if numpy.mod(sent_no + 1, 100) == 0:
                    wlog('Sample {} Done'.format(sent_no + 1))
            sent_no += 1

        wlog('Done ...')
        return '\n'.join(total_trans)

    def translate(self, queue, rqueue, pid):

        while True:
            req = queue.get()
            if req == None:
                break

            idx, src = req[0], req[1]
            wlog('{}-{}'.format(pid, idx))
            s_filter = filter(lambda x: x != 0, src)
            trans, _ = self.trans_onesent(s_filter)

            rqueue.put((idx, trans))

        return

    def multi_process(self, x_iter, n_process=5):
        queue = Queue()
        rqueue = Queue()
        processes = [None] * n_process
        for pidx in xrange(n_process):
            processes[pidx] = Process(target=self.translate, args=(queue, rqueue, pidx))
            processes[pidx].start()

        def _send_jobs(x_iter):
            for idx, line in enumerate(x_iter):
                # log(idx, line)
                queue.put((idx, line))
            return idx + 1

        def _finish_processes():
            for pidx in xrange(n_process):
                queue.put(None)

        def _retrieve_jobs(n_samples):
            trans = [None] * n_samples
            for idx in xrange(n_samples):
                resp = rqueue.get()
                trans[resp[0]] = resp[1]
                if numpy.mod(idx + 1, 1) == 0:
                    wlog('Sample {}/{} Done'.format((idx + 1), n_samples))
            return trans

        wlog('Translating ...')
        n_samples = _send_jobs(x_iter)     # sentence number in source file
        trans_res = _retrieve_jobs(n_samples)
        _finish_processes()
        wlog('Done ...')

        return '\n'.join(trans_res)

    def write_file_eval(self, out_fname, trans, data_prefix):

        fout = open(out_fname, 'w')    # valids/trans
        fout.writelines(trans)
        fout.close()

        ref_fpaths = []
        for ref_cnt in range(4):
            ref_fpath = '{}{}{}{}'.format(wargs.val_tst_dir,
                                          data_prefix, '.ref.plain.low', ref_cnt)
            #ref_fpath = '{}{}{}'.format(refs_path, 'nist03.ref', ref_cnt)
            if not os.path.exists(ref_fpath): continue
            ref_fpaths.append(ref_fpath)

        mteval_bleu = bleu_file(out_fname, ref_fpaths)
        os.rename(out_fname, "{}_{}.txt".format(out_fname, mteval_bleu))

        return mteval_bleu

    def trans_tests(self, tests_data, eid, bid):

        for _, test_prefix in zip(tests_data, wargs.tests_prefix):

            wlog('Translating {}'.format(test_prefix))
            trans = self.single_trans_file(tests_data[test_prefix])

            outprefix = wargs.dir_tests + '/' + test_prefix + '/trans'
            test_out = "{}_e{}_upd{}_b{}m{}_bch{}.txt".format(
                outprefix, eid, bid, self.k, self.search_mode, wargs.with_batch)

            _ = self.write_file_eval(test_out, trans, test_prefix)

    def trans_eval(self, valid_data, eid, bid, model_file, tests_data):

        wlog('Translating {}'.format(wargs.val_prefix))
        trans = self.single_trans_file(valid_data)
        #trans = translator.multi_process(viter, n_process=nprocess)

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
            if wargs.final_test is False: self.trans_tests(tests_data, eid, bid)
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
