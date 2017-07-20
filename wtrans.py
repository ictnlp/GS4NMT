# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import subprocess
import torch as tc

import const
import wargs
from utils import *
from torch import cuda
from inputs import Input
from model_rnnsearch import *
from cp_sample import Translator
from handle_data import extract_vocab, val_wrap_data

if __name__ == "__main__":

    A = argparse.ArgumentParser(prog='NMT decoder')

    A.add_argument('--model-file', dest='model_file', help='model file')

    A.add_argument('--test-file', dest='test_file', default=None,
                   help='the input test file path we will translate')

    '''
    A.add_argument('--search-mode', dest='search_mode', default=2,
                   help='0: Greedy, 1&2: naive beam search, 3: cube pruning')

    A.add_argument('--beam-size', dest='beam_size', default=wargs.beam_size, help='beamsize')

    A.add_argument('--use-valid', dest='use_valid', type=int, default=0,
                   help='Translate valid set. (DEFAULT=0)')

    A.add_argument('--use-batch', dest='use_batch', type=int, default=0,
                   help='Whether we apply batch on beam search. (DEFAULT=0)')

    A.add_argument('--vocab-norm', dest='vocab_norm', type=int, default=1,
                   help='Whether we normalize the distribution of vocabulary (DEFAULT=1)')

    A.add_argument('--len-norm', dest='len_norm', type=int, default=1,
                   help='During searching, whether we normalize accumulated loss by length.')

    A.add_argument('--use-mv', dest='use_mv', type=int, default=0,
                   help='We use manipulation vacabulary by add this parameter. (DEFAULT=0)')

    A.add_argument('--merge-way', dest='merge_way', default='Him1',
                   help='merge way in cube pruning. (DEFAULT=s_im1. Him1/Hi/AiKL/LM)')

    A.add_argument('--avg-att', dest='avg_att', type=int, default=0,
                   help='Whether we average attention vector. (DEFAULT=0)')

    A.add_argument('--m-threshold', dest='m_threshold', type=float, default=0.,
                   help='a super-parameter to merge in cube pruning. (DEFAULT=0. no merge)')
    '''

    args = A.parse_args()

    model_file = args.model_file
    test_file = args.test_file if args.test_file else None

    '''
    search_mode = args.search_mode
    beam_size = args.beam_size

    useValid = args.use_valid
    useBatch = args.use_batch
    vocabNorm = args.vocab_norm
    lenNorm = args.len_norm
    useMv = args.use_mv
    mergeWay = args.merge_way
    avgAtt = args.avg_att

    m_threshold = args.m_threshold

    switchs = [useBatch, vocabNorm, lenNorm, useMv, mergeWay, avgAtt]
    '''

    src_vocab = extract_vocab(None, wargs.src_dict)
    trg_vocab = extract_vocab(None, wargs.trg_dict)

    src_vocab_size, trg_vocab_size= src_vocab.size(), trg_vocab.size()

    wlog('Vocabulary size: |source|={}, |target|={}'.format(src_vocab_size, trg_vocab_size))

    wlog('Start decoding ... init model ... ', 0)

    nmtModel = NMT()
    classifier = Classifier(wargs.out_size, trg_vocab_size)

    if wargs.dec_gpu_id:
        cuda.set_device(wargs.dec_gpu_id[0])
        nmtModel.cuda()
        classifier.cuda()
        wlog('push model onto GPU[{}] ... '.format(wargs.dec_gpu_id[0]))
    else:
        nmtModel.cpu()
        classifier.cpu()
        wlog('push model onto CPU ... ')

    model_dict, class_dict, eid, bid, _ = load_pytorch_model(model_file)

    nmtModel.load_state_dict(model_dict)
    classifier.load_state_dict(class_dict)
    nmtModel.classifier = classifier

    wlog('\nFinish to load model.')

    dec_conf()
    translator = Translator(nmtModel, src_vocab.idx2key, trg_vocab.idx2key, wargs.search_mode)

    if not test_file:
        wlog('Translating one sentence ... ')
        # s = tc.Tensor([[0, 10811, 140, 217, 19, 1047, 482, 29999, 0, 0, 0]])
        s = '( 北京 综合 电 ) 曾 因 美国 总统 布什 访华 而 一 度 升温 的 中 美 关系 在 急速 ' \
                '冷却 , 中国 昨天 证实 取消 今年 海军 舰艇 编队 访问 美国 港口 的 计划 , 并 ' \
                '拒绝 确定 国家 副主席 胡锦涛 是否 会 按 原定 计划 访美 。'
        #s = "章启月 昨天 也 证实 了 俄罗斯 媒体 的 报道 , 说 中国 国家 主席 江泽民 前晚 应 " \
        #        "俄罗斯 总统 普京 的 要求 与 他 通 了 电话 , 双方 主要 是 就中 俄 互利 合作 " \
        #        "问题 交换 了 意见 。"
        t = "( beijing , syndicated news ) the sino - us relation that was heated momentarily " \
                "by the us president bush 's visit to china is cooling down rapidly . china " \
                "confirmed yesterday that it has called off its naval fleet visit to the us " \
                "ports this year and refused to confirm whether the country 's vice president " \
                "hu jintao will visit the united states as planned ."
        s = [src_vocab.key2idx[x] if x in src_vocab.key2idx else const.UNK for x in s.split(' ')]
        #wlog(s)
        s = tc.Tensor([s])
        t = [trg_vocab.key2idx[x] if x in trg_vocab.key2idx else const.UNK for x in t.split(' ')]
        #wlog(t)
        t = tc.Tensor([t])
        pv = tc.Tensor([0, 10782, 2102, 1735, 4, 1829, 1657, 29999])
        translator.trans_samples(s, t)
        sys.exit(0)

    test_file = wargs.val_tst_dir + wargs.val_prefix + '.src'
    wlog('Translating test file {} ... '.format(test_file))
    test_src_tlst, test_src_lens = val_wrap_data(test_file, src_vocab)
    test_input_data = Input(test_src_tlst, None, 1, volatile=True)

    trans = translator.single_trans_file(test_input_data)
    #trans = translator.multi_process(viter, n_process=nprocess)

    outdir = wargs.file_tran_dir
    init_dir(outdir)
    outprefix = outdir + '/trans'
    # wTrans/trans
    file_out = "{}_e{}_upd{}_b{}m{}_bch{}".format(
        outprefix, eid, bid, wargs.beam_size, wargs.search_mode, wargs.with_batch)

    mteval_bleu = translator.write_file_eval(file_out, trans, wargs.val_prefix)

    bleus_record_fname = '{}/record_bleu.log'.format(outdir)
    bleu_content = 'epoch [{}], batch[{}], BLEU score : {}'.format(eid, bid, mteval_bleu)
    with open(bleus_record_fname, 'a') as f:
        f.write(bleu_content + '\n')
        f.close()

    sfig = '{}.{}'.format(outprefix, 'sfig')
    sfig_content = ('{} {} {} {} {}').format(
        #alpha,
        #beta,
        eid,
        bid,
        wargs.search_mode,
        wargs.beam_size,
        #kl,
        mteval_bleu
    )
    append_file(sfig, sfig_content)

