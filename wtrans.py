# -*- coding: utf-8 -*-

import os
from cp_sample import Translator
# Get the arguments
import sys
import subprocess
import numpy as np
import time
import argparse
import wargs
from utils import *

if __name__ == "__main__":

    decoder = argparse.ArgumentParser(prog='NMT decoder')

    decoder.add_argument(
        '--model-file',
        dest='model_file',
        help='Model file.',
    )

    decoder.add_argument(
        '--vocab-data',
        dest='vocab_data',
        help='Vocabulary file.',
    )

    decoder.add_argument(
        '--valid-data',
        dest='valid_data',
        default=None,
        help='Translate valid set. (DEFAULT=None)',
    )

    args = decoder.parse_args()
    model_file = args.model_file
    vocab_data = args.vocab_data
    valid_data = args.valid_data if args.valid_data else None

    debug('\tSource vocab count: {}, target vocab count: {}'.format(
        len(vocab_data['src'].idx2key), len(vocab_data['trg'].idx2key)))

    debug('Start decoding ...')

    nmtModel = NMT()
    classifier = Logistic(args.out_size, vocab_data['trg'].size())

    if args.gpu_id:
        wlog('push model onto GPU')
        nmtModel.cuda()
        classifier.cuda()
    else:
        wlog('push model onto CPU')
        nmtModel.cpu()
        classifier.cpu()

    model_obj = tc.load(model_file)
    epoch = model_obj['epoch']
    batch = model_obj['bidx']
    nmtModel.load_state_dict(model_obj['model'])
    classifier.load_state_dict(model_obj['class'])
    nmtModel.classifier = classifier

    translator = Translator(
        nmtModel,
        svcb_i2w=vocab_data['src'].idx2key,
        tvcb_i2w=vocab_data['trg'].idx2key,
    )

    if not valid_data:
        # s = np.asarray([[0, 10811, 140, 217, 19, 1047, 482, 29999, 0, 0, 0]])
        # 章启月 昨天 也 证实 了 俄罗斯 媒体 的 报道 , 说 中国 国家 主席 江泽民 前晚 应 俄罗斯 总统
        # 普京 的 要求 与 他 通 了 电话 , 双方 主要 是 就中 俄 互利 合作 问题 交换 了 意见 。
        s = np.asarray([[3490, 1477, 41, 1711, 10, 422, 722, 3, 433, 2, 28, 11, 39, 161, 240, 1,
                         219, 422, 217, 1512, 3, 120, 19, 32, 3958, 10, 630, 2, 158, 147, 8,
                         11963, 651, 1185, 51, 36, 882, 10, 267, 4, 29999]])
        '''
        s = np.asarray([[334, 1212, 2, 126, 3, 1, 27, 1, 11841, 2358, 5313, 2621, 10312, 2564,
                         100, 316, 21219, 2, 289, 18, 680, 11, 3161, 3, 316, 21219, 2, 41, 18,
                         365, 680, 316, 7, 772, 3, 60, 2, 147, 1275, 316, 1, 6737, 17, 11608, 50,
                         5284, 2, 279, 84, 8635, 1, 2, 569, 3246, 680, 388, 342, 2, 84, 285,
                         4897, 41, 4144, 11996, 4, 29999]])
        '''
        # s = np.asarray([[3490]])
        t = np.asarray([[0, 10782, 2102, 1735, 4, 1829, 1657, 29999, 0]])
        pv = np.asarray([0, 10782, 2102, 1735, 4, 1829, 1657, 29999])
        translator.trans_samples(s, t)
        sys.exit(0)

    trans = translator.single_trans_valid(valid_data)
    #trans = translator.multi_process(viter, n_process=nprocess)

    outdir = wargs.wvalids
    init_dir(outdir)
    outprefix = outdir + '/trans'

    valid_out = "{}_e{}_upd{}_b{}m{}_bch{}".format(
        outprefix, epoch, batch, wargs.beam_size, wargs.search_mode, wargs.with_batch)
    fVal_save = open(valid_out, 'w')    # valids/trans
    fVal_save.writelines(trans)
    fVal_save.close()

    mteval_bleu, multi_bleu = valid_bleu(valid_out, config['val_tst_dir'], config['val_prefix'])
    mteval_bleu = float(mteval_bleu)

    bleu_scores_fname = '{}/train_bleu.log'.format(outdir)
    bleu_scores = [0.]
    if os.path.exists(bleu_scores_fname):
        with open(bleu_scores_fname) as f:
            lines = f.readlines()
            f.close()
        for line in lines:
            s_bleu = line.split(':')[-1].strip()
            bleu_scores.append(float(s_bleu))

    if mteval_bleu > max(bleu_scores):   # better than history
        # print 'cp {} {}/params.best.npz'.format(model_name, outdir)
        child = subprocess.Popen(
            'cp {} {}/params.best.npz'.format(model_name, outdir), shell=True)
        bleu_content = 'epoch [{}], batch[{}], BLEU score*: {}'.format(epoch, batch, mteval_bleu)
        #os.remove(model_name)
    else:
        bleu_content = 'epoch [{}], batch[{}], BLEU score : {}'.format(epoch, batch, mteval_bleu)

    with open(bleu_scores_fname, 'a') as f:
        f.write(bleu_content + '\n')
        f.close()

    # ori_mteval_bleu, ori_multi_bleu = fetch_bleu_from_file(oriref_bleu_log)
    sfig = '{}.{}'.format(outprefix, 'sfig')
    # sfig_content = str(eidx) + ' ' + str(uidx) + ' ' + str(mteval_bleu) + ' ' + \
    #    str(multi_bleu) + ' ' + str(ori_mteval_bleu) + ' ' + str(ori_multi_bleu)
    sfig_content = ('{} {} {} {} {} {} {} {} {}').format(
        alpha,
        beta,
        epoch,
        batch,
        search_mode,
        beam_size,
        kl,
        mteval_bleu,
        multi_bleu
    )
    append_file(sfig, sfig_content)

    os.rename(valid_out, "{}_{}_{}.txt".format(valid_out, mteval_bleu, multi_bleu))
