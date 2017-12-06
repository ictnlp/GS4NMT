#( -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import time
import argparse
import subprocess
import torch as tc
from torch import cuda

import wargs
from tools.inputs import Input
from tools.utils import *

from translate import Translator
from inputs_handler import extract_vocab, val_wrap_data, wrap_data
from models.losser import *

if __name__ == "__main__":

    A = argparse.ArgumentParser(prog='NMT translator ... ')

    #A.add_argument('--model', dest='model', default=0,
    #               help='0: groundhog, 1: rnnsearch, 2: ia, 3: ran, 4: rn, 5: sru, 6: cyknet')

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

    #model = args.model
    model_file = args.model_file

    wlog('Using model: {}'.format(wargs.model))

    if wargs.model == 0: from models.groundhog import *
    elif wargs.model == 1: from models.rnnsearch import *
    elif wargs.model == 2: from models.rnnsearch_ia import *
    elif wargs.model == 3: from models.ran_agru import *
    elif wargs.model == 4: from models.rnnsearch_rn import *
    elif wargs.model == 5: from models.nmt_sru import *
    elif wargs.model == 6: from models.nmt_cyk import *

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

    src_vocab_size, trg_vocab_size = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(src_vocab_size, trg_vocab_size))

    wlog('Start decoding ... init model ... ', 0)

    nmtModel = NMT(src_vocab_size, trg_vocab_size)
    classifier = Classifier(wargs.out_size, trg_vocab_size,
                            nmtModel.decoder.trg_lookup_table if wargs.copy_trg_emb is True else None)

    if wargs.gpu_id:
        cuda.set_device(wargs.gpu_id[0])
        nmtModel.cuda()
        classifier.cuda()
        wlog('Push model onto GPU[{}] ... '.format(wargs.gpu_id[0]))
    else:
        nmtModel.cpu()
        classifier.cpu()
        wlog('Push model onto CPU ... ')

    model_dict, class_dict, eid, bid, _ = load_pytorch_model(model_file)

    nmtModel.load_state_dict(model_dict)
    classifier.load_state_dict(class_dict)
    nmtModel.classifier = classifier

    wlog('\nFinish to load model.')

    dec_conf()

    nmtModel.eval()
    nmtModel.classifier.eval()
    tor = Translator(nmtModel, src_vocab.idx2key, trg_vocab.idx2key, print_att=wargs.print_att)

    if not args.test_file:
        wlog('Translating one sentence ... ')
        # s = tc.Tensor([[0, 10811, 140, 217, 19, 1047, 482, 29999, 0, 0, 0]])
        #s = '( 北京 综合 电 ) 曾 因 美国 总统 布什 访华 而 一 度 升温 的 中 美 关系 在 急速 ' \
        #        '冷却 , 中国 昨天 证实 取消 今年 海军 舰艇 编队 访问 美国 港口 的 计划 , 并 ' \
        #        '拒绝 确定 国家 副主席 胡锦涛 是否 会 按 原定 计划 访美 。'
        #s = '黑夜 处处 有 , 神州 最 漫长 。'
        #s = '当 我 到 路口 时 我 这 边 的 灯 是 绿色 的 。'
        #s = '我 爱 北京 天安门 。'
        #s = '经过 国际 奥委会 的 不懈 努力 , 意大利 方面 在 冬奥会 开幕 前 四 天 作出 让步 , 承诺' \
        #' 冬奥会 期间 警方 不 会 进入 奥运村 搜查 运动员 驻地 , 但是 , 药检 呈 阳性 的 运动员 仍将'\
        #' 接受 意大利 检察 机关 的 调查 。'
        #s = '我 想 预约 一下 理发 。'
        #s = '玻利维亚 举行 总统 与 国会 选举 。'

        #s = '经过 国际 奥委会 的 不懈 努力 , 意大利 方面 在 冬奥会 开幕 前 四 天 作出 让步 , 承诺' \
        #' 冬奥会 期间 警方 不 会 进入 奥运村 搜查 运动员 驻地 , 但是 , 药检 呈 阳性 的 运动员 仍将' \
        #' 接受 意大利 检察 机关 的 调查 。'

        #s = '谢瓦尔德纳泽说 , 格鲁吉亚 内务 部队 、 警察 部队 、 安全 部门 和 国防部 在 去年 秋天'\
        #' 开展 的 特别 行动 中 , 已 将 潜入 潘基 西峡 谷 的 外国 武装 团伙 清理 干净 , 但 不 能 排除'\
        #' 部分 非法 武装 分子 以 难民 身份 继续 躲藏 在 潘基西 峡谷 。'

        #s = '询以 美国 将 于 何时 提出 旨在 执行 安理会 一四四一 号 决议案 的 后续 决议案 , 佛莱谢'\
        #' 表示 「 现在 言之过早 」 , 但是 美国 将 会 就 内容 措词 问题 与 盟国 磋商 。'

        #s = '新华社 华盛顿 1月 31日 电 ( 记者 王振华 ) 美国 商务部 31日 发布 的 报告 显示 , 尽管'\
        #' 美国 的 个人 消费 去年 12月份 出现 了 较 快 的 增长 , 但是 全 年 的 增长 幅度 仅 与 经济'\
        #' 发生 衰退 的 2001年 持平 。'

        #s = '在 路易斯 安那 , 我 看见 一棵 松树 在 生长 , 它 独自 站 在 那里 , 枝条 上 挂着 苔藓 。'
        #t = 'I saw in Louisiana a live-oak growing, all alone stood it and the 3)moss hung down' \
        #    ' from the branches.'

        #s = '好吧 , 我 想 我 会 一直 爱 , 但 我 从未 有 过 , 直 到 我 被 包裹 在 密西西比州 的 男人'\
        #' 的 怀抱 。'

        #s = "章启月 昨天 也 证实 了 俄罗斯 媒体 的 报道 , 说 中国 国家 主席 江泽民 前晚 应 " \
        #        "俄罗斯 总统 普京 的 要求 与 他 通 了 电话 , 双方 主要 是 就中 俄 互利 合作 " \
        #        "问题 交换 了 意见 。"
        #t = "( beijing , syndicated news ) the sino - us relation that was heated momentarily " \
        #        "by the us president bush 's visit to china is cooling down rapidly . china " \
        #        "confirmed yesterday that it has called off its naval fleet visit to the us " \
        #        "ports this year and refused to confirm whether the country 's vice president " \
        #        "hu jintao will visit the united states as planned ."

        #s = '当 林肯 去 新奥尔良 时 , 我 听到 密西 西比 河 的 歌声 。'
        #t = "When Lincoln goes to New Orleans, I hear Mississippi river's singing sound"
        s = '新奥尔良 是 爵士 音乐 的 发源 地 。'
        #s = '新奥尔良 以 其 美食 而 闻名 。'
        # = '休斯顿 是 仅 次于 新奥尔良 和 纽约 的 美国 第三 大 港 。'
        t = "When Lincoln goes to New Orleans, I hear Mississippi river's singing sound"

        s = [src_vocab.key2idx[x] if x in src_vocab.key2idx else UNK for x in s.split(' ')]
        #wlog(s)
        s = tc.Tensor([s])
        t = [trg_vocab.key2idx[x] if x in trg_vocab.key2idx else UNK for x in t.split(' ')]
        #wlog(t)
        t = tc.Tensor([t])
        pv = tc.Tensor([0, 10782, 2102, 1735, 4, 1829, 1657, 29999])
        tor.trans_samples(s, t)
        sys.exit(0)

    input_file = '{}{}.{}'.format(wargs.val_tst_dir, args.test_file, wargs.val_src_suffix)
    ref_file = '{}{}.{}'.format(wargs.val_tst_dir, args.test_file, wargs.val_ref_suffix)
    #input_file = args.test_file

    wlog('Translating test file {} ... '.format(input_file))
    test_src_tlst, test_src_lens = val_wrap_data(input_file, src_vocab)
    test_input_data = Input(test_src_tlst, None, 1, volatile=True)

    batch_tst_data = None
    if os.path.exists(ref_file):
        wlog('With force decoding test file {} ... to get alignments'.format(input_file))
        wlog('\t\tRef file {}'.format(ref_file))
        tst_src_tlst, tst_trg_tlst = wrap_data(input_file, ref_file, src_vocab, trg_vocab,
                                               False, False, 1000000)
        batch_tst_data = Input(tst_src_tlst, tst_trg_tlst, 10, batch_sort=False)

    trans, alns = tor.single_trans_file(test_input_data, batch_tst_data=batch_tst_data)
    #trans, alns = tor.single_trans_file(test_input_data)
    #trans = tor.multi_process(viter, n_process=nprocess)

    if wargs.search_mode == 0: p1 = 'greedy'
    elif wargs.search_mode == 1: p1 = 'nbs'
    elif wargs.search_mode == 2: p1 = 'cp'
    p2 = 'GPU' if wargs.gpu_id else 'CPU'
    p3 = 'wb' if wargs.with_batch else 'wob'

    #test_file_name = input_file if '/' not in input_file else input_file.split('/')[-1]
    outdir = 'wexp-{}-{}-{}-{}-{}'.format(args.test_file, p1, p2, p3, model_file.split('/')[0])
    if wargs.ori_search: outdir = '{}-{}'.format(outdir, 'ori')
    init_dir(outdir)
    outprefix = outdir + '/trans_' + args.test_file
    # wTrans/trans
    file_out = "{}_e{}_upd{}_b{}m{}_bch{}".format(
        outprefix, eid, bid, wargs.beam_size, wargs.search_mode, wargs.with_batch)

    mteval_bleu = tor.write_file_eval(file_out, trans, args.test_file, alns)

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

