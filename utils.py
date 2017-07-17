import sys
import os
import re
import numpy
import shutil
import const
import wargs
import subprocess

import torch as tc
from torch.autograd import Variable

def print_time(time):
    '''
        :type time: float
        :param time: the number of seconds

        :returns: string, the text format of time
    '''
    if time < 60:
        return '%.3f sec' % time
    elif time < 3600:
        return '%.3f min' % (time / 60)
    else:
        return '%.3f hr' % (time / 3600)

def get_gumbel(LB, V, eps=1e-30):

    vg = Variable(
        -tc.log(-tc.log(tc.Tensor(LB, V).uniform_(0, 1) + eps) + eps), requires_grad=False)

    return vg

def LBtensor_to_StrList(x, xs_L):

    B = x.size(1)
    x = x.data.numpy().T
    xs = []
    for bid in range(B):
        x_one = x[bid][:int(xs_L[bid])]
        #x_one = str(x_one.astype('S10'))[1:-1].replace('\n', '')
        x_one = str(x_one.astype('S10')).replace('\n', '')
        #x_one = x_one.__str__().replace('  ', ' ')[2:-1]
        xs.append(x_one)
    return xs

def LBtensor_to_Str(x, xs_L):

    B = x.size(1)
    x = x.data.numpy().T
    xs = []
    for bid in range(B):
        x_one = x[bid][:int(xs_L[bid])]
        #x_one = str(x_one.astype('S10'))[1:-1].replace('\n', '')
        x_one = str(x_one.astype('S10')).replace('\n', '')
        #x_one = x_one.__str__().replace('  ', ' ')[2:-1]
        xs.append(x_one)
    return '\n'.join(xs)

def init_params(p):

    if len(p.size()) == 2:
        if p.size(0) == 1 or p.size(1) == 1:
            p.data.zero_()
        else:
            p.data.normal_(0, 0.01)
    elif len(p.size()) == 1:
        p.data.zero_()

def str_cat(pp, name):
    return '{}_{}'.format(pp, name)

def wlog(obj, newline=1):
    if newline:
        sys.stderr.write('{}\n'.format(obj))
    else:
        sys.stderr.write('{}'.format(obj))

DEBUG = False
def debug(s, nl=True):
    if DEBUG:
        if nl:
            sys.stderr.write('{}\n'.format(s))
        else:
            sys.stderr.write(s)
        sys.stderr.flush()

def init_dir(dir_name, delete=False):

    if not dir_name == '':
        if os.path.exists(dir_name):
            if delete:
                shutil.rmtree(dir_name)
                wlog('{} exists, delete'.format(dir_name))
            else:
                wlog('{} exists, no delete'.format(dir_name))
        else:
            os.mkdir(dir_name)
            wlog('Create {}'.format(dir_name))

def sent_filter(sent):

    list_filter = filter(lambda x: x != const.PAD, sent)

    return list_filter

def idx2sent(vec, vcb_i2w):
    # vec: [int, int, ...]
    r = [vcb_i2w[idx] for idx in vec]
    return ' '.join(r)

def _filter_reidx(best_trans, tV_i2w=None, ifmv=False, ptv=None):

    if ifmv and ptv is not None:
        # OrderedDict([(0, 0), (1, 1), (3, 5), (8, 2), (10, 3), (100, 4)])
        # reverse: OrderedDict([(0, 0), (1, 1), (5, 3), (2, 8), (3, 10), (4, 100)])
        # part[index] get the real index in large target vocab firstly
        true_idx = [ptv[i] for i in best_trans]
    else:
        true_idx = best_trans

    true_idx = filter(lambda y: y != const.BOS and y != const.EOS, true_idx)

    return idx2sent(true_idx, tV_i2w), true_idx

def back_tracking(beam, best_sample_endswith_eos):
    # (0.76025655120611191, [29999], 0, 7)
    if wargs.with_norm:
        best_loss, accum, w, bp, endi = best_sample_endswith_eos
    else:
        best_loss, w, bp, endi = best_sample_endswith_eos
    # from previous beam of eos beam, firstly bp:j is the item index of
    # {end-1}_{th} beam
    seq = []
    check = (len(beam[0][0]) == 4)
    for i in reversed(xrange(1, endi)):
        # the best (minimal sum) loss which is the first one in the last beam,
        # then use the back pointer to find the best path backward
        # contain eos last word, finally we filter, so no matter
        if check:
            _, _, w, backptr = beam[i][bp]
        else:
            _, _, _, w, backptr = beam[i][bp]
        seq.append(w)
        bp = backptr
    return seq[::-1], best_loss  # reverse

def init_beam(beam, cnt=50, score_0=0.0, loss_0=0.0, hs0=None, s0=None, detail=False):
    del beam[:]
    for i in range(cnt + 1):
        ibeam = []  # one beam [] for one char besides start beam
        beam.append(ibeam)
    # indicator for the first target word (<b>)
    if detail:
        beam[0].append((loss_0, hs0, s0, const.BOS, 0))
    else:
        beam[0].append((loss_0, s0, const.BOS, 0))

def append_file(file_prefix, content):
    f = open(file_prefix, 'a')
    f.write(content)
    f.write('\n')
    f.close()

def fetch_bleu_from_file(fbleufrom):
    fread = open(fbleufrom, 'r')
    result = fread.readlines()
    fread.close()
    f_bleu = 0.
    f_multibleu = 0.
    for line in result:
        bleu_pattern = re.search(r'BLEU score = (0\.\d+)', line)
        if bleu_pattern:
            s_bleu = bleu_pattern.group(1)
            f_bleu = format(float(s_bleu) * 100, '0.2f')
        multi_bleu_pattern = re.search(r'BLEU = (\d+\.\d+)', line)
        if multi_bleu_pattern:
            s_multibleu = multi_bleu_pattern.group(1)
            f_multibleu = format(float(s_multibleu), '0.2f')
    return f_bleu, f_multibleu

# valid_out: valids/trans...
def valid_bleu(valid_out, val_tst_dir, val_prefix):

    save_log = '{}.{}'.format(valid_out, 'log')

    cmd = ['sh evaluate.sh {} {} {} {}'.format(
        valid_out,
        val_prefix,
        save_log,
        val_tst_dir)
    ]

    child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

    bleu_out = child.communicate()
    child.wait()
    mteval_bleu, multi_bleu = fetch_bleu_from_file(save_log)
    os.remove(save_log)
    # we use the bleu without unk and mteval-v11, process reference
    return float(mteval_bleu), float(multi_bleu)

# valid_out: valids/trans...
def Calc_BLEU(valid_out, val_tst_dir, val_prefix):

    save_log = '{}.{}'.format(valid_out, 'log')
    cmd = 'sh evaluate.sh {} {} {} {}'.format(valid_out, val_prefix, save_log, val_tst_dir)
    #f = os.popen(cmd)
    os.system(cmd)
    wlog(cmd)
    wlog('Waiting for evaluating BLEU ... 5s')
    time.sleep(5)
    #print f.readline() == ''
    wlog('{} exists ? {}'.format(save_log, os.path.exists(save_log)))
    mteval_bleu, multi_bleu = fetch_bleu_from_file(save_log)
    # we use the bleu without unk and mteval-v11, process reference
    return float(mteval_bleu), float(multi_bleu)

def part_sort(vec, num):
    '''
    vec:    [ 3,  4,  5, 12,  1,  3,  29999, 33,  2, 11,  0]
    '''

    idx = numpy.argpartition(vec, num)[:num]

    '''
    put k-min numbers before the _th position and get indexes of the k-min numbers in vec (unsorted)
    idx = np.argpartition(vec, 5)[:5]:
        [ 4, 10,  8,  0,  5]
    '''

    kmin_vals = vec[idx]

    '''
    kmin_vals:  [1, 0, 2, 3, 3]
    '''

    k_rank_ids = numpy.argsort(kmin_vals)

    '''
    k_rank_ids:    [1, 0, 2, 3, 4]
    '''

    k_rank_ids_invec = idx[k_rank_ids]

    '''
    k_rank_ids_invec:  [10,  4,  8,  0,  5]
    '''

    '''
    sorted_kmin = vec[k_rank_ids_invec]
    sorted_kmin:    [0, 1, 2, 3, 3]
    '''

    return k_rank_ids_invec


