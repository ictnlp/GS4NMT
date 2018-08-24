import sys
import os
import re
import numpy
import shutil
import wargs
import json
import subprocess
import math
import random

import torch as tc
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

reload(sys)
sys.setdefaultencoding('utf-8')


def str1(content, encoding='utf-8'):
    return json.dumps(content, encoding=encoding, ensure_ascii=False, indent=4)
    pass

#DEBUG = True
DEBUG = False

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<pad>'
UNK_WORD = 'unk'
BOS_WORD = '<b>'
EOS_WORD = '<e>'

epsilon = 1e-20

class MaskSoftmax(nn.Module):

    def __init__(self):

        super(MaskSoftmax, self).__init__()

    def forward(self, x, mask=None, dim=-1):

        # input torch tensor or variable, take max for numerical stability
        x_max = tc.max(x, dim=dim, keepdim=True)[0]
        x_minus = x - x_max
        x_exp = tc.exp(x_minus)
        if mask is not None: x_exp = x_exp * mask
        x = x_exp / ( tc.sum( x_exp, dim=dim, keepdim=True ) + epsilon )

        return x

def log_prob(x, self_norm_alpha=None):

    # input torch tensor or variable
    x_max = tc.max(x, dim=-1, keepdim=True)[0]  # take max for numerical stability
    log_norm = tc.log( tc.sum( tc.exp( x - x_max ), dim=-1, keepdim=True ) + epsilon ) + x_max
    # get log softmax
    x = x - log_norm

    # Sum_( log(P(xi)) - alpha * square( log(Z(xi)) ) )
    if self_norm_alpha is not None: x = x - self_norm_alpha * tc.pow(log_norm, 2)

    return log_norm, x

def _load_model(model_path):
    state_dict = tc.load(model_path, map_location=lambda storage, loc: storage)
    if len(state_dict) == 4:
        model_dict, eid, bid, optim = state_dict['model'], state_dict['epoch'], state_dict['batch'], state_dict['optim']
        rst = ( model_dict, eid, bid, optim )
    elif len(state_dict) == 5:
        model_dict, class_dict, eid, bid, optim = state_dict['model'], state_dict['class'], state_dict['epoch'], state_dict['batch'], state_dict['optim']
        rst = ( model_dict, class_dict, eid, bid, optim )
    wlog('Loading pre-trained model from {} at epoch {} and batch {}'.format(model_path, eid, bid))
    wlog('Loading optimizer from {}'.format(model_path))
    wlog(optim)
    return rst

def toVar(x, isCuda=None):

    if not isinstance(x, tc.autograd.variable.Variable):
        if isinstance(x, int): x = tc.Tensor([x]).long()
        elif isinstance(x, list): x = tc.Tensor(x).long()
        if isCuda is not None: x = x.cuda()
        x = Variable(x, requires_grad=False, volatile=True)

    return x

def rm_elems_byid(l, ids):

    isTensor = isinstance(l, tc.FloatTensor)
    isTorchVar = isinstance(l, tc.autograd.variable.Variable)
    if isTensor is True: l = l.transpose(0, 1).tolist()
    if isTorchVar is True: l = l.transpose(0, 1).data.tolist() #  -> (B, srcL)

    if isinstance(ids, int): del l[ids]
    elif len(ids) == 1: del l[ids[0]]
    else:
        for idx in ids: l[idx] = PAD_WORD
        l = filter(lambda a: a != PAD_WORD, l)

    if isTensor is True: l = tc.Tensor(l).transpose(0, 1)  # -> (srcL, B')
    if isTorchVar is True:
        l = Variable(tc.Tensor(l).transpose(0, 1), requires_grad=False, volatile=True)
        if wargs.gpu_id: l = l.cuda()

    return l

# x, y are torch Tensors
def cor_coef(x, y):

    E_x, E_y = tc.mean(x), tc.mean(y)
    E_x_2, E_y_2 = tc.mean(x * x), tc.mean(y * y)
    rho = tc.mean(x * y) - E_x * E_y
    D_x, D_y = E_x_2 - E_x * E_x, E_y_2 - E_y * E_y
    return rho / math.sqrt(D_x * D_y) + eps

def format_time(time):
    '''
        :type time: float
        :param time: the number of seconds

        :print the text format of time
    '''
    rst = ''
    if time < 0.1: rst = '{:7.2f} ms'.format(time * 1000)
    elif time < 60: rst = '{:7.5f} sec'.format(time)
    elif time < 3600: rst = '{:6.4f} min'.format(time / 60.)
    else: rst = '{:6.4f} hr'.format(time / 3600.)

    return rst

def append_file(filename, content):

    f = open(filename, 'a')
    f.write(content + '\n')
    f.close()

def str_cat(pp, name):

    return '{}_{}'.format(pp, name)

def wlog(obj, newline=1):
    if newline == 1: sys.stderr.write('{}\n'.format(obj))
    else: sys.stderr.write('{}'.format(obj))

def debug(s, newline=1):

    if DEBUG is True:
        if newline == 1: sys.stderr.write('{}\n'.format(s))
        else: sys.stderr.write(s)
        sys.stderr.flush()

def get_gumbel(LB, V, eps=1e-30):

    return Variable(
        -tc.log(-tc.log(tc.Tensor(LB, V).uniform_(0, 1) + eps) + eps), requires_grad=False)

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

def init_params(p, name='what', uniform=False):

    if uniform is True:
        p.data.uniform_(-0.1, 0.1)
        wlog('{:7} -> grad {}\t{}'.format('Uniform', p.requires_grad, name))
    else:
        if len(p.size()) == 2:
            if p.size(0) == 1 or p.size(1) == 1:
                p.data.zero_()
                wlog('{:7}-> grad {}\t{}'.format('Zero', p.requires_grad, name))
            else:
                p.data.normal_(0, 0.01)
                wlog('{:7}-> grad {}\t{}'.format('Normal', p.requires_grad, name))
        elif len(p.size()) == 1:
            p.data.zero_()
            wlog('{:7}-> grad {}\t{}'.format('Zero', p.requires_grad, name))

def init_dir(dir_name, delete=False):

    if not dir_name == '':
        if os.path.exists(dir_name):
            if delete:
                shutil.rmtree(dir_name)
                wlog('\n{} exists, delete'.format(dir_name))
            else:
                wlog('\n{} exists, no delete'.format(dir_name))
        else:
            os.mkdir(dir_name)
            wlog('\nCreate {}'.format(dir_name))

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


# beam search
def init_beam(beam, cnt=50, score_0=0.0, loss_0=0.0, hs0=None, s0=None, dyn_dec_tup=None):
    del beam[:]
    for i in range(cnt + 1):
        ibeam = []  # one beam [] for one char besides start beam
        beam.append(ibeam)
    # indicator for the first target word (<b>)
    if dyn_dec_tup is not None:
        beam[0].append((loss_0, dyn_dec_tup, s0, BOS, 0))
    else:
        beam[0].append((loss_0, s0, BOS, 0))

def back_tracking(beam, best_sample_endswith_eos, attent_probs=None):
    # (0.76025655120611191, [29999], 0, 7)
    if wargs.len_norm: best_loss, accum, w, bp, endi = best_sample_endswith_eos
    else: best_loss, w, bp, endi = best_sample_endswith_eos
    # starting from bp^{th} item in previous {end-1}_{th} beam of eos beam, w is <eos>
    seq = []
    attent_matrix = [] if attent_probs is not None else None
    check = (len(beam[0][0]) == 4)
    #print len(attent_probs), endi
    for i in reversed(xrange(1, endi)): # [1, endi-1], not <bos> 0 and <eos> endi
        # the best (minimal sum) loss which is the first one in the last beam,
        # then use the back pointer to find the best path backward
        # <eos> is in pos endi, we do not keep <eos>
        if check is True:
            _, _, w, backptr = beam[i][bp]
        else:
            _, _, _, w, backptr = beam[i][bp]
        seq.append(w)
        bp = backptr
        # ([first word, ..., last word]) not bos and eos
        if attent_matrix is not None: attent_matrix.append(attent_probs[i-1][:, bp])

    if attent_probs is not None and len(attent_matrix) > 0:
        # attent_matrix: (trgL, srcL)
        attent_matrix = tc.stack(attent_matrix[::-1], dim=0)
        attent_matrix = attent_matrix.cpu().data.numpy()

    return seq[::-1], best_loss, attent_matrix # reverse

def filter_reidx(best_trans, tV_i2w=None, ifmv=False, ptv=None):

    if ifmv and ptv is not None:
        # OrderedDict([(0, 0), (1, 1), (3, 5), (8, 2), (10, 3), (100, 4)])
        # reverse: OrderedDict([(0, 0), (1, 1), (5, 3), (2, 8), (3, 10), (4, 100)])
        # part[index] get the real index in large target vocab firstly
        true_idx = [ptv[i] for i in best_trans]
    else:
        true_idx = best_trans

    true_idx = filter(lambda y: y != BOS and y != EOS, true_idx)

    return idx2sent(true_idx, tV_i2w), true_idx

def sent_filter(sent):

    list_filter = filter(lambda x: x != PAD and x!= BOS and x != EOS, sent)

    return list_filter

def idx2sent(vec, vcb_i2w):
    # vec: [int, int, ...]
    r = [vcb_i2w[idx] for idx in vec]
    return ' '.join(r)

def dec_conf():

    wlog('\n######################### Construct Decoder #########################\n')
    if wargs.search_mode == 0: wlog('# Greedy search => ')
    elif wargs.search_mode == 1: wlog('# Naive beam search => ')
    elif wargs.search_mode == 2: wlog('# Cube pruning => ')

    wlog('\t Beam size: {}'
         '\n\t KL_threshold: {}'
         '\n\t Batch decoding: {}'
         '\n\t Vocab normalized: {}'
         '\n\t Length normalized: {}'
         '\n\t Manipulate vocab: {}'
         '\n\t Cube pruning merge way: {}'
         '\n\t Average attent: {}\n\n'.format(
             wargs.beam_size,
             wargs.m_threshold,
             True if wargs.with_batch else False,
             True if wargs.vocab_norm else False,
             True if wargs.len_norm else False,
             True if wargs.with_mv else False,
             wargs.merge_way,
             True if wargs.avg_att else False
         )
    )

def memory_efficient(outputs, gold, gold_mask, classifier):

    batch_loss, batch_correct_num = 0, 0
    outputs = Variable(outputs.data, requires_grad=True, volatile=False)
    cur_batch_count = outputs.size(1)

    os_split = tc.split(outputs, wargs.snip_size)
    gs_split = tc.split(gold, wargs.snip_size)
    ms_split = tc.split(gold_mask, wargs.snip_size)

    for i, (o_split, g_split, m_split) in enumerate(zip(os_split, gs_split, ms_split)):

        loss, correct_num = classifier(o_split, g_split, m_split)
        batch_loss += loss.data[0]
        batch_correct_num += correct_num.data[0]
        loss.div(cur_batch_count).backward()
        del loss, correct_num

    grad_output = None if outputs.grad is None else outputs.grad.data

    return batch_loss, grad_output, batch_correct_num

def print_attention_text(attention_matrix, source_tokens, target_tokens, threshold=0.9, isP=False):
    """
    Return the alignment string from the attention matrix.
    Prints the attention matrix to standard out.
    :param attention_matrix: The attention matrix, np.ndarray, (trgL, srcL)
    :param source_tokens: A list of source tokens, List[str]
    :param target_tokens: A list of target tokens, List[str]
    :param threshold: The threshold for including an alignment link in the result, float
    """

    #assert attention_matrix.shape[0] == len(target_tokens)

    if isP is True:
        sys.stdout.write("  ")
        for j in target_tokens: sys.stdout.write("---")
        sys.stdout.write("\n")

    alnList = []
    src_max_ids, src_max_p = attention_matrix.argmax(1) + 1, attention_matrix.max(1)
    for (i, f_i) in enumerate(source_tokens):
        #maxJ, maxP = 0, 0.0

        if isP is True: sys.stdout.write(" |")
        for (j, _) in enumerate(target_tokens):
            align_prob = attention_matrix[j, i]
            if i == 0:  # start from 1
                alnList.append('{}:{}/{:.2f}'.format(src_max_ids[j], j+1, src_max_p[j]))
                #if maxP >= 0.5:
                #    alnList.append('{}:{}/{:.2f}'.format(i + 1, maxJ + 1, maxP))    # start from 1 here
            if isP is True:
                if align_prob > threshold: sys.stdout.write("(*)")
                elif align_prob > 0.4: sys.stdout.write("(?)")
                else: sys.stdout.write("   ")
            #if align_prob > maxP: maxJ, maxP = j, align_prob

        if isP is True: sys.stdout.write(" | %s\n" % f_i)

    if isP is True:
        sys.stdout.write("  ")
        for j in target_tokens:
            sys.stdout.write("---")
        sys.stdout.write("\n")
        for k in range(max(map(len, target_tokens))):
            sys.stdout.write("  ")
            for word in target_tokens:
                letter = word[k] if len(word) > k else " "
                sys.stdout.write(" %s " % letter)
            sys.stdout.write("\n")
        sys.stdout.write("\n")

    return ' '.join(alnList)

def plot_attention(attention_matrix, source_tokens, target_tokens, filename):
    """
    Uses matplotlib for creating a visualization of the attention matrix.
    :param attention_matrix: The attention matrix, np.ndarray
    :param source_tokens: A list of source tokens, List[str]
    :param target_tokens: A list of target tokens, List[str]
    :param filename: The file to which the attention visualization will be written to, str
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    #from pylab import mpl

    matplotlib.rc('font', family='sans-serif')
    #matplotlib.rc('font', serif='HelveticaNeue')
    matplotlib.rc('font', serif='SimHei')
    #matplotlib.rc('font', serif='Microsoft YaHei')
    #mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #mpl.rcParams['font.sans-serif'] = ['SimHei']
    #plt.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
    #mpl.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
    #plt.rcParams['font.sans-serif']=['WenQuanYi Micro Hei']
    #matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    #plt.rcParams['axes.unicode_minus'] = False
    #mpl.rcParams['axes.unicode_minus'] = False
    #zh_font = mpl.font_manager.FontProperties(fname='/home5/wen/miniconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf')

    assert attention_matrix.shape[0] == len(target_tokens)

    plt.clf()
    #plt.imshow(attention_matrix.transpose(), interpolation="nearest", cmap="Greys")
    plt.imshow(attention_matrix, interpolation="nearest", cmap="Greys")
    #plt.xlabel("Source", fontsize=16)
    #plt.ylabel("Target", fontsize=16)

    #plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('top')
    #plt.xticks(fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18)
    #plt.yticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18)

    #plt.grid(True, which='minor', linestyle='-')
    #plt.gca().set_xticks([i for i in range(0, len(target_tokens))])
    #plt.gca().set_yticks([i for i in range(0, len(source_tokens))])
    plt.gca().set_xticks([i for i in range(0, len(source_tokens))])
    plt.gca().set_yticks([i for i in range(0, len(target_tokens))])
    #plt.gca().set_xticklabels(source_tokens, rotation='vertical')
    #plt.gca().set_xticklabels(source_tokens, rotation=45, fontsize=20, fontweight='bold')
    plt.gca().set_xticklabels(source_tokens, rotation=70, fontsize=20)

    #source_tokens = [unicode(k, "utf-8") for k in source_tokens]
    #plt.gca().set_yticklabels(source_tokens, rotation='horizontal', fontproperties=zh_font)
    #plt.gca().set_yticklabels(source_tokens, rotation='horizontal')
    plt.gca().set_yticklabels(target_tokens, fontsize=24, fontweight='bold')

    plt.tight_layout()
    #plt.draw()
    #plt.show()
    #plt.savefig(filename, format='png', dpi=400)
    #plt.grid(True)
    #plt.savefig(filename, dpi=400)
    plt.savefig(filename, format='svg', dpi=600, bbox_inches='tight')
    #plt.savefig(filename)
    wlog("Saved alignment visualization to " + filename)

'''Layer normalize the tensor x, averaging over the last dimension.'''
class Layer_Norm(nn.Module):

    def __init__(self, d_hid, eps=1e-3):
        super(Layer_Norm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(tc.ones(d_hid), requires_grad=True)
        self.b = nn.Parameter(tc.zeros(d_hid), requires_grad=True)

    def forward(self, z):

        if z.size(-1) == 1: return z
        mu = tc.mean(z, dim=-1, keepdim=True)
        #sigma = tc.std(z, dim=-1, keepdim=True) # std has problems
        variance = tc.mean(tc.pow(z - mu, 2), dim=-1, keepdim=True)
        sigma = tc.sqrt(variance)
        z_norm = tc.div((z - mu), (sigma + self.eps))
        if z_norm.is_cuda:
            z_norm = z_norm * self.g.cuda().expand_as(z_norm) + self.b.cuda().expand_as(z_norm)
        else:
            z_norm = z_norm * self.g.expand_as(z_norm) + self.b.expand_as(z_norm)
        return z_norm

'''Apply Normalization.'''
def apply_norm(x, epsilon, norm_type=None):
    if norm_type is None: return x
    if norm_type == "layer":
        ln = Layer_Norm(x.size(-1), eps=epsilon)
        print (ln(x))
        return ln(x)
    if norm_type == "batch":
        bn = nn.BatchNorm1d(x.size(-1), eps=epsilon, momentum=0.99)
        return bn(x)
    if norm_type == "noam":
        """One version of layer normalization."""
        return F.normalize(x, p=2, dim=-1, eps=epsilon)
    raise ValueError("Parameter normalizer must be one of: 'layer', 'batch', 'noam', 'none'.")

'''
    a: add previous input tensor
    n: apply normalization
    d: apply dropout
  For example, if sequence=="dna", then the output is
    previous_value + normalize(dropout(x))
  Args:
    pre_layer_in: Tensor to be added as a residual connection ('a')
    pre_layer_out: Tensor to be transformed.
    handle_type: 'dna'
    norm_type: None, 'layer', 'batch', 'noam'
    epsilon: a float (parameter for normalization)
    dropout_rate: a float
  Returns:
    a Tensor
'''
def layer_prepostprocess(pre_layer_out, pre_layer_in=None, handle_type=None, norm_type=None,
                         epsilon=1e-6, dropout_rate=0., training=True):
    if handle_type is None: return pre_layer_out
    if 'a' in handle_type: assert pre_layer_in is not None, 'Residual requires previous input !'
    for c in handle_type:
      if c == 'a': pre_layer_out += pre_layer_in
      elif c == 'n': pre_layer_out = apply_norm(pre_layer_out, epsilon, norm_type)
      elif c == 'd': pre_layer_out = F.dropout(pre_layer_out, p=dropout_rate, training=training)
      else: wlog('Unknown handle type {}'.format(c))
    return pre_layer_out



