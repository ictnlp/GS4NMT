from __future__ import division

import os
import math
import re
import sys
import numpy

def wlog(obj, newline=1):

    if newline: sys.stderr.write('{}\n'.format(obj))
    else: sys.stderr.write('{}'.format(obj))

'''
convert some code of Moses mteval-v11b.pl into python code
'''
def token(s):

    # language-independent part:
    s, n = re.subn('<skipped>', '', s)    # strip "skipped" tags
    s, n = re.subn('-\n', '', s)  # strip end-of-line hyphenation and join lines
    s, n = re.subn('\n', ' ', s)  # join lines
    s, n = re.subn('&quot;', '"', s)  # convert SGML tag for quote to "
    s, n = re.subn('&amp;', '&', s)   # convert SGML tag for ampersand to &
    s, n = re.subn('&lt;', '<', s)    # convert SGML tag for less-than to >
    s, n = re.subn('&gt;', '>', s)    # convert SGML tag for more-than to <

    # language-dependent part:
    s = ' ' + s + ' '
    s = s.lower()   # lowercase all characters

    # tokenize punctuation
    s, n = re.subn('([\{-\~\[-\` -\&\(-\+\:-\@\/])', lambda x: ' ' + x.group(0) + ' ', s)

    # tokenize period and comma unless preceded by a digit
    s, n = re.subn('([^0-9])([\.,])', lambda x: x.group(1) + ' ' + x.group(2) + ' ', s)

    # tokenize period and comma unless followed by a digit
    s, n = re.subn('([\.,])([^0-9])', lambda x: ' ' + x.group(1) + ' ' + x.group(2), s)

    # tokenize dash when preceded by a digit
    s, n = re.subn('([0-9])(-)', lambda x: x.group(1) + ' ' + x.group(2) + ' ', s)

    s, n = re.subn('\s+', ' ', s)    # only one space between words
    s, n = re.subn('^\s+', '', s)    # no leading space
    s, n = re.subn('\s+$', '', s)    # no trailing space

    return s

def merge_dict(d1, d2):
    '''
        Merge two dicts. The count of each item is the maximum count in two dicts.
    '''
    result = d1
    for key in d2:
        value = d2[key]
        if result.has_key(key):
            result[key] = max(result[key], value)
        else:
            result[key] = value
    return result

def sentence2dict(sentence, n):
    '''
        Count the number of n-grams in a sentence.

        :type sentence: string
        :param sentence: sentence text

        :type n: int
        :param n: maximum length of counted n-grams
    '''
    words = sentence.split(' ')
    result = {}
    for k in range(1, n + 1):
        for pos in range(len(words) - k + 1):
            gram = ' '.join(words[pos : pos + k])
            if result.has_key(gram):
                result[gram] += 1
            else:
                result[gram] = 1
    return result

def bleu(hypo_c, refs_c, n=4):
    '''
        Calculate BLEU score given translation and references.

        :type hypo_c: string
        :param hypo_c: the translations

        :type refs_c: list
        :param refs_c: the list of references

        :type n: int
        :param n: maximum length of counted n-grams
    '''
    #hypo_c="today weather very good", refs_c=["today weather good", "would rain"],n=4
    correctgram_count = [0] * n
    ngram_count = [0] * n
    hypo_sen = hypo_c.split('\n')
    refs_sen = [refs_c[i].split('\n') for i in range(len(refs_c))]
    hypo_length = 0
    ref_length = 0
    #print hypo_sen
    #print len(hypo_sen)
    for num in range(len(hypo_sen)):
        hypo = hypo_sen[num]
        hypo = token(hypo)
        h_length = len(hypo.split(' '))

        refs = [token(refs_sen[i][num]) for i in range(len(refs_c))]
        ref_lengths = sorted([len(refs[i].split(' ')) for i in range(len(refs))])

        # problem is not the brevity penalty, mteval-v11.perl of Moses also has brevity penalty,
        # the problem is Moses use the minimal length among four references
        ref_length += ref_lengths[0]
        hypo_length += h_length    # this is same with mteval-v11b.pl

        # why this ? more strict
        #hypo_length += (h_length if h_length < ref_lengths[0] else ref_lengths[0])

        #print ref_lengths[0], ref_length, h_length, hypo_length

        # another choice is use the minimal length difference of hypothesis and four references !!
        #ref_distances = [abs(r - h_length) for r in ref_lengths]
        #ref_length += ref_lengths[numpy.argmin(ref_distances)]
        '''
        if num == 0:
            print h_length
            print ref_lengths[0]
            for i in range(len(refs_c)):
                print token(refs_sen[i][num]), len(token(refs_sen[i][num]).split(' '))
            print ref_lengths[numpy.argmin(ref_distances)]
        '''
        refs_dict = {}
        for i in range(len(refs)):  # four refs for one sentence
            ref = refs[i]
            ref_dict = sentence2dict(ref, n)
            refs_dict = merge_dict(refs_dict, ref_dict)

        #if num == 0:
        #    for key in refs_dict.keys():
        #        print key, refs_dict[key]
        hypo_dict = sentence2dict(hypo, n)

        for key in hypo_dict:
            value = hypo_dict[key]
            length = len(key.split(' '))
            ngram_count[length - 1] += value
            #if num == 0:
            #    print key, value, length
            #    print min(value, refs_dict[key])
            if refs_dict.has_key(key):
                correctgram_count[length - 1] += min(value, refs_dict[key])

    result = 0.
    bleu_n = [0.] * n
    #if correctgram_count[0] == 0: return 0.
    wlog('Total words count, ref {}, hyp {}'.format(ref_length, hypo_length))
    for i in range(n):
        wlog('{}-gram, ref {}, match {}'.format(i+1, ngram_count[i], correctgram_count[i]), 0)
        if correctgram_count[i] == 0:
            #correctgram_count[i] += 1
            #ngram_count[i] += 1
            wlog('')
            return 0.
        bleu_n[i] = correctgram_count[i] / ngram_count[i]
        wlog('\tPrecision: {}'.format(bleu_n[i]))
        result += math.log(bleu_n[i]) / n

    bp = 1
    #bleu = geometric_mean(precisions) * bp     # same with mean function ?

    # there are no brevity penalty in mteval-v11b.pl, so with bp BLEU is a little lower
    if hypo_length < ref_length: bp = math.exp(1 - ref_length / hypo_length)

    BLEU = bp * math.exp(result)
    wlog('bp {} | bp_exp {} | {}-gram BLEU {}'.format(1 - ref_length / hypo_length, bp, n, BLEU))

    return BLEU

def bleu_file(hypo, refs, ngram=4):

    '''
        Calculate the BLEU score given translation files and reference files.

        :type hypo: string
        :param hypo: the path to translation file

        :type refs: list
        :param refs: the list of path to reference files
    '''

    wlog('Starting evaluating {}-gram BLEU ... '.format(ngram))
    wlog('\tcandidate file: {}'.format(hypo))
    wlog('\treferences file:')
    for ref in refs: wlog('\t\t{}'.format(ref))

    #hypo = open(hypo, 'r').read().strip('\n')
    #refs = [open(ref_fpath, 'r').read().strip('\n') for ref_fpath in refs]
    hypo = open(hypo, 'r').read().strip()
    refs = [open(ref_fpath, 'r').read().strip() for ref_fpath in refs]

    #print type(hypo)
    #print hypo.endswith('\n')
    #print type(refs)
    #print type(refs[0])
    result = bleu(hypo, refs, ngram)
    result = float('%.2f' % (result * 100))

    return result

if __name__ == "__main__":

    ref_fpaths = []
    for idx in range(4):
        #ref_fpath = '{}/{}'.format('work0', 'ref.seg.plain')
        #ref_fpath = '{}/{}'.format('data1', 'ref.seg.plain')
        #ref_fpath = '{}/{}'.format('data2', 'ref.seg.plain')
        #ref_fpath = '{}/{}'.format('data3', 'ref.seg.plain')
        ref_fpath = '{}{}'.format('/home/wen/3.corpus/segment_allnist_stanseg/nist03.ref', idx)
        if not os.path.exists(ref_fpath): continue
        ref_fpaths.append(ref_fpath)

    #print bleu_file('work0/hyp.seg.plain', ref_fpaths)
    #print bleu_file('data1/hyp.seg.plain', ref_fpaths)
    #print bleu_file('data2/hyp.seg.plain', ref_fpaths)
    #print bleu_file('data3/hyp.seg.plain', ref_fpaths)
    #print bleu_file('out', ref_fpaths)
    print bleu_file('trans_e10_upd15008_b10m1_bch1_32.64.txt', ref_fpaths)






