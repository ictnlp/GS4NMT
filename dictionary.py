import torch as tc
from utils import *

class Dictionary(object):

    def __init__(self, filename=None, real=True):

        self.idx2key = {}
        self.key2idx = {}
        self.real = real
        self.freq = {}

        if filename is not None:
            self.load_from_file(data)
        else:
            self.idx2key[PAD] = PAD_WORD
            self.idx2key[UNK] = UNK_WORD
            self.idx2key[BOS] = BOS_WORD
            self.idx2key[EOS] = EOS_WORD

            self.key2idx[PAD_WORD] = PAD
            self.key2idx[UNK_WORD] = UNK
            self.key2idx[BOS_WORD] = BOS
            self.key2idx[EOS_WORD] = EOS

    def __repr__(self):

        rst = []
        for k, v in self.idx2key.iteritems():
            rst.append('{}:{}'.format(k, v))
        return ' '.join(rst)

    def size(self):
        return len(self.idx2key)

    def add(self, key, idx=None):

        if idx is not None:
            self.idx2key[idx] = key
            self.key2idx[key] = idx
        else:
            if key in self.key2idx:
                idx = self.key2idx[key]
            else:
                idx = len(self.key2idx)
                self.key2idx[key] = idx
                self.idx2key[idx] = key

            if idx not in self.freq:
                self.freq[idx] = 1
            else:
                self.freq[idx] += 1

    def keep_vocab_size(self, vocab_size):

        if self.size() <= vocab_size:
            wlog('{} <= {} tokens, Bingo!~'.format(self.size(), vocab_size))
            return self, sum(self.freq.itervalues())

        idx_freq = [k for k in self.freq.iterkeys()]
        _, idx = tc.sort(tc.Tensor(
            [self.freq[k] for k in idx_freq]),
            dim=0,
            descending=True)
        keep_dict = Dictionary()
        keep_word_cnt = 0
        for i in idx[:vocab_size]:
            keep_dict.add(self.idx2key[idx_freq[i]])
            keep_word_cnt += self.freq[idx_freq[i]]

        return keep_dict, keep_word_cnt

    def load_from_file(self, filename):

        idx2key = tc.load(filename)
        for idx, key in idx2key.iteritems():
            self.add(key, idx)

    def write_into_file(self, filename):

        content = ''
        for idx in range(self.size()):
            key = self.idx2key[idx]
            content += '{} {}\n'.format(key, idx)

        tc.save(self.idx2key, filename)

        if self.real:
            txt_dict_file = filename + '.txt'
            file = open(txt_dict_file, 'w')
            file.write(content)

        file.close()

    def keys2idx(self, list_words, unk_word, bos_word=None, eos_word=None):

        list_idx = [self.key2idx[bos_word]] if bos_word else []
        for w in list_words:
            list_idx.extend(
                [self.key2idx[w] if w in self.key2idx else self.key2idx[unk_word]]
            )
        list_idx.extend([self.key2idx[eos_word]] if eos_word else [])

        # we use int32 to represent the index of vocabulary
        # enough: [-2,147,483,648 to 2,147,483,647]
        return tc.LongTensor(list_idx)


