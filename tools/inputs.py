from __future__ import division

import math
import wargs
import torch as tc
from utils import *
from torch.autograd import Variable

class Input(object):

    def __init__(self, src_tlst, trg_tlst, batch_size, volatile=False, batch_sort=True):

        self.src_tlst = src_tlst

        cnt_sent = len(src_tlst)

        if trg_tlst is not None:
            self.trg_tlst = trg_tlst
            assert cnt_sent == len(trg_tlst)
            wlog('Build bilingual Input, Batch size {}, Sort in batch? {}'.format(batch_size, batch_sort))
        else:
            self.trg_tlst = None
            wlog('Build monolingual Input, Batch size {}, Sort in batch? {}'.format(batch_size, batch_sort))

        self.batch_size = batch_size
        self.gpu_id = wargs.gpu_id
        self.volatile = volatile
        self.batch_sort = batch_sort

        self.num_of_batches = int(math.ceil(cnt_sent / self.batch_size))

    def __len__(self):

        return self.num_of_batches
        #return len(self.src_tlst)

    def handle_batch(self, batch, right_align=False):

        lens = [ts.size(0) for ts in batch]

        self.this_batch_size = len(batch)
        max_len_batch = max(lens)

        # 80 * 40
        pad_batch = tc.Tensor(self.this_batch_size, max_len_batch).long()
        pad_batch.fill_(PAD)

        for idx in range(self.this_batch_size):
            length = lens[idx]
            offset = max_len_batch - length if right_align else 0
            # modify Tensor pad_batch
            pad_batch[idx].narrow(0, offset, length).copy_(batch[idx])

        return pad_batch, lens

    def __getitem__(self, idx):

        assert idx < self.num_of_batches, \
                'idx:{} >= number of batches:{}'.format(idx, self.num_of_batches)

        src_batch = self.src_tlst[idx * self.batch_size : (idx + 1) * self.batch_size]

        srcs, slens = self.handle_batch(src_batch)

        if self.trg_tlst is not None:

            trg_batch = self.trg_tlst[idx * self.batch_size : (idx + 1) * self.batch_size]
            trgs, tlens = self.handle_batch(trg_batch)

        # sort the source and target sentence
        idxs = range(self.this_batch_size)

        if self.batch_sort is True:
            if self.trg_tlst is None:
                zipb = zip(idxs, srcs, slens)
                idxs, srcs, slens = zip(*sorted(zipb, key=lambda x: x[-1]))
            else:
                zipb = zip(idxs, srcs, trgs, slens)
                idxs, srcs, trgs, slens = zip(*sorted(zipb, key=lambda x: x[-1]))

        lengths = tc.IntTensor(slens).view(1, -1)   # (1, batch_size)
        lengths = Variable(lengths, volatile=self.volatile)

        def tuple2Tenser(tx):

            if tx is None: return tx

            # (max_len_batch, batch_size)
            tx = tc.stack(tx, dim=0).t().contiguous()
            if wargs.gpu_id: tx = tx.cuda()    # push into GPU

            return Variable(tx, volatile=self.volatile)

        tsrcs = tuple2Tenser(srcs)
        src_mask = tsrcs.ne(0).float()

        if self.trg_tlst is not None:

            ttrgs = tuple2Tenser(trgs)
            trg_mask = ttrgs.ne(0).float()

            return idxs, tsrcs, ttrgs, lengths, src_mask, trg_mask

        else:

            return idxs, tsrcs, lengths, src_mask


    def shuffle(self):

        data = list(zip(self.src_tlst, self.trg_tlst))
        self.src_tlst, self.trg_tlst = zip(*[data[i] for i in tc.randperm(len(data))])





