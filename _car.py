import wargs
import torch as tc
from torch import cuda
from inputs import Input
from utils import init_dir, wlog, sent_filter, load_pytorch_model
from optimizer import Optim
from train import *
from data_handler import *

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

from model_rnnsearch import *
from translate import Translator
from car_trainer import Trainer

class DataHisto():

    def __init__(self, chunk_D0):

        self.chunk_Ds = chunk_D0
        assert len(chunk_D0[0]) == len(chunk_D0[1])
        self.size = len(chunk_D0[0])

    def add_batch_data(self, chunk_Dk):

        self.chunk_Ds = (self.chunk_Ds[0] + chunk_Dk[0],
                         self.chunk_Ds[1] + chunk_Dk[1])
        self.size += len(chunk_Dk[0])

    def merge_batch(self, new_batch):

        # fine selectly sampling from history training chunks
        sample_xs, sample_ys = [], []

        '''
        while not len(sample_xs) == wargs.batch_size:
            k = np.random.randint(0, self.size, (1,))[0]
            srcL, trgL = len(self.chunk_Ds[0][k]), len(self.chunk_Ds[1][k])
            neg = (srcL<10) or (srcL>wargs.max_seq_len) or (trgL<10) or (trgL>wargs.max_seq_len)
            if neg: continue
            sample_xs.append(self.chunk_Ds[0][k])
            sample_ys.append(self.chunk_Ds[1][k])
        '''
        ids = np.random.randint(0, self.size, (wargs.batch_size,))
        for idx in ids:
            sample_xs.append(self.chunk_Ds[0][idx])
            sample_ys.append(self.chunk_Ds[1][idx])

        batch_src, batch_trg = [], []
        #shuf_idx = tc.randperm(new_batch[1].size(1))
        #for idx in range(new_batch[1].size(1) / 2):
        for idx in range(new_batch[1].size(1)):
            src = tc.Tensor(sent_filter(new_batch[1][:, idx].data.tolist()))
            trg = tc.Tensor(sent_filter(new_batch[2][:, idx].data.tolist()))
            sample_xs.append(src)
            sample_ys.append(trg)

        return Input(sample_xs, sample_ys, wargs.batch_size * 2)

def main():

    # Check if CUDA is available
    if cuda.is_available():
        wlog('CUDA is available, specify device by gpu_id argument (i.e. gpu_id=[3])')
    else:
        wlog('Warning: CUDA is not available, try CPU')

    if wargs.gpu_id:
        cuda.set_device(wargs.gpu_id[0])
        wlog('Using GPU {}'.format(wargs.gpu_id[0]))

    init_dir(wargs.dir_model)
    init_dir(wargs.dir_valid)
    init_dir(wargs.dir_tests)
    for prefix in wargs.tests_prefix:
        if not prefix == wargs.val_prefix: init_dir(wargs.dir_tests + '/' + prefix)

    wlog('Preparing data ... ', 0)

    train_srcD_file = wargs.dir_data + 'train.10k.zh5'
    wlog('\nPreparing source vocabulary from {} ... '.format(train_srcD_file))
    src_vocab = extract_vocab(train_srcD_file, wargs.src_dict, wargs.src_dict_size)

    train_trgD_file = wargs.dir_data + 'train.10k.en5'
    wlog('\nPreparing target vocabulary from {} ... '.format(train_trgD_file))
    trg_vocab = extract_vocab(train_trgD_file, wargs.trg_dict, wargs.trg_dict_size)

    train_src_file = wargs.dir_data + 'train.10k.zh0'
    train_trg_file = wargs.dir_data + 'train.10k.en0'
    wlog('\nPreparing training set from {} and {} ... '.format(train_src_file, train_trg_file))
    train_src_tlst, train_trg_tlst = wrap_data(train_src_file, train_trg_file, src_vocab, trg_vocab)
    #list [torch.LongTensor (sentence), torch.LongTensor, torch.LongTensor, ...], no padding
    wlog('Sentence-pairs count in training data: {}'.format(len(train_src_tlst)))
    src_vocab_size, trg_vocab_size = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(src_vocab_size, trg_vocab_size))
    batch_train = Input(train_src_tlst, train_trg_tlst, wargs.batch_size)

    tests_data = None
    if wargs.tests_prefix is not None:
        tests_data = {}
        for prefix in wargs.tests_prefix:
            test_file = wargs.val_tst_dir + prefix + '.src'
            test_src_tlst, _ = val_wrap_data(test_file, src_vocab)
            # we select best model by nist03 testing data
            if prefix == wargs.val_prefix:
                wlog('\nPreparing model-select set from {} ... '.format(test_file))
                batch_valid = Input(test_src_tlst, None, 1, volatile=True, prefix=prefix)
            else:
                wlog('\nPreparing test set from {} ... '.format(test_file))
                tests_data[prefix] = Input(test_src_tlst, None, 1, volatile=True)

    nmtModel = NMT()
    classifier = Classifier(wargs.out_size, trg_vocab_size)

    if wargs.pre_train:

        model_dict, class_dict, eid, bid, optim = load_pytorch_model(wargs.pre_train)
        if isinstance(optim, list): _, _, optim = optim
        # initializing parameters of interactive attention model
        for p in nmtModel.named_parameters(): p[1].data = model_dict[p[0]]
        for p in classifier.named_parameters(): p[1].data = class_dict[p[0]]
        #wargs.start_epoch = eid + 1
    else:

        for p in nmtModel.parameters(): init_params(p, uniform=True)
        for p in classifier.parameters(): init_params(p, uniform=True)
        optim = Optim(
            wargs.opt_mode, wargs.learning_rate, wargs.max_grad_norm,
            learning_rate_decay=wargs.learning_rate_decay,
            start_decay_from=wargs.start_decay_from,
            last_valid_bleu=wargs.last_valid_bleu
        )

    if wargs.gpu_id:
        wlog('Push model onto GPU ... ')
        nmtModel.cuda()
        classifier.cuda()
    else:
        wlog('Push model onto CPU ... ')
        nmtModel.cpu()
        classifier.cuda()

    nmtModel.classifier = classifier
    wlog(nmtModel)
    pcnt1 = len([p for p in nmtModel.parameters()])
    pcnt2 = sum([p.nelement() for p in nmtModel.parameters()])
    wlog('Parameters number: {}/{}'.format(pcnt1, pcnt2))

    optim.init_optimizer(nmtModel.parameters())

    #tor = Translator(nmtModel, src_vocab.idx2key, trg_vocab.idx2key)
    #tor.trans_tests(tests_data, pre_dict['epoch'], pre_dict['batch'])

    trainer = Trainer(nmtModel, src_vocab.idx2key, trg_vocab.idx2key, optim, trg_vocab_size)

    dev_src0 = wargs.dir_data + 'dev.1k.zh0'
    dev_trg0 = wargs.dir_data + 'dev.1k.en0'
    wlog('\nPreparing dev set for tuning from {} and {} ... '.format(dev_src0, dev_trg0))
    dev_src0, dev_trg0 = wrap_data(dev_src0, dev_trg0, src_vocab, trg_vocab)
    wlog(len(train_src_tlst))
    # add 1000 to train
    train_all_chunks = (train_src_tlst, train_trg_tlst)
    dh = DataHisto(train_all_chunks)

    dev_src1 = wargs.dir_data + 'dev.1k.zh1'
    dev_trg1 = wargs.dir_data + 'dev.1k.en1'
    wlog('\nPreparing dev set for tuning from {} and {} ... '.format(dev_src1, dev_trg1))
    dev_src1, dev_trg1 = wrap_data(dev_src1, dev_trg1, src_vocab, trg_vocab)

    dev_src2 = wargs.dir_data + 'dev.1k.zh2'
    dev_trg2 = wargs.dir_data + 'dev.1k.en2'
    wlog('\nPreparing dev set for tuning from {} and {} ... '.format(dev_src2, dev_trg2))
    dev_src2, dev_trg2 = wrap_data(dev_src2, dev_trg2, src_vocab, trg_vocab)

    dev_src3 = wargs.dir_data + 'dev.1k.zh3'
    dev_trg3 = wargs.dir_data + 'dev.1k.en3'
    wlog('\nPreparing dev set for tuning from {} and {} ... '.format(dev_src3, dev_trg3))
    dev_src3, dev_trg3 = wrap_data(dev_src3, dev_trg3, src_vocab, trg_vocab)

    dev_src4 = wargs.dir_data + 'dev.1k.zh4'
    dev_trg4 = wargs.dir_data + 'dev.1k.en4'
    wlog('\nPreparing dev set for tuning from {} and {} ... '.format(dev_src4, dev_trg4))
    dev_src4, dev_trg4 = wrap_data(dev_src4, dev_trg4, src_vocab, trg_vocab)
    wlog(len(dev_src4+dev_src3+dev_src2+dev_src1+dev_src0))
    dev_input = Input(dev_src4+dev_src3+dev_src2+dev_src1+dev_src0, dev_trg4+dev_trg3+dev_trg2+dev_trg1+dev_trg0, wargs.batch_size)
    trainer.train(dh, dev_input, 0, batch_valid, tests_data, merge=True, name='DH_{}'.format('dev'))

    '''
    chunk_size = 1000
    rand_ids = tc.randperm(len(train_src_tlst))[:chunk_size * 1000]
    rand_ids = rand_ids.split(chunk_size)
    #train_chunks = [(dev_src, dev_trg)]
    train_chunks = []
    for k in range(len(rand_ids)):
        rand_id = rand_ids[k]
        chunk_src_tlst = [train_src_tlst[i] for i in rand_id]
        chunk_trg_tlst = [train_trg_tlst[i] for i in rand_id]
        #wlog('Sentence-pairs count in training data: {}'.format(len(src_samples_train)))
        #batch_train = Input(train_src_tlst, train_trg_tlst, wargs.batch_size)
        #batch_train = Input(src_samples_train, trg_samples_train, wargs.batch_size)
        train_chunks.append((chunk_src_tlst, chunk_trg_tlst))

    chunk_D0 = train_chunks[0]
    dh = DataHisto(chunk_D0)
    c0_input = Input(chunk_D0[0], chunk_D0[1], wargs.batch_size)
    trainer.train(dh, c0_input, 0, batch_valid, tests_data, merge=False, name='DH_{}'.format(0))
    for k in range(1, len(train_chunks)):
        wlog('*' * 30, False)
        wlog(' Next Data {} '.format(k), False)
        wlog('*' * 30)
        chunk_Dk = train_chunks[k]
        ck_input = Input(chunk_Dk[0], chunk_Dk[1], wargs.batch_size)
        trainer.train(dh, ck_input, k, batch_valid, tests_data, merge=True, name='DH_{}'.format(k))
        dh.add_batch_data(chunk_Dk)
    '''

    if tests_data and wargs.final_test:

        bestModel = NMT()
        classifier = Classifier(wargs.out_size, trg_vocab_size)

        assert os.path.exists(wargs.best_model)
        model_dict = tc.load(wargs.best_model)

        best_model_dict = model_dict['model']
        best_model_dict = {k: v for k, v in best_model_dict.items() if 'classifier' not in k}

        bestModel.load_state_dict(best_model_dict)
        classifier.load_state_dict(model_dict['class'])

        if wargs.gpu_id:
            wlog('Push NMT model onto GPU ... ')
            bestModel.cuda()
            classifier.cuda()
        else:
            wlog('Push NMT model onto CPU ... ')
            bestModel.cpu()
            classifier.cpu()

        bestModel.classifier = classifier

        tor = Translator(bestModel, src_vocab.idx2key, trg_vocab.idx2key)
        tor.trans_tests(tests_data, model_dict['epoch'], model_dict['batch'])





if __name__ == "__main__":

    main()















