import wargs
import torch as tc
from torch import cuda
from inputs import Input
from utils import init_dir, wlog, sent_filter
from optimizer import Optim
from train import *
import const

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

from model_rnnsearch import *
from translate import Translator
#from gan_train import Trainer
#from gan_sample_train import Trainer
#from rl_gan_train import Trainer
#from gan_train_rl import Trainer
from gan_train_rl_fit import Trainer

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

        # sample from history training chunks
        ids = np.random.randint(0, self.size, (wargs.batch_size / 2,))
        sample_xs, sample_ys = [], []
        for idx in ids:
            sample_xs.append(self.chunk_Ds[0][idx])
            sample_ys.append(self.chunk_Ds[1][idx])

        batch_src, batch_trg = [], []
        shuf_idx = tc.randperm(new_batch[1].size(1))
        for idx in range(new_batch[1].size(1) / 2):
            src = tc.Tensor(sent_filter(new_batch[1][:, shuf_idx[idx]].data.tolist()))
            trg = tc.Tensor(sent_filter(new_batch[2][:, shuf_idx[idx]].data.tolist()))
            sample_xs.append(src)
            sample_ys.append(trg)

        return Input(sample_xs, sample_ys, wargs.batch_size)

def main():

    # Check if CUDA is available
    if cuda.is_available():
        wlog('CUDA is available, specify device by gpu_id argument (i.e. gpu_id=[3])')
    else:
        wlog('Warning: CUDA is not available, train CPU')

    if wargs.gpu_id: cuda.set_device(wargs.gpu_id[0])

    init_dir(wargs.dir_model)
    init_dir(wargs.dir_valid)
    init_dir(wargs.dir_tests)
    for prefix in wargs.tests_prefix: init_dir(wargs.dir_tests + '/' + prefix)

    wlog('Loading data ... ', 0)

    inputs_dict = tc.load(wargs.inputs_data)

    vocab_data, train_data, valid_data = inputs_dict['vocab'], inputs_dict['train'], inputs_dict['valid']

    batch_dev = None
    if inputs_dict.has_key('dev'):
        dev_data = {}
        dev_src, dev_trg = inputs_dict['dev']['src'], inputs_dict['dev']['trg']
        batch_dev = Input(dev_src, dev_trg, wargs.batch_size)

    train_src_tlst, train_trg_tlst = train_data['src'], train_data['trg']
    train_data_input = Input(train_src_tlst, train_trg_tlst, wargs.batch_size)
    if wargs.train_shuffle:
        train_data_input.shuffle()
        wlog('finish shuffle training data ...')

    train_src_tlst, train_trg_tlst = train_data_input.src_tlst, train_data_input.trg_tlst
    valid_src_tlst, valid_src_lens = valid_data['src'], valid_data['len']
    batch_valid = Input(valid_src_tlst, None, 1, volatile=True)

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

    src_vocab_size, trg_vocab_size = vocab_data['src'].size(), vocab_data['trg'].size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(src_vocab_size, trg_vocab_size))

    tests_data = None
    if inputs_dict.has_key('tests'):
        tests_data = {}
        tests_tensor = inputs_dict['tests']
        for prefix in tests_tensor.keys():
            tests_data[prefix] = Input(tests_tensor[prefix], None, 1, volatile=True)


    '''
    # lookup_table on cpu to save memory
    src_lookup_table = nn.Embedding(wargs.src_dict_size + 4,
                                    wargs.src_wemb_size, padding_idx=const.PAD).cpu()
    trg_lookup_table = nn.Embedding(wargs.trg_dict_size + 4,
                                    wargs.trg_wemb_size, padding_idx=const.PAD).cpu()

    wlog('Lookup table on CPU ... ')
    wlog(src_lookup_table)
    wlog(trg_lookup_table)
    '''

    sv = vocab_data['src'].idx2key
    tv = vocab_data['trg'].idx2key

    nmtModel = NMT()
    classifier = Classifier(wargs.out_size, trg_vocab_size)
    #print nmtModel

    if wargs.pre_train:

        pre_dict = tc.load(wargs.pre_train)
        pre_model_dict = pre_dict['model']
        pre_model_dict = {k: v for k, v in pre_model_dict.items() if 'classifier' not in k}

        nmtModel.load_state_dict(pre_model_dict)
        classifier.load_state_dict(pre_dict['class'])

        wlog('Loading pre-trained model from {} at epoch {} and batch {}'.format(
            wargs.pre_train, pre_dict['epoch'], pre_dict['batch']))

        wlog('Loading optimizer from {}'.format(wargs.pre_train))
        optim = pre_dict['optim']
        wlog(optim)

        #wargs.start_epoch = pre_dict['epoch'] + 1

    else:

        for p in nmtModel.parameters():
            #p.data.uniform_(-0.1, 0.1)
            if len(p.size()) == 2:
                if p.size(0) == 1 or p.size(1) == 1:
                    p.data.zero_()
                else:
                    p.data.normal_(0, 0.01)
            elif len(p.size()) == 1:
                p.data.zero_()

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

    '''
    nmtModel.src_lookup_table = src_lookup_table
    nmtModel.trg_lookup_table = trg_lookup_table
    print nmtModel.src_lookup_table.weight.data.is_cuda

    nmtModel.classifier.init_weights(nmtModel.trg_lookup_table)
    '''

    wlog(nmtModel)
    pcnt1 = len([p for p in nmtModel.parameters()])
    pcnt2 = sum([p.nelement() for p in nmtModel.parameters()])
    wlog('Parameters number: {}/{}'.format(pcnt1, pcnt2))

    optim.init_optimizer(nmtModel.parameters())

    #tor = Translator(nmtModel, sv, tv)
    #tor.trans_tests(tests_data, pre_dict['epoch'], pre_dict['batch'])

    trainer = Trainer(nmtModel, sv, tv, optim, trg_vocab_size)

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

    #if batch_dev is not None: trainer.train(batch_dev, name='dev')

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

        tor = Translator(bestModel, sv, tv)
        tor.trans_tests(tests_data, model_dict['epoch'], model_dict['batch'])





if __name__ == "__main__":

    main()















