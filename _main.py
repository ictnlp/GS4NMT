import torch as tc
from torch import cuda

import wargs
from tools.inputs import Input
from tools.utils import init_dir, wlog, load_pytorch_model
from tools.optimizer import Optim
from inputs_handler import *

# Check if CUDA is available
if cuda.is_available():
    wlog('CUDA is available, specify device by gpu_id argument (i.e. gpu_id=[3])')
else:
    wlog('Warning: CUDA is not available, train on CPU')

if wargs.gpu_id:
    cuda.set_device(wargs.gpu_id[0])
    wlog('Using GPU {}'.format(wargs.gpu_id[0]))

if wargs.model == 0: from models.groundhog import *
elif wargs.model == 1: from models.rnnsearch import *
elif wargs.model == 2: from models.rnnsearch_ia import *
elif wargs.model == 3: from models.ran_agru import *
elif wargs.model == 4: from models.rnnsearch_rn import *
elif wargs.model == 5: from models.nmt_sru import *
elif wargs.model == 6: from models.nmt_cyk import *
elif wargs.model == 7: from models.non_local import *
from models.losser import *

from trainer import *
from translate import Translator

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

def main():

    init_dir(wargs.dir_model)
    init_dir(wargs.dir_valid)

    vocab_data = {}
    train_srcD_file = wargs.src_vocab_from
    wlog('\nPreparing source vocabulary from {} ... '.format(train_srcD_file))
    src_vocab = extract_vocab(train_srcD_file, wargs.src_dict, wargs.src_dict_size)
    vocab_data['src'] = src_vocab

    train_trgD_file = wargs.trg_vocab_from
    wlog('\nPreparing target vocabulary from {} ... '.format(train_trgD_file))
    trg_vocab = extract_vocab(train_trgD_file, wargs.trg_dict, wargs.trg_dict_size)
    vocab_data['trg'] = trg_vocab

    train_src_file = wargs.train_src
    train_trg_file = wargs.train_trg
    wlog('\nPreparing training set from {} and {} ... '.format(train_src_file, train_trg_file))
    train_src_tlst, train_trg_tlst = wrap_data(train_src_file, train_trg_file,
                                               src_vocab, trg_vocab, max_seq_len=wargs.max_seq_len)
    '''
    list [torch.LongTensor (sentence), torch.LongTensor, torch.LongTensor, ...]
    no padding
    '''

    '''
    devs = {}
    dev_src = wargs.val_tst_dir + wargs.val_prefix + '.src'
    dev_trg = wargs.val_tst_dir + wargs.val_prefix + '.ref0'
    wlog('\nPreparing dev set for tuning from {} and {} ... '.format(dev_src, dev_trg))
    dev_src, dev_trg = wrap_data(dev_src, dev_trg, src_vocab, trg_vocab)
    devs['src'], devs['trg'] = dev_src, dev_trg
    '''

    valid_file = '{}{}.{}'.format(wargs.val_tst_dir, wargs.val_prefix, wargs.val_src_suffix)
    wlog('\nPreparing validation set from {} ... '.format(valid_file))
    valid_src_tlst, valid_src_lens = val_wrap_data(valid_file, src_vocab)

    wlog('Sentence-pairs count in training data: {}'.format(len(train_src_tlst)))
    src_vocab_size, trg_vocab_size = vocab_data['src'].size(), vocab_data['trg'].size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(src_vocab_size, trg_vocab_size))

    batch_train = Input(train_src_tlst, train_trg_tlst, wargs.batch_size)
    batch_valid = Input(valid_src_tlst, None, 1, volatile=True)

    tests_data = None
    if wargs.tests_prefix is not None:
        init_dir(wargs.dir_tests)
        tests_data = {}
        for prefix in wargs.tests_prefix:
            init_dir(wargs.dir_tests + '/' + prefix)
            test_file = '{}{}.{}'.format(wargs.val_tst_dir, prefix, wargs.val_src_suffix)
            wlog('Preparing test set from {} ... '.format(test_file))
            test_src_tlst, _ = val_wrap_data(test_file, src_vocab)
            tests_data[prefix] = Input(test_src_tlst, None, 1, volatile=True)

    '''
    # lookup_table on cpu to save memory
    src_lookup_table = nn.Embedding(wargs.src_dict_size + 4,
                                    wargs.src_wemb_size, padding_idx=utils.PAD).cpu()
    trg_lookup_table = nn.Embedding(wargs.trg_dict_size + 4,
                                    wargs.trg_wemb_size, padding_idx=utils.PAD).cpu()

    wlog('Lookup table on CPU ... ')
    wlog(src_lookup_table)
    wlog(trg_lookup_table)
    '''

    sv = vocab_data['src'].idx2key
    tv = vocab_data['trg'].idx2key

    nmtModel = NMT(src_vocab_size, trg_vocab_size)
    classifier = Classifier(wargs.out_size, trg_vocab_size,
                            nmtModel.decoder.trg_lookup_table if wargs.copy_trg_emb is True else None)

    if wargs.pre_train:

        assert os.path.exists(wargs.pre_train)
        model_dict, class_dict, eid, bid, optim = load_pytorch_model(wargs.pre_train)
        # initializing parameters of interactive attention model
        for name, param in nmtModel.named_parameters():
            if name in model_dict:
                param.requires_grad = False
                param.data = model_dict[name]
                wlog('Model \t {}'.format(name))
            else: init_params(param, name, True)

        for name, param in classifier.named_parameters():
            if name in class_dict:
                param.requires_grad = False
                param.data = class_dict[name]
                wlog('Model \t {}'.format(name))
            else: init_params(param, name, True)

        wargs.start_epoch = eid + 1

        #tor = Translator(nmtModel, sv, tv)
        #tor.trans_tests(tests_data, eid, bid)

    else:
        for n, p in nmtModel.named_parameters(): init_params(p, n, True)
        for n, p in classifier.named_parameters(): init_params(p, n, True)
        optim = Optim(
            wargs.opt_mode, wargs.learning_rate, wargs.max_grad_norm,
            learning_rate_decay=wargs.learning_rate_decay,
            start_decay_from=wargs.start_decay_from,
            last_valid_bleu=wargs.last_valid_bleu
        )

    if wargs.gpu_id:
        nmtModel.cuda()
        classifier.cuda()
        wlog('Push model onto GPU[{}] ... '.format(wargs.gpu_id[0]))
    else:
        nmtModel.cpu()
        classifier.cpu()
        wlog('Push model onto CPU ... ')

    nmtModel.classifier = classifier
    nmtModel.decoder.map_vocab = classifier.map_vocab

    '''
    nmtModel.src_lookup_table = src_lookup_table
    nmtModel.trg_lookup_table = trg_lookup_table
    print nmtModel.src_lookup_table.weight.data.is_cuda

    nmtModel.classifier.init_weights(nmtModel.trg_lookup_table)
    '''

    wlog(nmtModel)
    wlog(optim)
    pcnt1 = len([p for p in nmtModel.parameters()])
    pcnt2 = sum([p.nelement() for p in nmtModel.parameters()])
    wlog('Parameters number: {}/{}'.format(pcnt1, pcnt2))

    optim.init_optimizer(nmtModel.parameters())

    #tor = Translator(nmtModel, sv, tv, wargs.search_mode)
    #tor.trans_tests(tests_data, pre_dict['epoch'], pre_dict['batch'])

    trainer = Trainer(nmtModel, batch_train, vocab_data, optim, batch_valid, tests_data)

    trainer.train()


if __name__ == "__main__":

    main()















