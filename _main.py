import torch as tc
from torch import cuda

import wargs
from tools.inputs import Input
from tools.utils import init_dir, wlog, _load_model
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

from models.groundhog import *
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

    sv = vocab_data['src'].idx2key
    tv = vocab_data['trg'].idx2key

    nmtModel = NMT(src_vocab_size, trg_vocab_size)

    if wargs.pre_train is not None:

        assert os.path.exists(wargs.pre_train), 'Requires pre-trained model'
        wlog('load model from {} ...'.format(wargs.pre_train))
        _dict = _load_model(wargs.pre_train)
        # initializing parameters of interactive attention model
        class_dict = None
        if len(_dict) == 4: model_dict, eid, bid, optim = _dict
        elif len(_dict) == 5:
            model_dict, class_dict, eid, bid, optim = _dict
        for name, param in nmtModel.named_parameters():
            if name in model_dict:
                param.requires_grad = not wargs.fix_pre_params
                param.data.copy_(model_dict[name])
                wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            elif name.endswith('map_vocab.weight'):
                if class_dict is not None:
                    param.requires_grad = not wargs.fix_pre_params
                    param.data.copy_(class_dict['map_vocab.weight'])
                    wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            elif name.endswith('map_vocab.bias'):
                if class_dict is not None:
                    param.requires_grad = not wargs.fix_pre_params
                    param.data.copy_(class_dict['map_vocab.bias'])
                    wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            else: init_params(param, name, True)

        wargs.start_epoch = eid + 1
    else:
        for n, p in nmtModel.named_parameters(): init_params(p, n, True)
        optim = Optim(
            wargs.opt_mode, wargs.learning_rate, wargs.max_grad_norm,
            learning_rate_decay=wargs.learning_rate_decay,
            start_decay_from=wargs.start_decay_from,
            last_valid_bleu=wargs.last_valid_bleu
        )

    if wargs.gpu_id:
        nmtModel.cuda()
        wlog('Push model onto GPU[{}] ... '.format(wargs.gpu_id[0]))
    else:
        nmtModel.cpu()
        wlog('Push model onto CPU ... ')

    wlog(nmtModel)
    wlog(optim)
    pcnt1 = len([p for p in nmtModel.parameters()])
    pcnt2 = sum([p.nelement() for p in nmtModel.parameters()])
    wlog('Parameters number: {}/{}'.format(pcnt1, pcnt2))

    optim.init_optimizer(nmtModel.parameters())

    trainer = Trainer(nmtModel, batch_train, vocab_data, optim, batch_valid, tests_data)
    trainer.train()


if __name__ == "__main__":

    main()















