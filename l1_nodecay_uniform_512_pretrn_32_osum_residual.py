import wargs
import torch as tc
from torch import cuda
from inputs import Input
from utils import init_dir, wlog, load_pytorch_model
from optimizer import Optim
from train import *
import const

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

#from model_groundhog import *
#from model_rnnsearch import *
#from model_ran import *
from model_rnnsearch_rn import *
from translate import Translator

def main():

    # Check if CUDA is available
    if cuda.is_available():
        wlog('CUDA is available, specify device by gpu_id argument (i.e. gpu_id=[3])')
    else:
        wlog('Warning: CUDA is not available, train CPU')

    if wargs.gpu_id:
        cuda.set_device(wargs.gpu_id[0])
        wlog('Using GPU {}'.format(wargs.gpu_id[0]))

    init_dir(wargs.dir_model)
    init_dir(wargs.dir_valid)
    init_dir(wargs.dir_tests)
    for prefix in wargs.tests_prefix: init_dir(wargs.dir_tests + '/' + prefix)

    wlog('Loading data ... ', 0)

    inputs = tc.load(wargs.inputs_data)

    vocab_data, train_data, valid_data = inputs['vocab'], inputs['train'], inputs['valid']

    train_src_tlst, train_trg_tlst = train_data['src'], train_data['trg']
    valid_src_tlst, valid_src_lens = valid_data['src'], valid_data['len']

    wlog('Sentence-pairs count in training data: {}'.format(len(train_src_tlst)))
    src_vocab_size, trg_vocab_size = vocab_data['src'].size(), vocab_data['trg'].size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(src_vocab_size, trg_vocab_size))

    batch_train = Input(train_src_tlst, train_trg_tlst, wargs.batch_size)
    batch_valid = Input(valid_src_tlst, None, 1, volatile=True)

    tests_data = None
    if inputs.has_key('tests'):
        tests_data = {}
        tests_tensor = inputs['tests']
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

    if wargs.pre_train:

        model_dict, class_dict, eid, bid, optim = load_pytorch_model(wargs.pre_train)

        # initializing parameters of interactive attention model
        for name, param in nmtModel.named_parameters():
            if name in model_dict:
                #if name.startswith('decoder') and not name == 'decoder.trg_lookup_table.weight':
                #    init_params(param, name, False)
                #    model_dict[name] = param
                #else:
                param.data = model_dict[name]
                wlog('Model \t {}'.format(name))
            else: init_params(param, name, True)

        '''
        for p in nmtModel.named_parameters():
            if p[0] in model_dict: p[1].data = model_dict[p[0]]
            else: init_params(p[1], False)
        '''
        for n, p in classifier.named_parameters():
            if n in class_dict:
                p.data = class_dict[n]
                wlog('Model \t {}'.format(n))
            else: init_params(p, n, True)

        wargs.start_epoch = eid + 1

    else:
        for n, p in nmtModel.named_parameters(): init_params(p, n, False)
        for n, p in classifier.named_parameters(): init_params(p, n, False)
        optim = Optim(
            wargs.opt_mode, wargs.learning_rate, wargs.max_grad_norm,
            learning_rate_decay=wargs.learning_rate_decay,
            start_decay_from=wargs.start_decay_from,
            last_valid_bleu=wargs.last_valid_bleu
        )

    if wargs.gpu_id:
        nmtModel.cuda()
        classifier.cuda()
        wlog('push model onto GPU[{}] ... '.format(wargs.gpu_id[0]))
    else:
        nmtModel.cpu()
        classifier.cuda()
        wlog('Push model onto CPU ... ')

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

    #tor = Translator(nmtModel, sv, tv, wargs.search_mode)
    #tor.trans_tests(tests_data, pre_dict['epoch'], pre_dict['batch'])

    train(nmtModel, batch_train, batch_valid, tests_data, vocab_data, optim)

    if tests_data and wargs.final_test:

        assert os.path.exists(wargs.best_model)

        best_model_dict, best_class_dict, eid, bid, optim = load_pytorch_model(wargs.best_model)

        nmtModel.load_state_dict(best_model_dict)
        classifier.load_state_dic(best_class_dict)
        nmtModel.classifier = classifier

        tor = Translator(nmtModel, sv, tv, wargs.search_mode)
        tor.trans_tests(tests_data, eid, bid)




if __name__ == "__main__":

    main()















