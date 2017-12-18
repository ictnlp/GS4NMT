
volatile = False
log_norm = False


# Maximal sequence length in training data
max_seq_len = 50

'''
Embedding layer
'''
# Size of word embedding of source word and target word
src_wemb_size = 512
trg_wemb_size = 512

'''
Encoder layer
'''
# Size of hidden units in encoder
enc_hid_size = 512

'''
Attention layer
'''
# Size of alignment vector
align_size = 512

'''
Decoder layer
'''
# Size of hidden units in decoder
dec_hid_size = 512
# Size of the output vector
out_size = 512

drop_rate = 0.5

# Directory to save model, test output and validation output
dir_model = 'wmodel'
dir_valid = 'wvalid'
dir_tests = 'wtests'

# Validation data
val_shuffle = True
#val_tst_dir = '/home/wen/3.corpus/allnist_stanfordseg_jiujiu/'
#val_tst_dir = '/home5/wen/2.data/allnist_stanseg/'
#val_tst_dir = '/home5/wen/2.data/segment_allnist_stanseg/'
#val_tst_dir = '/home5/wen/2.data/segment_allnist_stanseg_low/'
#val_tst_dir = '/home5/wen/2.data/mt/nist_data_stanseg/'
val_tst_dir = '/home/wen/3.corpus/mt/nist_data_stanseg/'
#val_tst_dir = '/home/wen/3.corpus/segment_allnist_stanseg/'
#val_tst_dir = '/home/wen/3.corpus/wmt2017/de-en/'
#val_tst_dir = './data/'

#val_prefix = 'wmt17.dev'
val_prefix = 'nist02'
#val_prefix = 'devset1_2.lc'
#val_prefix = 'newstest2014.tc'
val_src_suffix = 'src'
val_ref_suffix = 'ref.plain_'
#val_src_suffix = 'zh'
#val_ref_suffix = 'en'
#val_src_suffix = 'en'
#val_ref_suffix = 'de'
ref_cnt = 4

#tests_prefix = ['nist02', 'nist03', 'nist04', 'nist05', 'nist06', 'nist08', 'wmt17.tst']
tests_prefix = ['nist03', 'nist04', 'nist05', 'nist06', 'nist08']
#tests_prefix = ['data2', 'data3', 'test']
#tests_prefix = ['devset3.lc', '900']
#tests_prefix = ['devset3.lc']
#tests_prefix = ['newstest2015.tc', 'newstest2016.tc', 'newstest2017.tc']
#tests_prefix = None

# Training data
train_shuffle = True
batch_size = 80
sort_k_batches = 20

# Data path
dir_data = 'data/'
train_src = dir_data + 'train.src'
train_trg = dir_data + 'train.trg'

# Dictionary
src_vocab_from = train_src
trg_vocab_from = train_trg
src_dict_size = 30000
trg_dict_size = 30000
src_dict = dir_data + 'src.dict.tcf'
trg_dict = dir_data + 'trg.dict.tcf'

inputs_data = dir_data + 'inputs.pt'

# Training
max_epochs = 30

epoch_shuffle = False
epoch_shuffle_minibatch = 1

small = False

display_freq = 10 if small else 1000
sampling_freq = 100 if small else 5000
sample_size = 5
if_fixed_sampling = False

epoch_eval = False
final_test = False
eval_valid_from = 20 if small else 50000
eval_valid_freq = 50 if small else 20000

save_one_model = True
start_epoch = 1

model_prefix = dir_model + '/model'
best_model = dir_valid + '/best.model.pt' if dir_valid else 'best.model.pt'
# pretrained model
#pre_train = None
pre_train = best_model
fix_pre_params = True

# decoder hype-parameters
search_mode = 1
with_batch = 1
ori_search = 0
beam_size = 10
vocab_norm = 1
len_norm = 1
with_mv = 0
merge_way = 'Y'
avg_att = 0
m_threshold = 100.
ngram = 3
length_norm = 0.
cover_penalty = 0.

# optimizer

'''
Starting learning rate. If adagrad/adadelta/adam is used, then this is the global learning rate.
Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001
'''
opt_mode = 'adadelta'
learning_rate = 1.0

#opt_mode = 'adam'
#learning_rate = 1e-3

#opt_mode = 'sgd'
#learning_rate = 1.

max_grad_norm = 1.0

# Start decaying every epoch after and including this epoch
start_decay_from = None
learning_rate_decay = 0.5
last_valid_bleu = 0.

snip_size = 10
file_tran_dir = 'wexp-gpu-nist03'
laynorm = False
segments = False
seg_val_tst_dir = 'orule_1.7'

# model
enc_rnn_type = 'sru'    # rnn, gru, lstm, sru
enc_layer_cnt = 4
dec_rnn_type = 'sru'    # rnn, gru, lstm, sru
dec_layer_cnt = 4

with_bpe = False
with_postproc = True
copy_trg_emb = False

# 0: groundhog, 1: rnnsearch, 2: ia, 3: ran, 4: rn, 5: sru, 6: cyknet
model = 4

# convolutional layer
#filter_window_size = [1, 3, 5]   # windows size
filter_window_size = [1]   # windows size
#filter_feats_size = [32, 64, 96]
filter_feats_size = [96]
mlp_size = 256

# generate BTG tree when decoding
dynamic_cyk_decoding = False
print_att = True

# Scheduled Sampling of Samy bengio's paper
ss_type = 1     # 1: linear decay, 2: exponential decay, 3: inverse sigmoid decay
ss_eps_begin = 1   # set None for no scheduled sampling
ss_eps_end = 1
#ss_decay_rate = 0.005
ss_decay_rate = (ss_eps_begin - ss_eps_end) / 10.
ss_k = 0.98     # k < 1 for exponential decay, k >= 1 for inverse sigmoid decay

# free parameter for self-normalization
# 0 is equivalent to the standard neural network objective function.
self_norm_alpha = None

nonlocal_mode = 'dot'  # gaussian, dot, embeddedGaussian
#dec_gpu_id = [1]
#dec_gpu_id = None
gpu_id = [0]
#gpu_id = None


