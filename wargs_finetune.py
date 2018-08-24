loss_learning_rate = 0.01
loss = "sen_p2_loss"
feed_previous = True
mix_loss = None
alpha = None

volatile = False
log_norm = False


# Maximal sequence length in training data
max_seq_len = 50

'''
Embedding layer
'''
# Size of word embedding of source word and target word
src_wemb_size = 256
trg_wemb_size = 256

'''
Encoder layer
'''
# Size of hidden units in encoder
enc_hid_size = 256

'''
Attention layer
'''
# Size of alignment vector
align_size = 256

'''
Decoder layer
'''
# Size of hidden units in decoder
dec_hid_size = 256
# Size of the output vector
out_size = 256

drop_rate = 0.5

# Directory to save model, test output and validation output
dir_model = 'wmodel'
dir_valid = 'wvalid'
dir_tests = 'wtests'

# Validation data
val_shuffle = True
#val_tst_dir = '/home5/wen/2.data/mt/nist_data_stanseg/'
#val_tst_dir = '/home/wen/3.corpus/wmt2017/de-en/'
val_tst_dir = './data/devtest/'

#val_prefix = 'nist02'
val_prefix = 'devset1_2.lc'
#val_prefix = 'newstest2014.tc'
#val_src_suffix = 'src'
#val_ref_suffix = 'ref.plain_'
val_src_suffix = 'zh'
val_ref_suffix = 'en'
#val_src_suffix = 'en'
#val_ref_suffix = 'de'
ref_cnt = 16

#tests_prefix = None
#tests_prefix = ['nist03', 'nist04', 'nist05', 'nist06', 'nist08']
#tests_prefix = ['devset3.lc', '900']
tests_prefix = ['devset3.lc']
#tests_prefix = ['newstest2015.tc', 'newstest2016.tc', 'newstest2017.tc']

# Training data
train_shuffle = True
batch_size = 40
sort_k_batches = 10

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
max_epochs = 50

epoch_shuffle = False
epoch_shuffle_minibatch = 1

small = True

display_freq = 10 if small else 1000
sampling_freq = 100 if small else 5000
sample_size = 3
if_fixed_sampling = False

epoch_eval = True
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
fix_pre_params = False

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

with_bpe = False
with_postproc = False
retok = False
copy_trg_emb = False

print_att = True
self_norm_alpha = None

#dec_gpu_id = [1]
#dec_gpu_id = None
gpu_id = [0]
#gpu_id = None


