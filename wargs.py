
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
#
enc_layer_cnt = 1

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
#
dec_layer_cnt = 1

drop_rate = 0.5

# Directory to save model, test output and validation output
dir_model = 'wmodel'
dir_valid = 'wvalid'
dir_tests = 'wtests'

# Validation data
val_shuffle = True
#val_tst_dir = '/home/wen/2.data/allnist_stanfordseg_jiujiu/'
#val_tst_dir = '/home/wen/3.corpus/allnist_stanfordseg_jiujiu/'
val_tst_dir = '/home5/wen/2.data/allnist_stanfordseg_jiujiu/'
val_prefix = 'nist02'

tests_prefix = ['nist03', 'nist04', 'nist05', 'nist06', 'nist08']

# Training data
train_shuffle = True
batch_size = 80
sort_k_batches = 20

# Data path
dir_data = 'data/'
train_src = dir_data + 'train.1k.zh'
train_trg = dir_data + 'train.1k.en'

# Dictionary
src_dict_size = 30000
trg_dict_size = 30000
src_dict = dir_data + 'src.dict.tcf'
trg_dict = dir_data + 'trg.dict.tcf'

inputs_data = dir_data + 'inputs.pt'

# Training
max_epochs = 20

epoch_shuffle = False
epoch_shuffle_minibatch = 1

small = False

display_freq = 10 if small else 1000
sampling_freq = 100 if small else 5000
sample_size = 5
if_fixed_sampling = False

epoch_eval = False
eval_valid_from = 10000 if small else 50000
eval_valid_freq = 5000 if small else 5000

save_one_model = True
start_epoch = 1
final_test = False

model_prefix = dir_model + '/model'
best_model = dir_valid + '/best.model.pt' if dir_valid else 'best.model.pt'
# pretrained model
pre_train = None
#pre_train = best_model

# decoder hype-parameters
search_mode = 1
ori_search = 0
beam_size = 10
with_batch = 0
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
opt_mode = 'adadelta'
learning_rate = 1.0
#opt_mode = 'adam'
#opt_mode = 'sgd'
max_grad_norm = 1.0
learning_rate_decay = 0.00001
# Start decaying every epoch after and including this epoch
last_valid_bleu = 0.
start_decay_from = None

max_gen_batches = 100

gpu_id = [0]
#gpu_id = None

#dec_gpu_id = [1]
#dec_gpu_id = None
file_tran_dir = 'wTrans'
