## GSNMT: Greedy Search with Probabilistic N-gram Matching for Neural Machine Translation
Probabilistic sequence-level objectives is employed to alleviate the exposure bias and finetune the NMT model. We first pretrain the NMT model using cross-entropy loss and then finetune the model using probabilistic sequence-level objectives where greedy search is employed to alleviate the exposure bias.

> Chenze Shao, Yang Feng and Xilin Chen. Greedy Search with Probabilistic N-gram Matching for Neural Machine Translation. In Proceedings of Emnlp, 2018.

### Runtime Environment
This system has been tested in the following environment.
+ Ubuntu 16.04.1 LTS 64 bits
+ Python 2.7
+ Pytorch 0.3.0
+ Dependency
	+ download [Standford parser Version 3.8.0](https://nlp.stanford.edu/software/stanford-parser-full-2017-06-09.zip)
	+ unzip
	+ ``export CLASSPATH="./stanford-parser-full-2017-06-09/stanford-parser.jar:$CLASSPATH"``

### Toy Dataset
+ The training data consists of 44K sentences from the tourism and travel domain
+ Validation Set was composed of the ASR devset 1 and devset 2 from IWSLT 2005
+ Testing dataset is the IWSLT 2005 test set.

### Data Preparation
Name the file names of the datasets according to the variables in the ``wargs.py`` file  
Both sides of the training dataset and the source sides of the validation/test sets are tokenized by using the Standford tokenizer.

#### Training Dataset

+ **Source side**: ``dir_data + train_prefix + '.' + train_src_suffix``  
+ **Target side**: ``dir_data + train_prefix + '.' + train_trg_suffix``  

#### Validation Set

+ **Source side**: ``val_tst_dir + val_prefix + '.' + val_src_suffix``    
+ **Target side**:  
	+ One reference  
``val_tst_dir + val_prefix + '.' + val_ref_suffix``  
	+ multiple references  
``val_tst_dir + val_prefix + '.' + val_ref_suffix + '0'``  
``val_tst_dir + val_prefix + '.' + val_ref_suffix + '1'``  
``......``

#### Test Dataset
+ **Source side**: ``val_tst_dir + test_prefix + '.' + val_src_suffix``  
+ **Target side**:  
``for test_prefix in tests_prefix``
	+ One reference  
``val_tst_dir + test_prefix + '.' + val_ref_suffix``  
	+ multiple references  
``val_tst_dir + test_prefix + '.' + val_ref_suffix + '0'``  
``val_tst_dir + test_prefix + '.' + val_ref_suffix + '1'``  
``......``
 
### Pretraining
Pretrain the NMT model using cross-entropy loss.
run ``cp wargs_pretrain.py wargs.py&&python _main.py``

### Finetuning
Finetue the NMT model using P-P2 loss.
run ``cp wargs_finetune.py wargs.py&&python _main.py``

### Inference
Assume that the trained model is named ``best.model.pt``  
Before decoding, parameters about inference in the file ``wargs.py`` should be configured  
+ translate one sentence  
run ``python wtrans.py -m best.model.pt``
+ translate one file  
	+ put the test file to be translated into the path ``val_tst_dir + '/'``  
	+ run ``python wtrans.py -m best.model.pt -i test_prefix``







