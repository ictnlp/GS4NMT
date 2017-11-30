# wpynmt

1. update on 2017.10.18

	a. update the file structures, different models corresponding to different searchers;
	b. fix some other small bugs;
	c. add some tricks:
			such as copy trg vocab weight;
			fit segmentation for source sentence;
			add layer norm for gru.py;
			with bpe support, but need to preprocess by open-source bpe tools;
			add rn model and sru model;


# translate

python wtrans.py --model-file wmodels.1m.rnnsearch/best.model.pt --test-file 900


# evaluate alignment

score-alignments.py -d /home5/wen/2.data/mt/900_alignment/900 -s zh -t en -g wa -i trans_900_e13_upd10000_b10m1_bch1.aln

