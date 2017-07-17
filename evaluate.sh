#THEANO_FLAGS=device=cpu,floatX=float32,lib.cnmem=500, python2.7 $1
#rm -rf models/* valids/* tests/*
#export LD_LIBRARY_PATH="/usr/local/apps/cuda/cuda-7.5.18/lib64"

echo 'output file:   '.${1} >> ${3}
#tstvaldir='/scratch2/wzhang/3.corpus/2.mt/nist-all/allnist_stanfordseg_jiujiu/'
tstvaldir=${4}
val_prefix=${tstvaldir}${2}
val_src_sgm=${val_prefix}.src.sgm
val_ref4=${val_prefix}.ref
val_oriref_sgm=${val_prefix}.low.ref.sgm
echo 'source sgm:    '.${val_src_sgm} >> ${3}
echo 'four refers:   '.${val_ref4} >> ${3}
echo 'ori refer sgm: '.${val_oriref_sgm} >> ${3}

# bpe
#cp $1 $1.bpe
#sed -i "s/@@ //g" $1
#sed -i 's/(@@ )|(@@ ?$)//g' $1
# drop unk first
bash rm_unk.sh $1

detokenizer.perl -l en < $1.nounk > $1.nounk.detok
plain2sgm tst $1.nounk.detok > $1.nounk.detok.res.sgm
mteval-v11b.pl -s ${val_src_sgm} -r ${val_oriref_sgm} -t $1.nounk.detok.res.sgm >> ${3}

# no post-processing
multi-bleu.perl -lc ${val_ref4} < $1.nounk >> ${3}

rm $1.nounk.detok.res.sgm
