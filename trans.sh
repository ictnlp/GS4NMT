#THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=5000 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 10 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 20 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 50 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 100 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 500 1
python wtrans.py \
    --model-file $1 \
    --vocab-data $2 \
    --valid-data $3 \
