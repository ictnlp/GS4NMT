from utils import *
import wargs

if __name__ == "__main__":

    valid_out = 'test/trans_e15_upd15000_b10m2_bch1'
    print Calc_BLEU(valid_out, wargs.val_tst_dir, wargs.val_prefix)
