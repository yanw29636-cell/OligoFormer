import argparse
import os

# CUDA_VISIBLE_DEVICES must be set before torch initializes a CUDA context, which
# happens as a side effect of importing the modules below (they each call
# torch.cuda.is_available() at import time). Pre-parse just --cuda here so the
# right physical GPU is already selected by the time those imports run.
_cuda_pre_parser = argparse.ArgumentParser(add_help=False)
_cuda_pre_parser.add_argument('--cuda', type=int, default=0)
_cuda_args, _ = _cuda_pre_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_cuda_args.cuda)

from train_single import train_single
from test_single import test_single
from test import test
from train import train
from infer import infer
from mismatch import mismatch
import warnings
warnings.filterwarnings("ignore")
def main():
    parser = argparse.ArgumentParser(description='OligoFormer')
    # Data options
    parser.add_argument('--datasets', 		type=str, nargs='+', default=['Hu','Mix'], help="[Train,Test]: ['Hu','Mix','Taka']")
    parser.add_argument('--path',     		type=str, default='./data/', help='train or infer')
    parser.add_argument('--output_dir',  	type=str, default="result/", help='output directory')
    parser.add_argument('--best_model',  	type=str, default="./model/best_model.pth", help='best model directory')
    parser.add_argument('--resume',  	type=str, default=None, help='retrain on existing models')
    parser.add_argument('--cuda',     		type=int, default=0, help='ID number of GPU')
    parser.add_argument('--seed',     		type=int, default=42, help='random seed')


    # Training Hyper-parameters
    parser.add_argument('--learning_rate',  type=float, default=0.0001, help='learning rate') # 0.00001
    parser.add_argument('--batch_size',     type=int,   default=16, help='input batch size')
    parser.add_argument('--epoch',          type=int,   default=200, help='number of epochs to train')
    parser.add_argument('--early_stopping', type=int,   default=30, help='early stopping')
    parser.add_argument('--relabel_median', action='store_true', help='recompute the binary y label per-dataset via its own median(label) split instead of the fixed 0.7 threshold baked into the CSV; mitigates class-prior mismatch when train/val come from different datasets (train.py only)')
    parser.add_argument('--embedding_source', type=str, default='rnafm', choices=['rnafm', 'rinalmo'], help="pretrained RNA embedding to use: 'rnafm' (default, data/RNAFM/) or 'rinalmo' (RiNALMo-650M via multimolecule, data/RiNALMo/ — generate with scripts/RiNALMo-features.sh in a separate conda env). Train/test only; infer.py still only supports rnafm.")
    parser.add_argument('--weight_decay',   type=float, default=0.999, help='gamma for the ExponentialLR scheduler (per-epoch LR decay factor) -- NOT L2 regularization; Adam is created with no weight_decay')
    parser.add_argument('--l2_decay',   type=float, default=0.0, help='L2 regularization strength passed to Adam as weight_decay (default 0.0 matches prior behavior, i.e. no L2 penalty)')
    parser.add_argument('--kfold',    type=int, default=5, help='K-fold cross validation')

    # Model-selection methodology (train.py only). As of 2026-09-25 the defaults hold out part
    # of datasets[1] for selection and report on the rest; before that "val" and "test" were both
    # the whole of datasets[1], so checkpoint selection, early stopping and F1-threshold
    # calibration all happened on the reported rows and inflated rocauc by ~0.021. Pass
    # --val_split 0 --select_on all to reproduce a pre-2026-09-25 number.
    parser.add_argument('--val_split', type=float, default=0.4, help='fraction of datasets[1] held out as the model-selection (val) set; the rest becomes a test set that nothing in the selection path sees. Set to 0.0 for the pre-2026-09-25 legacy behaviour, val == test == all of datasets[1], which reports rocauc ~0.021 too high')
    parser.add_argument('--split_seed', type=int, default=0, help='RNG seed for the --val_split stratified split, deliberately separate from --seed so every seed/config in a comparison scores against the same test rows')
    parser.add_argument('--train_val_split', type=float, default=0.0, help='fraction of datasets[0] (the TRAINING set) held out as the model-selection set instead of taking it out of datasets[1]; leaves all of datasets[1] available as test rows, but necessarily shrinks the training set. Requires --val_split > 0')
    parser.add_argument('--select_on', type=str, default='val', choices=['all', 'val', 'trainval'], help="which rows drive checkpoint selection and early stopping when --val_split > 0: 'all' = all of datasets[1] (the leaky pre-2026-09-25 rule, kept for A/B comparison), 'val' = the held-out val part of datasets[1], 'trainval' = the held-out part of datasets[0] (requires --train_val_split)")
    parser.add_argument('--select_metric', type=str, default='loss_and_auc', choices=['loss_and_auc', 'rocauc'], help="model-selection rule: 'loss_and_auc' (legacy) requires val loss AND rocauc to both beat their running records in the same epoch; 'rocauc' selects on rocauc alone")
    parser.add_argument('--epoch_metrics_csv', type=str, default=None, help='write per-epoch metrics for every evaluated split to this CSV, so selection rules can be replayed post-hoc from one training run')
    parser.add_argument('--no_early_stop', action='store_true', help='ignore --early_stopping and run the full --epoch budget (use when logging a complete trajectory for post-hoc selection-rule comparison)')
    parser.add_argument('--no_checkpoint', action='store_true', help='do not write .pth checkpoints (trajectory-logging runs only need the CSV)')

    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=26, help='vocab size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding hidden dimension')
    parser.add_argument('--lstm_dim', type=int, default=32, help='lstm dimension')
    parser.add_argument('--n_head', type=int, default=8, help='the number of heads')
    parser.add_argument('--n_layers', type=int, default=1, help='the number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--lm', type=int, default=19, help='flanking length')
    parser.add_argument('--lm1', type=int, default=19, help='left length')
    parser.add_argument('--lm2', type=int, default=19, help='right length')
    parser.add_argument('--lstm_batch_first', action='store_true', help="construct the two nn.LSTMs with batch_first=True, so the recurrence runs along the 19/57 nucleotide positions instead of across the samples in a batch. Off by default because every existing checkpoint (model/best_model.pth included) was trained under the original convention and would be silently reinterpreted; tensor shapes are identical either way, so state_dicts load under both")

    # option parameter
    parser.add_argument('-t','--test',action='store_true', help='test mode')
    parser.add_argument('-s','--single',action='store_true', help='single dataset mode')  

    # Infer module
    parser.add_argument('-i','--infer', type=int, default=0, help='0: None; 1: infer by fasta; 2: infer manually')
    parser.add_argument('-top','--top_n', type=int, default=-1, help='-1: all siRNAs; n: only calculate top n siRNAs')
    parser.add_argument('-i1','--infer_fasta', type=str, default='./data/example.fa', help='fasta file to infer')
    parser.add_argument('-i2','--infer_siRNA_fasta', nargs='?', const=False, help='siRNA fasta file to infer')
    
    # Functionality module
    parser.add_argument('-nf', '--no_func', action='store_true', help='not use filter based on known criteria determinant to functionality')
    parser.add_argument('--gc_min', type=float, default=30, help='func_filter: minimum allowed GC%% (siRNA rejected below this)')
    parser.add_argument('--gc_max', type=float, default=65, help='func_filter: maximum allowed GC%% (siRNA rejected above this)')
    parser.add_argument('--homopolymer_len', type=int, default=5, help='func_filter: reject any run of this many (or more) identical bases')
    parser.add_argument('--gc_run_len', type=int, default=6, help='func_filter: reject any run of this many (or more) consecutive G/C bases')
    parser.add_argument('--cds_start', type=int, default=None, help='1-based CDS start coordinate within the -i1 mRNA sequence; when given together with --cds_end, adds a region (5UTR/CDS/3UTR) column')
    parser.add_argument('--cds_end', type=int, default=None, help='1-based CDS end coordinate within the -i1 mRNA sequence; when given together with --cds_start, adds a region (5UTR/CDS/3UTR) column')
    parser.add_argument('--exclude_5utr', action='store_true', help='hard-filter out candidates classified as 5UTR (requires --cds_start/--cds_end)')
    
    # Off-target module
    parser.add_argument('-off','--off_target', action='store_true', help='evaluate off-target')
    parser.add_argument('-a','--all_human', action='store_true', help='scan all human UTR and ORF')
    parser.add_argument('--utr', type=str, default='./off-target/ref/100_human_UTR.txt', help='reference UTR sequence')
    parser.add_argument('--orf', type=str, default='./off-target/ref/100_human_ORF.txt', help='reference ORF sequence')
    parser.add_argument('--pita_threshold', type=float, default=-10, help='pita filter threshold')
    parser.add_argument('--targetscan_threshold', type=float, default=1, help='targetscan filter threshold')
    
    # Toxicity module
    parser.add_argument('-tox','--toxicity', action='store_true', help='evaluate toxicity')
    parser.add_argument('--toxicity_threshold', type=float, default=50.0, help='toxicity filter threshold')
    
    # Mismatch module
    parser.add_argument('-m','--mismatch', type=int, default=0, help='0: None; 1: compare two RNAs; 2: tranverse with 1,2 mismatches; 3: tranverse with 1,2,3 mismatches')

    Args = parser.parse_args()
    os.makedirs(Args.output_dir, exist_ok=True)

    if Args.mismatch > 0:
        mismatch(Args)
    elif Args.infer > 0:
        if Args.embedding_source == 'rinalmo':
            raise ValueError("--embedding_source rinalmo is not supported for --infer: infer.py has no on-the-fly RiNALMo embedding extraction for arbitrary input sequences (RiNALMo-features.sh only precomputes embeddings for the fixed train/test datasets). Use a checkpoint trained with --embedding_source rnafm (the default) for inference.")
        infer(Args)
    elif Args.test:
        test_single(Args) if Args.single else test(Args)
    else:
        train_single(Args) if Args.single else train(Args)

if __name__ == '__main__':
    main()

