"""Replay model-selection rules post-hoc against per-epoch trajectories.

`train.py --val_split ... --epoch_metrics_csv ...` logs, for every epoch, the metrics on
three row sets: the whole of datasets[1] ('full'), the held-out selection part ('val') and
the held-out evaluation part ('test'). Because all three come from one training run, every
selection rule below is scored on the *same* trajectory and the *same* test rows, so the
comparison is paired: training noise cancels out and only the selection rule differs.

Arms:
  A legacy    -- select on 'full' (= val+test) with the loss-AND-rocauc conjunction. This is
                 what train.py does today, and the rows it selects on are the rows it reports.
  B split     -- same conjunction rule, but selecting on 'val' only.
  C split+auc -- select on 'val' by rocauc alone (drops the conjunction).

Reported per arm:
  reported  -- the number the pipeline would print (arm A prints its selection set's score,
               which is the historically quoted figure; B/C print the honest test score)
  test      -- test-set score at the selected epoch, threshold carried over from val
  oracle    -- best test score anywhere in the trajectory, as an upper bound

Usage:
  python scripts/compare_selection.py run_dir_or_csv [more.csv ...] [--patience 100]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ARMS = {
    'A_legacy':    dict(select='full', metric='loss_and_auc'),
    'B_split':     dict(select='val',  metric='loss_and_auc'),
    'C_split_auc': dict(select='val',  metric='rocauc'),
}

# Only present in runs made with --train_val_split, where the selection set is carved out of
# the TRAINING dataset instead of the test dataset. Such a run trains on less data, so D/D0 are
# comparable to each other (same run) but not directly to A/B/C from a full-training-set run;
# D0 is the bridge, being B's rule evaluated under D's reduced training set.
TRAINVAL_ARMS = {
    'D_trainval':  dict(select='trainval', metric='loss_and_auc'),
    'D0_split':    dict(select='val',      metric='loss_and_auc'),
}


def select_epoch(df, select, metric, patience):
    """Replay train.py's selection + early-stopping loop, returning the chosen epoch.

    Mirrors the live code exactly: both running records are monotone, so under
    'loss_and_auc' an epoch must beat the best-ever loss AND the best-ever rocauc in the
    same epoch to be kept, and early stopping fires once `patience` epochs pass with no
    such epoch.
    """
    loss = df[f'{select}_loss'].values
    auc = df[f'{select}_rocauc'].values
    best_auc, best_loss, best_epoch = 0.0, 1e10, 0
    for i in range(len(df)):
        if metric == 'rocauc':
            improved = auc[i] > best_auc
        else:
            improved = loss[i] < best_loss and auc[i] > best_auc
        if improved:
            best_auc, best_loss, best_epoch = auc[i], loss[i], i
        if patience is not None and i - best_epoch > patience:
            break
    return best_epoch


def analyse(df, patience, arms):
    out = {}
    for arm, cfg in arms.items():
        e = select_epoch(df, cfg['select'], cfg['metric'], patience)
        row = df.iloc[e]
        out[arm] = {
            'epoch': int(row['epoch']),
            'reported_rocauc': row[f"{cfg['select']}_rocauc"],
            'test_rocauc': row['test_rocauc'],
            'test_prauc': row['test_prauc'],
            'test_f1': row['test_f1'],
            'test_mcc': row['test_mcc'],
            # what arm A's epoch would report if scored on test with a test-calibrated
            # threshold -- isolates threshold optimism from selection optimism
            'testself_f1': row['testself_f1'],
        }
    out['oracle'] = {
        'epoch': int(df.iloc[int(np.argmax(df['test_rocauc'].values))]['epoch']),
        'reported_rocauc': np.nan,
        'test_rocauc': df['test_rocauc'].max(),
        'test_prauc': np.nan, 'test_f1': np.nan, 'test_mcc': np.nan, 'testself_f1': np.nan,
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='per-epoch CSVs, or a directory containing them')
    ap.add_argument('--patience', type=int, default=100,
                    help='early-stopping patience to simulate (default 100, the production recipe); -1 disables')
    args = ap.parse_args()

    paths = []
    for p in args.inputs:
        paths.extend(sorted(glob.glob(os.path.join(p, '*.csv'))) if os.path.isdir(p) else [p])
    if not paths:
        sys.exit('no CSVs found')
    patience = None if args.patience < 0 else args.patience

    per_seed = {}
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in ('full_rocauc', 'val_rocauc', 'test_rocauc') if c not in df.columns]
        if missing:
            sys.exit(f'{p}: missing {missing} -- was it produced with --val_split > 0?')
        seed = int(df['seed'].iloc[0]) if 'seed' in df.columns else os.path.basename(p)
        arms = dict(ARMS)
        if 'trainval_rocauc' in df.columns:
            arms.update(TRAINVAL_ARMS)
        per_seed[seed] = analyse(df, patience, arms)
        arm_names = list(arms)
        print(f'{os.path.basename(p)}: seed={seed} epochs={len(df)}')

    seeds = sorted(per_seed)
    print(f'\npatience={args.patience}   seeds={seeds}   n={len(seeds)}\n')

    hdr = f"{'arm':<14}{'epoch':>14}{'reported':>18}{'test_rocauc':>18}{'test_prauc':>16}{'test_f1':>16}"
    print(hdr)
    print('-' * len(hdr))
    for arm in arm_names + ['oracle']:
        def agg(key):
            v = np.array([per_seed[s][arm][key] for s in seeds], dtype=float)
            if np.all(np.isnan(v)):
                return '--'
            return f'{np.nanmean(v):.4f}±{np.nanstd(v):.4f}'
        ep = np.array([per_seed[s][arm]['epoch'] for s in seeds], dtype=float)
        print(f"{arm:<14}{f'{ep.mean():.1f}±{ep.std():.1f}':>14}{agg('reported_rocauc'):>18}"
              f"{agg('test_rocauc'):>18}{agg('test_prauc'):>16}{agg('test_f1'):>16}")

    def mean_test(arm):
        return float(np.mean([per_seed[s][arm]['test_rocauc'] for s in seeds]))

    if 'D_trainval' in arm_names:
        print('\nselection-set comparison (test_rocauc, mean over seeds; same reduced training set):')
        print(f'  D0 select on val part of datasets[1] ({"187 rows"})          : {mean_test("D0_split"):.4f}')
        print(f'  D  select on held-out part of datasets[0] (training set)  : {mean_test("D_trainval"):.4f}'
              f'   [D-D0: {mean_test("D_trainval") - mean_test("D0_split"):+.4f}]')
        print('  (both report on the same held-out test rows; compare D0 against B_split from a')
        print('   full-training-set run to price the training-data the train-side split costs)')
        print('\npaired per-seed deltas (test_rocauc):')
        print(f"  {'seed':<8}{'D0_split':>12}{'D_trainval':>14}{'D-D0':>10}")
        for s in seeds:
            d0, d = per_seed[s]['D0_split']['test_rocauc'], per_seed[s]['D_trainval']['test_rocauc']
            print(f'  {s:<8}{d0:>12.4f}{d:>14.4f}{d - d0:>+10.4f}')
        return

    a_rep = float(np.mean([per_seed[s]['A_legacy']['reported_rocauc'] for s in seeds]))
    print('\ndecomposition of the legacy number (rocauc, mean over seeds):')
    print(f'  legacy reported (select on all of datasets[1], report on the same rows) : {a_rep:.4f}')
    print(f'  same epoch, scored on held-out test rows instead                        : {mean_test("A_legacy"):.4f}'
          f'   [evaluation leak: {a_rep - mean_test("A_legacy"):+.4f}]')
    print(f'  honest selection (val only) + held-out test                             : {mean_test("B_split"):.4f}'
          f'   [selection leak: {mean_test("A_legacy") - mean_test("B_split"):+.4f}]')
    print(f'  honest selection by rocauc alone + held-out test                        : {mean_test("C_split_auc"):.4f}'
          f'   [rule change vs B: {mean_test("C_split_auc") - mean_test("B_split"):+.4f}]')
    print(f'  oracle (best test epoch, unreachable)                                   : {mean_test("oracle"):.4f}')

    # per-seed paired deltas: the point of reusing one trajectory per seed is that these are
    # paired, so a consistent sign across seeds is meaningful even with few seeds
    print('\npaired per-seed deltas (test_rocauc):')
    print(f"  {'seed':<8}{'A_legacy':>12}{'B_split':>12}{'C_split_auc':>14}{'B-A':>10}{'C-B':>10}")
    for s in seeds:
        a, b, c = (per_seed[s][k]['test_rocauc'] for k in ('A_legacy', 'B_split', 'C_split_auc'))
        print(f'  {s:<8}{a:>12.4f}{b:>12.4f}{c:>14.4f}{b - a:>+10.4f}{c - b:>+10.4f}')


if __name__ == '__main__':
    main()
