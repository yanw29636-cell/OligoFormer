import torch
import os
import sys
import sklearn
import random
import pickle as pkl
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import data_process_loader, FM_DIMS
from torch.utils.data import DataLoader
from model import Oligo
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score,roc_curve,auc,precision_recall_curve,matthews_corrcoef
from metrics import  sensitivity, specificity, find_best_f1_threshold
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib as mpl
from logger import TrainLogger
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def find_metrics_best_for_shuffle(label, prob, cut_spe=0.95):
    fpr, tpr, thresholds = roc_curve(label, prob)
    a = 1 - fpr
    b = tpr
    Sensitivity = b
    Specificity = a
    mask = Specificity >= cut_spe
    Sensitivity_ = Sensitivity[mask]
    if (len(Sensitivity_) == 1) & (Sensitivity_[0] == 0):
        Sensitivity_best = ((Sensitivity[1] - Sensitivity[0]) / (Specificity[1] - Specificity[0])) * cut_spe + Sensitivity[1] - ((Sensitivity[1] - Sensitivity[0]) / (Specificity[1] - Specificity[0])) *  Specificity[1]
        best_threshold = thresholds[1]
    else:
        best_idx = np.argmax(Sensitivity_)
        Sensitivity_best = Sensitivity_[best_idx]
        best_threshold = thresholds[np.where(mask)[0][best_idx]]
    return Sensitivity_best, Sensitivity, Specificity, best_threshold

def get_kfold_data_2(i, datasets, k=5, v=1):
    datasets = shuffle_dataset(datasets, 42).reset_index(drop=True)
    v = v * 10
    if k<5:
        fold_size = len(datasets) // 5
    else:
        fold_size = len(datasets) // k

    test_start = i * fold_size

    if i != k - 1 and i != 0:
        test_end = (i + 1) * fold_size
        TestSet = datasets[test_start:test_end]
        TrainSet = pd.concat([datasets[0:test_start], datasets[test_end:]])

    elif i == 0:
        test_end = fold_size
        TestSet = datasets[test_start:test_end]
        TrainSet = datasets[test_end:]

    else:
        TestSet = datasets[test_start:]
        TrainSet = datasets[0:test_start]

    return TrainSet.reset_index(drop=True), TestSet.reset_index(drop=True)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    dataset = shuffle(dataset)
    return dataset

def write_pkl(pkl_data,pkl_name):
	pkl_file = open(pkl_name, "wb")
	pkl.dump(pkl_data, pkl_file)
	pkl_file.close()

def plotPRC(model,X,Y,name):
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	y_pred = model.predict(X)[:,1]
	precision, recall, threshold = precision_recall_curve(Y[:,1],y_pred)
	prc = auc(recall, precision)
	plt.plot(recall, precision,label='OligoFormer (PRC: %s \u00B1 0.001)' % (np.round(prc, 3)))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc='best')
	plt.savefig(name)
	Y_TRUE = pd.DataFrame(Y)
	Y_PRED = pd.DataFrame(model.predict(X)[:,1])
	with open(name.split('PRC')[0] + 'test_prediction.txt', 'w') as f:
		for i in range(Y_TRUE.shape[0]):
			f.write(str(Y_TRUE.iloc[i,1]) + " " + str(Y_PRED.iloc[i,0]) + '\n')

def plotAUC(model,X,Y,name):
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	y_pred = model.predict(X)[:,1]
	fpr, tpr, threshold = roc_curve(Y[:,1],y_pred)
	roc = auc(fpr, tpr)
	plt.plot(fpr, tpr,label='OligoFormer (AUC: %s \u00B1 0.001)' % (np.round(roc, 3)))
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.legend(loc='best')
	plt.savefig(name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)
        return self.avg

def val(model, criterion, dataloader, threshold=None):
	"""Evaluate `model` on `dataloader`.

	threshold=None (default, legacy behaviour) calibrates the decision threshold on this
	very set via find_best_f1_threshold. Passing a threshold instead applies one calibrated
	elsewhere (e.g. on a held-out val split), which is what an honest test-set score needs --
	self-calibrating on the test set is itself a mild form of leakage.
	The calibrated/applied threshold is returned so callers can forward it.
	"""
	model.eval()
	running_loss = AverageMeter()
	pred_list = []
	label_list = []
	with torch.no_grad():
		for i, data in enumerate(dataloader):
			siRNA = data[0].to(device)
			mRNA = data[1].to(device)
			siRNA_FM = data[2].to(device)
			mRNA_FM = data[3].to(device)
			label = data[4].to(device)
			td = data[6].to(device)
			y = data[5].to(device)
			pred,_,_ = model(siRNA,mRNA,siRNA_FM,mRNA_FM,td)
			loss = criterion(torch.log(pred.clamp(min=1e-8)), y.long())
			label = data[5]
			pred_prob = F.softmax(pred, dim=-1) # pred_prob = pred #
			pred_prob, indices = torch.max(pred_prob, dim=-1)
			pred_prob[indices == 0] = 1. - pred_prob[indices == 0]
			pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
			label_list.append(label)
			running_loss.update(loss.item(), label.shape[0])
	pred = np.concatenate(pred_list, axis=0)
	label = np.concatenate(label_list, axis=0)
	# calibrate the decision threshold on this validation set instead of a naive 0.5/argmax cutoff
	best_threshold = find_best_f1_threshold(label, pred) if threshold is None else threshold
	pred_cls = (pred >= best_threshold).astype(int)
	acc = accuracy_score(label, pred_cls)
	sen = sensitivity(label,pred_cls)
	spe = specificity(label,pred_cls)
	pre = precision_score(label, pred_cls)
	rec = recall_score(label, pred_cls)
	f1score=f1_score(label,pred_cls)
	rocauc = roc_auc_score(label, pred)
	prauc=average_precision_score(label, pred)
	mcc=matthews_corrcoef(label,pred_cls)
	epoch_loss = running_loss.get_average()
	running_loss.reset()
	model.train()
	return epoch_loss, acc, sen, spe, pre, rec, f1score, rocauc, prauc, mcc, label, pred, best_threshold

# metric names in the order val() returns them, for the per-epoch CSV
_METRIC_NAMES = ['loss', 'acc', 'sen', 'spe', 'pre', 'rec', 'f1', 'rocauc', 'prauc', 'mcc']

def _metrics_at(label, pred, threshold, loss):
	"""Recompute val()'s metric tuple from cached predictions at a different threshold.

	Lets us score the test set under both its own calibrated threshold (what the legacy
	pipeline effectively reported) and one carried over from the val split, without a second
	forward pass. loss, rocauc and prauc are threshold-free and just pass through.
	"""
	pred_cls = (pred >= threshold).astype(int)
	return [loss, accuracy_score(label, pred_cls), sensitivity(label, pred_cls),
		specificity(label, pred_cls), precision_score(label, pred_cls),
		recall_score(label, pred_cls), f1_score(label, pred_cls),
		roc_auc_score(label, pred), average_precision_score(label, pred),
		matthews_corrcoef(label, pred_cls)]

def _split_val_test(df, frac, split_seed):
	"""Carve `df` into (val, test) with `frac` of the rows going to val.

	Stratified on the binary y column so both halves keep the dataset's class prior, and
	seeded off `split_seed` rather than the training seed so that every seed/config in a
	comparison scores against the *same* test rows.

	Split on unique siRNA rather than on rows: Mix contains 8 siRNAs measured twice (16 rows
	with differing label values), and letting a replicate pair straddle the split would put a
	near-identical row on both sides.
	"""
	uniq = np.sort(df['siRNA'].unique())
	strat = df.groupby('siRNA')['y'].first().loc[uniq].values
	val_g, test_g = train_test_split(
		uniq, train_size=frac, random_state=split_seed, shuffle=True, stratify=strat)
	val_g, test_g = set(val_g), set(test_g)
	val = df[df['siRNA'].isin(val_g)].reset_index(drop=True)
	test = df[df['siRNA'].isin(test_g)].reset_index(drop=True)
	return val, test

def train(Args):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(Args.cuda)
	random.seed(Args.seed)
	os.environ['PYTHONHASHSEED']=str(Args.seed)
	np.random.seed(Args.seed)
	torch.manual_seed(Args.seed)
	torch.cuda.manual_seed_all(Args.seed)
	train_df = pd.read_csv(Args.path + Args.datasets[0] + '.csv', dtype=str)
	full_df = pd.read_csv(Args.path + Args.datasets[1] + '.csv',  dtype=str)
	if Args.relabel_median:
		for df in (train_df, full_df):
			labels = df['label'].astype(float)
			df['y'] = (labels > labels.median()).astype(int).astype(str)
	# --val_split 0 (default) is the legacy setup: "valid" and "test" are both the whole of
	# datasets[1], so checkpoint selection / early stopping / F1-threshold calibration all
	# happen on the same rows that get reported. >0 carves a held-out test part that nothing
	# in the selection path ever sees.
	split_mode = Args.val_split > 0
	if split_mode:
		valid_df, test_df = _split_val_test(full_df, Args.val_split, Args.split_seed)
	else:
		valid_df = full_df
		test_df = full_df.copy()
	# --train_val_split carves the selection set out of datasets[0] instead, leaving all of
	# datasets[1] untouched. It necessarily shrinks the training set, so a run using it is not
	# directly comparable to one that trains on all of datasets[0]; log both selection sets in
	# the same run and the confound can at least be isolated post-hoc.
	trainval_df = None
	if Args.train_val_split > 0:
		if not split_mode:
			raise ValueError('--train_val_split requires --val_split > 0 so there are held-out test rows to report on')
		trainval_df, train_df = _split_val_test(train_df, Args.train_val_split, Args.split_seed)
	params = {'batch_size': Args.batch_size,
			'shuffle': True,
			'num_workers': 0,
			'drop_last': False}
	# Evaluation never needs shuffling; in split mode we turn it off so the extra eval passes
	# don't consume torch RNG draws and perturb the training trajectory. Legacy mode keeps
	# shuffle=True everywhere so its runs stay bit-identical to previously recorded ones.
	# Evaluation runs under no_grad, so a larger batch is free wall-clock; only fp accumulation
	# order changes, and every arm of the comparison reads the same run anyway.
	eval_params = dict(params, shuffle=False, batch_size=max(Args.batch_size, 128)) if split_mode else params
	if Args.embedding_source == 'rinalmo':
		if not os.path.exists('./data/RiNALMo'):
			os.system('bash scripts/RiNALMo-features.sh')
	elif not os.path.exists('./data/RNAFM'):
		os.system('bash scripts/RNA-FM-features.sh')
	def _loader(df, name, prm):
		return DataLoader(data_process_loader(df.index.values, df.label.values, df.y.values, df, name, Args.path, Args.embedding_source), **prm)
	train_ds = _loader(train_df, Args.datasets[0], params)
	trainval_ds = _loader(trainval_df, Args.datasets[0], eval_params) if trainval_df is not None else None
	valid_ds = _loader(valid_df, Args.datasets[1], eval_params)
	test_ds = _loader(test_df, Args.datasets[1], eval_params)
	# No full_ds loader: val and test partition full_df exactly, so the 'full' metrics are
	# recovered by concatenating the two prediction vectors -- one fewer forward pass per epoch.
	OFmodel = Oligo(vocab_size = Args.vocab_size, embedding_dim = Args.embedding_dim, lstm_dim = Args.lstm_dim,  n_head = Args.n_head, n_layers = Args.n_layers, lm1 = Args.lm1, lm2 = Args.lm2, fm_dim = FM_DIMS[Args.embedding_source], lstm_batch_first = Args.lstm_batch_first).to(device)
	if Args.resume is not None:
		OFmodel.load_state_dict(torch.load(Args.resume,map_location=device))
	criterion = nn.NLLLoss()
	best_AUC = 0.0
	best_loss = 1e10
	best_epoch = 0
	tolerence_epoch = Args.early_stopping
	optimizer = optim.Adam(OFmodel.parameters(), lr=Args.learning_rate, weight_decay=Args.l2_decay)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = Args.weight_decay)
	params = dict(
		data_path=Args.path,
		save_dir=Args.output_dir,
		dataset=Args.datasets[1],
		batch_size=Args.batch_size
	)
	logger = TrainLogger(params)
	logger.info("pyTorch!!!")
	logger.info(f"Number of train: {train_df.shape[0]}")
	logger.info(f"Number of val: {valid_df.shape[0]}")
	logger.info(f"Number of test: {test_df.shape[0]}")
	if split_mode:
		logger.info(f"split_mode: val_split={Args.val_split} split_seed={Args.split_seed} "
			f"select_on={Args.select_on} select_metric={Args.select_metric} "
			f"early_stopping={'off' if Args.no_early_stop else tolerence_epoch}")
	csv_f = None
	if Args.epoch_metrics_csv:
		os.makedirs(os.path.dirname(os.path.abspath(Args.epoch_metrics_csv)), exist_ok=True)
		csv_f = open(Args.epoch_metrics_csv, 'w')
		# 'test' is scored with the threshold calibrated on val; 'testself' with one calibrated
		# on the test rows themselves (the legacy, optimistic convention). rocauc/prauc/loss are
		# threshold-free and therefore identical between the two blocks.
		prefixes = ['full', 'val', 'test', 'testself'] if split_mode else ['val', 'test']
		if trainval_ds is not None:
			prefixes = prefixes + ['trainval']
		csv_f.write('seed,epoch,train_loss,' + ','.join(
			f'{p}_{m}' for p in prefixes for m in _METRIC_NAMES) + ',thr_val,thr_testself\n')
		csv_f.flush()
	print('-----------------Start training!-----------------')
	running_loss = AverageMeter()
	for epoch in range(Args.epoch):
		for i, data in enumerate(train_ds):
			siRNA = data[0].to(device)
			mRNA = data[1].to(device)
			siRNA_FM = data[2].to(device)
			mRNA_FM = data[3].to(device)
			label = data[4].to(device)
			td = data[6].to(device)
			y = data[5].to(device)
			siRNA.requires_grad = True
			output, siRNA_attention, mRNA_attention = OFmodel(siRNA,mRNA,siRNA_FM,mRNA_FM,td)
			loss = criterion(torch.log(output.clamp(min=1e-8)), y.long())
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			running_loss.update(loss, output.shape[0])
			optimizer.step()
		scheduler.step()
		epoch_loss = running_loss.get_average()
		running_loss.reset()
		#train_loss, train_acc, train_sen, train_spe, train_pre, train_rec, train_f1, train_rocauc, train_prauc, train_mcc, train_label, train_pred = val(OFmodel, criterion, train_ds)
		val_r = val(OFmodel, criterion, valid_ds)
		val_m, val_label, val_pred, thr_val = list(val_r[:10]), val_r[10], val_r[11], val_r[12]
		if split_mode:
			test_r = val(OFmodel, criterion, test_ds, threshold=thr_val)
			test_m, test_label, test_pred = list(test_r[:10]), test_r[10], test_r[11]
			thr_testself = find_best_f1_threshold(test_label, test_pred)
			testself_m = _metrics_at(test_label, test_pred, thr_testself, test_m[0])
			# 'full' = all of datasets[1] = val part + test part, reconstructed from the two
			# prediction vectors rather than re-running the model over the union.
			full_label = np.concatenate([val_label, test_label])
			full_pred = np.concatenate([val_pred, test_pred])
			full_loss = (val_m[0] * len(val_label) + test_m[0] * len(test_label)) / len(full_label)
			full_m = _metrics_at(full_label, full_pred, find_best_f1_threshold(full_label, full_pred), full_loss)
			# arm A reproduces the legacy rule (select on all of datasets[1], i.e. val+test);
			# arm B/C select on the val part only and never touch the test rows.
			csv_row = full_m + val_m + test_m + testself_m
			trainval_m = None
			if trainval_ds is not None:
				trainval_m = list(val(OFmodel, criterion, trainval_ds)[:10])
				csv_row = csv_row + trainval_m
			sel_m = {'all': full_m, 'val': val_m, 'trainval': trainval_m}[Args.select_on]
			if sel_m is None:
				raise ValueError('--select_on trainval requires --train_val_split > 0')
		else:
			test_r = val(OFmodel, criterion, test_ds)
			test_m, thr_testself = list(test_r[:10]), test_r[12]
			sel_m = val_m
			csv_row = val_m + test_m
		val_loss, val_acc, val_pre, val_rec, val_f1, val_rocauc, val_prauc = (
			val_m[0], val_m[1], val_m[4], val_m[5], val_m[6], val_m[7], val_m[8])
		sel_loss, sel_rocauc = sel_m[0], sel_m[7]
		if csv_f is not None:
			csv_f.write('%d,%d,%.6f,' % (Args.seed, epoch, epoch_loss) +
				','.join('%.6f' % v for v in csv_row) + ',%.6f,%.6f\n' % (thr_val, thr_testself))
			csv_f.flush()
		if Args.select_metric == 'rocauc':
			improved = sel_rocauc > best_AUC
		else:
			improved = sel_loss < best_loss and sel_rocauc > best_AUC
		if improved:
			best_AUC = sel_rocauc
			best_loss = sel_loss
			best_epoch = epoch
			msg = "epoch-%d, loss-%.4f, val_loss-%.4f, val_acc-%.4f,val_pre-%.4f, val_rec-%.4f, val_rocauc-%.4f, val_prc-%.4f,val_f1-%.4f ***" % (epoch, epoch_loss, val_loss, val_acc, val_pre, val_rec,val_rocauc,val_prauc,val_f1)
			if not Args.no_checkpoint:
				torch.save(OFmodel.state_dict(), os.path.join(logger.get_model_dir(), msg+'.pth'))
		else:
			msg = "epoch-%d, loss-%.4f, val_loss-%.4f, val_acc-%.4f,val_pre-%.4f, val_rec-%.4f, val_rocauc-%.4f, val_prc-%.4f,val_f1-%.4f ***" % (epoch, epoch_loss, val_loss, val_acc, val_pre, val_rec,val_rocauc,val_prauc,val_f1)
		logger.info(msg)
		if not Args.no_early_stop and epoch - best_epoch > tolerence_epoch:
			break
	if csv_f is not None:
		csv_f.close()

