#!/usr/bin/env python
'''
| Filename    : supervised.ppdb.py
| Description :
| Author      : Pushpendre Rastogi
| Created     : Wed Dec 14 20:05:29 2016 (-0500)
| Last-Updated: .
|           By: .
|     Update #: 0
'''
import argparse
import cPickle as pkl
import contextlib
import functools
import gzip
import itertools
import numpy
import os
import pdb
import scipy.stats
import signal
import sys
import time
import traceback
from collections import OrderedDict
from eval import setup, embed_baseline, embed_cocoon
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
from collections import defaultdict

DIR='PATH for feature set'
labels=os.path.join(DIR, 'training_samples_labels')
features=os.path.join(DIR, 'training_samples_features.gz')

def undesirable(feat):
    return any(feat.startswith(e)
               for e
               in ['embedding_', 'unigrams-', 'normed-unigrams-'])

def get_feat_hash(features_fn, labels):
    all_possible_feat = OrderedDict()
    counter = 0
    vocab_set = set()
    with gzip.open(features_fn) as f1, open(labels) as f2:
        for i, (row1, row2) in enumerate(itertools.izip(f1, f2)):
            for phrase in row2.strip().split('\t')[1:3]:
                vocab_set.update(phrase.strip().split(' '))
            for e in row1.strip().split():
                if not undesirable(e):
                    feat, val = e.split(':')
                    if feat not in all_possible_feat:
                        all_possible_feat[feat] = counter
                        counter+=1
    assert len(all_possible_feat) == counter
    return all_possible_feat, i+1, frozenset(vocab_set)

def create_feature_matrix(features, labels, feat_hash, num_examples,
                          embed_fn=None, embed_fn_str=None):
    X_width = len(feat_hash)
    if args.embed:
        assert embed_fn is not None
        if embed_fn_str == 'embed_baseline':
            X_width += (args.dim)**2
        elif embed_fn_str == 'embed_cocoon':
            X_width += (args.dim / args.dim_divide * 2)**2
        else:
            raise NotImplementedError()
    else:
        assert embed_fn is None
    X = numpy.zeros((num_examples, X_width), dtype='float32')
    Y = numpy.zeros((num_examples,), dtype='float32')
    with gzip.open(features) as f1, open(labels) as f2:
        for i, (row1, row2) in enumerate(itertools.izip(f1, f2)):
            _, phrase1, phrase2, Yi = row2.strip().split('\t')
            Y[i] = numpy.float(Yi)
            if embed_fn is not None:
                X[i, len(feat_hash):] = numpy.outer(
                    embed_fn(phrase1.strip()),
                    embed_fn(phrase2.strip())).ravel()
            for e in row1.strip().split():
                if not undesirable(e):
                    feat, val = e.split(':')
                    X[i, feat_hash[feat]] = numpy.float(val)
    return X, Y

@contextlib.contextmanager
def tictoc(msg):
    stream = sys.stderr
    t = time.time()
    print >> stream, "Started", msg
    yield
    print >> stream, "\nCompleted", msg, "in %0.1fs" % (time.time() - t)

def mean(arr):
    if len(arr) == 0:
        raise NotImplementedError()
    return float(sum(arr))/len(arr)

def spearman_score(model, X, Y):
    # It also returns the p-value which we discard.
    v = scipy.stats.spearmanr(model.predict(X), Y)[0]
    return v


def crossval(n, k, shuffle=True):
    return KFold(n=n, n_folds=k, shuffle=shuffle)

def tolerant_print(cver, s):
    if hasattr(cver, s):
        print s, getattr(cver, s)


X_train = None
Y_train = None
def fit_and_test(alpha, A, B):
    return alpha, spearman_score(
        Ridge(alpha=alpha).fit(X_train[A], Y_train[A]),
        X_train[B],
        Y_train[B])

def get_best_alpha(lot):
    d = defaultdict(float)
    for k, v in lot:
        d[k] += v
    l = sorted(d.items(), key=lambda x: x[1], reverse=True)
    print >> sys.stderr, l
    return l[0][0]

def main_job(feature_fn_str, feature_fn):
    with tictoc('Featurizing'):
        X, Y = create_feature_matrix(
            features, labels, feat_hash, num_examples,
            embed_fn=feature_fn,
            embed_fn_str=feature_fn_str)
    scores = []
    for train_idx, test_idx in crossval(X.shape[0], 5):
        global X_train
        global Y_train
        X_train = X[train_idx]
        Y_train = Y[train_idx]
        values = Parallel(n_jobs=10)(
            delayed(fit_and_test)(alpha, ctrain_idx, ctest_idx)
            for alpha in [40, 60, 80, 100]
            for ctrain_idx, ctest_idx in crossval(X_train.shape[0], 3))
        best_alpha = get_best_alpha(values)
        print >> sys.stderr, best_alpha
        scores.append(
            spearman_score(
                Ridge(alpha=best_alpha).fit(X_train, Y_train),
                X[test_idx],
                Y[test_idx]))
    print feature_fn_str, \
        'args.embed', args.embed, \
        'args.dim', args.dim, \
        'args.dim_divide', args.dim_divide, \
        'mean(scores)', mean(scores), \
        'scores', scores
    return

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='')
    arg_parser.add_argument('--embed', default=0, type=int)
    arg_parser.add_argument('--dim', default=100, type=int)
    arg_parser.add_argument('--dim_divide', default=4, type=int)
    arg_parser.add_argument('--test_baseline', default=1, type=int)
    args=arg_parser.parse_args()
    feat_hash, num_examples, vocab_set = get_feat_hash(features, labels)
    if args.embed:
        feature_fn_list = [('embed_cocoon',
                            functools.partial(embed_cocoon,
                                              dim=args.dim,
                                              window=args.dim_divide/2,
                                              dim_div=args.dim_divide))]
        if args.test_baseline:
            feature_fn_list.append(('embed_baseline', embed_baseline))
        with tictoc('Setup'):
            setup(args.dim, args.dim_divide, vocab_set)
    else:
        feature_fn_list = [('None', None)]
    for feature_fn_str, feature_fn in feature_fn_list:
        main_job(feature_fn_str, feature_fn)
