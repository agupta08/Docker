# -*- coding: utf-8 -*-
import argparse
import logging

# import GPyOpt
import pandas as pd
import pickle
from polyglot.text import Text
from tpot import TPOTClassifier

import numpy as np
from bayes_opt import BayesianOptimization

import os
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score, log_loss)

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter
from imblearn.ensemble import EasyEnsemble, BalanceCascade


# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(filename, char_class, char_desc):
    # Note: skiping wrong rows
    data = pd.read_excel(filename)
    _class_ = data[char_class]
    _desc_ = data[char_desc]
    return (_class_, _desc_)


def pickle_object(object_to_pickle, destination):
    with open(destination, 'wb') as f:
        pickle.dump(object_to_pickle, f)


def transliterate_and_split(message):
    words = Text(message).transliterate("en")
    return [word for word in words if len(word) > 1 and word.isalpha()]


def tf_idf(corpus):
    vectorizer = TfidfVectorizer(
            # analyzer = 'word',
            analyzer=transliterate_and_split,
            # stop_words = 'english',
            min_df=0,
            decode_error='ignore',
            strip_accents='ascii',
            ngram_range=(1, 3))
    # Fit and transform input corpus
    model = vectorizer.fit_transform(corpus).toarray()
    return (vectorizer, model)


def run_tpot(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25)
    tpot = TPOTClassifier(generations=50, population_size=50, verbosity=2, num_cv_folds=5)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))


def train_models(cfg):
    global data, target
    logger.info('Start loading the data file')
    (target, _desc_) = load_data(cfg.datafile, cfg.char_class, cfg.char_desc)

    logger.info('Starting tfidf')
    (vectorizer, data) = tf_idf(_desc_)


    # run_tpot(train, _class_)
    logger.info('Starting Optimization')

    rfccv(78.9507 , 8.8488 , 1.0335 , 140.7349 , 0.1505)
    imBalance()

def imBalance():
    logger.info('Running SMOTETomek')
    try:
        sme = SMOTETomek(random_state=42, k=170)
        print('Original dataset shape {}'.format(Counter(target)))
        X_res, y_res = sme.fit_sample(data, target)
        print('Resampled dataset shape {}'.format(Counter(y_res)))
    except:
        logger.exception('Exception Error')

    logger.info('Running Easy Ensemble')
    try:
        sme = EasyEnsemble(random_state=42)
        print('Original dataset shape {}'.format(Counter(target)))
        X_res, y_res = sme.fit_sample(data, target)
        print('Resampled dataset shape {}'.format(Counter(y_res)))
    except:
        logger.exception('Exception Error')


def rfccv(n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features):
    clf = RFC(n_estimators=int(n_estimators),
              min_samples_split=int(min_samples_split),
              min_samples_leaf=int(min_samples_leaf),
              max_depth=int(max_depth),
              max_features=min(max_features, 0.999),
              random_state=2,
              class_weight='balanced')
    clf.fit(data, target)
    clf_probs = clf.predict_proba(data)
    score = log_loss(target, clf_probs)
    print("\tLog Loss: %1.3f" % score)
    y_pred = clf.predict(data)
    print("\tPrecision: %1.3f" % precision_score(target, y_pred, average='weighted'))
    print("\tRecall: %1.3f" % recall_score(target, y_pred,  average='weighted'))
    print("\tF1: %1.3f\n" % f1_score(target, y_pred, average='weighted'))


    sig_clf = CalibratedClassifierCV(clf, method="isotonic")
    sig_clf.fit(data, target)
    sig_clf_probs = sig_clf.predict_proba(data)
    score = log_loss(target, sig_clf_probs)
    print("\tLog Loss: %1.3f" % score)
    y_pred = sig_clf.predict(data)
    print("\tPrecision: %1.3f" % precision_score(target, y_pred, average='weighted'))
    print("\tRecall: %1.3f" % recall_score(target, y_pred,  average='weighted'))
    print("\tF1: %1.3f\n" % f1_score(target, y_pred, average='weighted'))
    return




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for Brand Classification")
    parser.add_argument('--datafile', help='Input data file', type=str, dest='datafile', default="NEW_MF_2.xlsx")
    parser.add_argument('--char-class', help='Char class', type=str, dest='char_class', default="ALLBRAND")
    parser.add_argument('--char-desc', help='Output data dir', type=str, dest='char_desc', default="Description")

    cfg = parser.parse_args()
    logger.info('Starting Optimization for %s' % cfg.char_class)
    train_models(cfg)
