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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import uuid


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
            analyzer = 'word',
            #analyzer=transliterate_and_split,
             stop_words = 'english',
            min_df=0,
            decode_error='ignore',
            strip_accents='ascii',
            ngram_range=(1, 3))
    # Fit and transform input corpus
    model = vectorizer.fit_transform(corpus).toarray()
    return (vectorizer, model)


def run_tpot(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25)
    tpot = TPOTClassifier(generations=10, population_size=5, verbosity=2, num_cv_folds=5)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))


def train_models(cfg):
    global data, target
    logger.info('Start loading the data file')
    (target, _desc_) = load_data(cfg.datafile, cfg.char_class, cfg.char_desc)

    logger.info('Starting tfidf')
    (vectorizer, data) = tf_idf(_desc_)
    pickle_object(vectorizer, os.path.join(cfg.output_dir, 'vectorizer4.pickle'))

    # run_tpot(train, _class_)
    logger.info('Starting Optimization')
    runByOptRF(cfg.output_dir, cfg.n_iter, cfg.init_points, cfg.char_class)
    #runByOptXB(cfg.output_dir, cfg.n_iter, cfg.init_points)
    #runByOptABoost(cfg.output_dir, cfg.n_iter, cfg.init_points)
    # print(rfccv(1.14693784e+02,   7.36528405e+00,   1.72591442e+00,   4.09992268e+01,   1.00000000e-01))
    # runGpyOptRF()


def rfccv(n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features):
    clf = RFC(n_estimators=int(n_estimators),
              min_samples_split=int(min_samples_split),
              min_samples_leaf=int(min_samples_leaf),
              max_depth=int(max_depth),
              max_features=min(max_features, 0.999),
              random_state=2, class_weight='balanced')
    score = cross_val_score(clf, data, target, cv=5, n_jobs=2).mean()
    return score

def rfccvBalanced(n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features):
    clf = RFC(n_estimators=int(n_estimators),
              min_samples_split=int(min_samples_split),
              min_samples_leaf=int(min_samples_leaf),
              max_depth=int(max_depth),
              max_features=min(max_features, 0.999),
              random_state=2, class_weight='balanced')
    clf.fit(data, target)
    score = cross_val_score(clf, data, target, cv=5, n_jobs=2).mean()

    sig_clf = CalibratedClassifierCV(clf, method="isotonic")
    sig_clf.fit(data, target)
    score = cross_val_score(sig_clf, data, target, cv=5, n_jobs=2).mean()
    return score



def xbCV(learning_rate, n_estimators, max_depth, min_child_weight,gamma, subsample,colsample_bytree ):
    clf = XGBClassifier(learning_rate=learning_rate,
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth),
                        min_child_weight=int(min_child_weight),
                        gamma=gamma,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree)
    score = cross_val_score(clf, data, target, cv=5, n_jobs=2).mean()
    return score



def runByOptXB(model_output_dir, n_iter, init_points):
    xbcBO = BayesianOptimization(xbCV, {
                  'learning_rate': (0.01, 0.5),
                  'n_estimators': (10, 50),
                  'max_depth': (3, 10),
                  'min_child_weight': (6, 12),
                  'gamma': (0, 0.5),
                  'subsample': (0.6, 1.0),
                  'colsample_bytree': (0.6, 1.)})
    xbcBO.maximize(n_iter=n_iter, init_points=init_points)
    print('RFC: %f' % rfcBO.res['max']['max_val'])
    d = rfcBO.res['max']['max_params']
    print(d)



def AbCV(learning_rate, n_estimators,max_depth, algorithm ):
    alg = 'SAMME' if round(algorithm)<2.0 else 'SAMME.R'
    clf = ABC(DTC(max_depth=int(max_depth)),
                        learning_rate=learning_rate,
                        n_estimators=int(n_estimators),
                        algorithm = alg)
    score = cross_val_score(clf, data, target, cv=5, n_jobs=5).mean()
    return score

def runByOptABoost(model_output_dir, n_iter, init_points):
    xbcBO = BayesianOptimization(AbCV, {
                  'learning_rate': (0.01, 1.0),
                  'max_depth' :(20,150),
                  'n_estimators': (10, 1000),
                    'algorithm' : (1,3)})
    xbcBO.maximize(n_iter=n_iter, init_points=init_points)
    print('AdaBoost: %f' % rfcBO.res['max']['max_val'])
    d = rfcBO.res['max']['max_params']
    print(d)


def runByOptRF(model_output_dir, n_iter, init_points, char_class):
    rfcBO = BayesianOptimization(rfccv, {'n_estimators': (10, 250),
                                         'min_samples_split': (2, 25),
                                         'min_samples_leaf': (1, 10),
                                         'max_depth': (3, 200),
                                         'max_features': (0.1, 0.999)})
    rfcBO.maximize(n_iter=n_iter, init_points=init_points)
    print('RFC: %f' % rfcBO.res['max']['max_val'])
    d = rfcBO.res['max']['max_params']
    print(d)

    logger.info('Dumping data to CSV')
    rfcBO.points_to_csv(char_class + '_'+str(uuid.uuid4()))

    clf = RFC(n_estimators=int(d['n_estimators']),
              min_samples_split=int(d['min_samples_split']),
              max_depth=int(d['max_depth']),
              max_features=min(d['max_features'], 0.999),
              min_samples_leaf=int(d['min_samples_leaf']),
              random_state=2)

    clf.fit(data, target)
    pickle_object(clf, os.path.join(model_output_dir, "model4.pickle"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for Classification using RFC")
    parser.add_argument('--datafile', help='Input data file', type=str, dest='datafile', default="MF_Data_4.xlsx")
    parser.add_argument('--output-dir', help='Output data dir', type=str, dest='output_dir', default="/home/mluser/PycharmProjects/Thai/Docker-2/charsearch/output/")
    parser.add_argument('--n-iter', help='Number of iterations', type=int, dest='n_iter', default=100)
    parser.add_argument('--init-points', help='Init points', type=int, dest='init_points', default=10)
    parser.add_argument('--char-class', help='Char class', type=str, dest='char_class', default="86_GENDER")
    parser.add_argument('--char-desc', help='Output data dir', type=str, dest='char_desc', default="Desc")

    cfg = parser.parse_args()
    logger.info('Starting Optimization for %s' % cfg.char_class)
    train_models(cfg)
