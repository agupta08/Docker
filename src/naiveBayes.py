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
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import VotingClassifier



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
    pickle_object(vectorizer, os.path.join(cfg.output_dir, 'vectorizer.pickle'))

    # run_tpot(train, _class_)
    logger.info('Starting Optimization')

    runByOptNB(cfg.output_dir, cfg.n_iter, cfg.init_points)
    #runByOptXB(cfg.output_dir, cfg.n_iter, cfg.init_points)
    #runByOptABoost(cfg.output_dir, cfg.n_iter, cfg.init_points)
    # print(rfccv(1.14693784e+02,   7.36528405e+00,   1.72591442e+00,   4.09992268e+01,   1.00000000e-01))
    # runGpyOptRF()


def BernoulliNBcv(alpha):
    clf = BernoulliNB(alpha=alpha)
    score = cross_val_score(clf, data, target, cv=5, n_jobs=5).mean()
    return score


def runByOptNB(model_output_dir, n_iter, init_points):
    rfcBO = BayesianOptimization(BernoulliNBcv, {'alpha': (0.001, 0.1)})
    rfcBO.maximize(n_iter=n_iter, init_points=init_points)
    print('RFC: %f' % rfcBO.res['max']['max_val'])
    d = rfcBO.res['max']['max_params']
    print(d)

    clf = BernoulliNB(alpha=0.061832276807032509)
    rfc = RFC(n_estimators=int(184.0961),
              min_samples_split=int(2.0),
              max_depth=int(200.0),
              max_features=0.1,
              min_samples_leaf=int(1.0),
              random_state=2)
    clf.fit(data, target)
    rfc.fit(data, target)

    vClf =VotingClassifier(estimators=[('bnb', clf), ('rfc', rfc)], voting='soft')
    vClf.fit(data, target)
    logger.info('Dumping Model Pickle')
    pickle_object(vClf, os.path.join(model_output_dir, "Voting_model.pickle"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for Classification using Naive Bayes")
    parser.add_argument('--datafile', help='Input data file', type=str, dest='datafile', default="NEW_MF_3.xlsx")
    parser.add_argument('--output-dir', help='Output data dir', type=str, dest='output_dir', default="/home/mluser/PycharmProjects/Thai/Docker-2/charsearch/output/ALLBRAND/NaiveBayes")
    parser.add_argument('--n-iter', help='Number of iterations', type=int, dest='n_iter', default=1)
    parser.add_argument('--init-points', help='Init points', type=int, dest='init_points', default=1)
    parser.add_argument('--char-class', help='Char class', type=str, dest='char_class', default="ALLBRAND")
    parser.add_argument('--char-desc', help='Output data dir', type=str, dest='char_desc', default="Description")

    cfg = parser.parse_args()
    logger.info('Starting Optimization for %s' % cfg.char_class)
    train_models(cfg)
