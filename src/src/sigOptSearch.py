# -*- coding: utf-8 -*-
from polyglot.text import Text

import pandas as pd
import numpy as np
from scipy import interp
from time import time
import argparse
import logging

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC

from tpot import TPOTClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

from sigopt_sklearn.search import SigOptSearchCV
import sigopt
client_token = 'RPSQZFSVGWDEVTYMSJVPJBPEJMOSHRMEIWCSESVJFOFIZYNR'
conn = sigopt.Connection(client_token=client_token)


# Should be input parameter
MODELS_DESTINATION_DIR = "./lib"
DATA_DESTINATION_DIR = "./data"
FILENAME = 'NEW_MF.xlsx'

CHAR_CLASS = 'ALLBRAND'
CHAR_DESCRIPTION = 'Description'

#setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filename):
    # Note: skiping wrong rows
    data = pd.read_excel(filename)
    _class_ = data[CHAR_CLASS]
    _desc_ = data[CHAR_DESCRIPTION]
    return (_class_, _desc_)

def transliterate_and_split(message):
    words = Text(message).transliterate("en")
    return [word for word in words if len(word) > 1 and word.isalpha()]


def tf_idf(corpus):
    vectorizer = TfidfVectorizer(
            #analyzer = 'word',
            analyzer = transliterate_and_split,
            #stop_words = 'english',
            min_df=0,
            decode_error = 'ignore',
            strip_accents = 'ascii',
            ngram_range=(1,3))
    # Fit and transform input corpus
    model = vectorizer.fit_transform(corpus).toarray()
    return (vectorizer, model)


def train_models(cfg):
    global data, target
    logger.info('Start loading the data file')
    (target, _desc_) = load_data(cfg.datafile)

    logger.info('Starting tfidf')
    (vectorizer, data) = tf_idf(_desc_)

    #run_tpot(train, _class_)
    logger.info('Initializing Optimization')

    runSigOpt()




def runSigOpt():
    rfc = RFC()
    rfcParams = {'n_estimators': (10, 8800),
                'min_samples_split': (1, 44),
                "criterion": ["gini", "entropy"],
                'min_samples_leaf': (1, 11),
                'max_depth': (3, 200),
                'max_features': (0.1, 0.999),
                  'max_leaf_nodes' : (10, 44)}
    rfcParams1 = {'n_estimators': (10, 250),
                  'min_samples_split': (2, 25),
                  'max_depth': (3, 200),
                   'max_features': (0.1, 0.999)}
    clf = SigOptSearchCV(rfc, rfcParams1,
                         client_token=client_token, n_jobs=2, n_iter=200, verbose=1)
    clf.fit(data,target)
    score = cross_val_score(clf, data, target, cv=5, n_jobs=5).mean()
    print(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Train models for Brand Classification")
    parser.add_argument('--datafile', help = 'Input data file', type=str, default = FILENAME)
    cfg = parser.parse_args()
    logger.info('Starting Optimization for %s' % (CHAR_CLASS))
    train_models(cfg)