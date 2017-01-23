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
from sklearn.feature_extraction.text import TfidfVectorizer
import autosklearn.classification
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import train_test_split as tts
import sklearn.datasets as dt
from  sklearn.metrics import accuracy_score as accScore


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




def train_models(cfg):
    global data, target
    logger.info('Start loading the data file')
    #(target, _desc_) = load_data(cfg.datafile, cfg.char_class, cfg.char_desc)

    logger.info('Starting tfidf')
    #(vectorizer, data) = tf_idf(_desc_)
    #pickle_object(vectorizer, os.path.join(cfg.output_dir, 'vectorizer.pickle'))

    # run_tpot(train, _class_)
    logger.info('Starting Optimization')
    digits = dt.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = tts(X, y, random_state=8340)
    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(X_train, y_train)
    automl.show_models()
    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for Brand Classification")
    parser.add_argument('--datafile', help='Input data file', type=str, dest='datafile', default="/home/mluser/PycharmProjects/Thai/Docker-2/charsearch/src/NEW_MF.xlsx")
    parser.add_argument('--char-class', help='Char class', type=str, dest='char_class', default="ALLBRAND")
    parser.add_argument('--char-desc', help='Output data dir', type=str, dest='char_desc', default="Description")

    cfg = parser.parse_args()
    logger.info('Starting Optimization for %s' % cfg.char_class)
    train_models(cfg)
