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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


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



def train_models(cfg):
    global data, target
    logger.info('Start loading the data file')
    (target, _desc_) = load_data(cfg.datafile, cfg.char_class, cfg.char_desc)

    logger.info('Starting tfidf')
    (vectorizer, data) = tf_idf(_desc_)

    logger.info('Dumping Vector Files to explore')
    pickle_object(vectorizer, os.path.join(cfg.output_dir, 'vectorizer.pickle'))

    logger.info('Running RFC and Model Pickle')
    rfccv2(98.7170 , 21.6229 , 1.1359 , 120.3378 , 0.2378)
    logger.info('Completed')

def rfccv2(n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features):
    clf = RFC(n_estimators=int(n_estimators),
              min_samples_split=int(min_samples_split),
              max_depth=int(max_depth),
              max_features=min(max_features, 0.999),
              min_samples_leaf=int(min_samples_leaf),
              random_state=2)
    score = cross_val_score(clf, data, target, cv=5, n_jobs=2).mean()
    logger.info(score)
    clf.fit(data, target)
    pickle_object(clf, os.path.join(cfg.output_dir, 'model.pickle'))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models for Brand Classification")
    parser.add_argument('--datafile', help='Input data file', type=str, dest='datafile', default="MF_Data_4.xlsx")
    parser.add_argument('--output-dir', help='Output data dir', type=str, dest='output_dir', default="/home/mluser/PycharmProjects/Thai/Docker-2/charsearch/output/86_VARIANTA/")
    parser.add_argument('--n-iter', help='Number of iterations', type=int, dest='n_iter', default=5)
    parser.add_argument('--init-points', help='Init points', type=int, dest='init_points', default=5)
    parser.add_argument('--char-class', help='Char class', type=str, dest='char_class', default="86_VARIANTA")
    parser.add_argument('--char-desc', help='Output data dir', type=str, dest='char_desc', default="Desc")

    cfg = parser.parse_args()
    logger.info('Starting Optimization for %s' % cfg.char_class)
    train_models(cfg)
