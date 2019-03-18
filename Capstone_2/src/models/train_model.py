import numpy as np
import pandas as pd
from collections import Counter
import re
import string
import itertools

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SVMSMOTE

import nltk
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import Word2Vec
from gensim.parsing.preprocessing import preprocess_string
from gensim.test.utils import get_tmpfile

np.random.seed(42)

def load_and_split_df(filepath, features='text', label='readmission', index_col=0):
    '''
    A function to load a dataframe from a CSV and split into train/test sets
    
    filepath: the path to the desired CSV file
    features: the name of the column(s) containing feature variables
    label: the name of the column containing the label
    
    returns train and test features and labels as numpy arraysS
    '''
    # load the data
    df = pd.read_csv(filepath, index_col=index_col)

    # define text feature
    text = df[features].values

    # define target
    target = df[label].values
    
    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(text, 
                                                        target, 
                                                        stratify=target, 
                                                        test_size=0.33, 
                                                        random_state=42)
    
    return X_train, X_test, y_train, y_test

def logistic_regression(lr_params, train_feat, train_label, model,
                        test_feat, test_label, vec_params=None, random_state=42):
    '''
    A function to model data using logistic regression with under- or over-sampling.
    '''
    if model == 'svmsmote':
        pipe = make_pipeline(CountVectorizer(**vec_params),
                             SVMSMOTE(random_state=random_state),
                             LogisticRegression(**lr_params))
    elif model == 'rus':
        pipe = make_pipeline(CountVectorizer(**vec_params),
                             RandomUnderSampler(random_state=random_state),
                             LogisticRegression(**lr_params))
    
    
    pipe_fit = pipe.fit(train_feat, train_label)
    y_pred = pipe_fit.predict(test_feat)
    
    cnf_matrix = confusion_matrix(test_label, y_pred)
    
    return pipe, pipe_fit, y_pred, cnf_matrix
    
def random_forest_undersampler(vec_params, rf_params, train_feat, train_label, test_feat,
                               test_label, random_state=42, tfidf=False):
    '''
    A function to classify text data using count vectorization, random under-sampling,
    and a random forest. Returns an array of predicted labels.
    
    vec_params = parameters for the CountVectorizer
    rf_params = parameters for the RandomForestClassifier
    train_features = an array of training features
    test_features = an array of testing features
    labels = an array of training labels
    '''
    if tfidf:
        pipe = make_pipeline(CountVectorizer(**vec_params),
                             TfidfTransformer(),
                             RandomUnderSampler(random_state=random_state),
                             RandomForestClassifier(**rf_params))
    else:
        pipe = make_pipeline(CountVectorizer(**vec_params),
                             RandomUnderSampler(random_state=random_state),
                             RandomForestClassifier(**rf_params))
    
    pipe_fit = pipe.fit(train_feat, train_label)
    y_pred = pipe_fit.predict(test_feat)
    
    cnf_matrix = confusion_matrix(test_label, y_pred)
    
    return pipe, pipe_fit, y_pred, cnf_matrix

def svm_text_classification(vec_params, svm_params, train_feat, train_label, test_feat,
                            test_label, random_state=42):
    '''
    A function to classify text data using count vectorization, random under-sampling,
    and a random forest.
    
    train_features = an array of training features
    test_features = an array of testing features
    labels = an array of training labels
    vec_params = parameters for the CountVectorizer
    rf_params = parameters for the RandomForestClassifier
    '''
    
    pipe = make_pipeline(CountVectorizer(**vec_params),
                             RandomUnderSampler(random_state=random_state),
                             SVC(**svm_params))
    
    pipe_fit = pipe.fit(train_feat, train_label)
    y_pred = pipe_fit.predict(test_feat)
    
    cnf_matrix = confusion_matrix(test_label, y_pred)
    
    return pipe, pipe_fit, y_pred, cnf_matrix


# Word2Vec Modeling
def create_w2v_dataframe(file, idx=0, label_col='readmission', text_col='text', test_size=0.33, random_state=42):
    '''
    A function to load and split a dataframe for Word2Vec processing.
    '''
    # load dataframe
    df = pd.read_csv(file, index_col=idx)
    
    # drop all but text and label data
    df = df[[label_col, text_col]]
    df.columns = ['label', 'text']
    df = df.reset_index(drop = True)
    
    # split data into training and validation set
    df_trn, df_val = train_test_split(df, 
                                      stratify = df.label, 
                                      test_size = test_size, 
                                      random_state = random_state)
    
    return df_trn, df_val

def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation


def lda_get_good_tokens(df):
    df['tokenized_text'] = list(map(nltk.word_tokenize, df.text))
    df['tokenized_text'] = list(map(get_good_tokens, df.tokenized_text))

    
def remove_stopwords(df):
    # define stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    
    # remove stopwords
    df['stopwords_removed'] = list(map(lambda doc:
                                       [word for word in doc if word not in stopwords],
                                       df.tokenized_text))


def stem_words(df):
    lemm = nltk.stem.WordNetLemmatizer()
    df['lemmatized_text'] = list(map(lambda sentence:
                                     list(map(lemm.lemmatize, sentence)),
                                     df.stopwords_removed))

    p_stemmer = nltk.stem.porter.PorterStemmer()
    df['stemmed_text'] = list(map(lambda sentence:
                                  list(map(p_stemmer.stem, sentence)),
                                  df.lemmatized_text))
    
def document_to_lda_features(lda_model, document):
    """ Transforms a bag of words document to features.
    It returns the proportion of how much each topic was
    present in the document.
    """
    
    topic_importances = lda_model.get_document_topics(document, minimum_probability=0)
    topic_importances = np.array(topic_importances)
    
    return topic_importances[:,1]

def w2v_preprocessing(df):
    """ All the preprocessing steps for word2vec are done in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.
    """
    
    df['text'] = df.text.str.lower()
    df['document_sentences'] = df.text.str.split('.')  # split texts into individual sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(nltk.word_tokenize, sentences)),
                                         df.document_sentences))  # tokenize sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(get_good_tokens, sentences)),
                                         df.tokenized_sentences))  # remove unwanted characters
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(filter(lambda lst: lst, sentences)),
                                         df.tokenized_sentences))  # remove empty lists
  
def get_w2v_features(w2v_model, sentence_group):
    """
    Transform a sentence_group (containing multiple lists
    of words) into a feature vector. It averages out all the
    word vectors of the sentence_group.
    """

    chain = itertools.chain(*sentence_group)
    inner_chain = itertools.chain(*chain)
    words = np.array(list(inner_chain))  # words in text
    index2word_set = set(w2v_model.wv.vocab.keys())  # words known to model
    
    featureVec = np.zeros(w2v_model.vector_size, dtype="float32")
    
    # Initialize a counter for number of words in a review
    nwords = 0
    # Loop over each word in the comment and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1.
            
    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec


def word2vec_logistic_regression(train_feat, train_label, test_feat, 
                                 test_label, model='w2v_lda', 
                                 lr_params = {'solver':'liblinear','penalty':'l2', 
                                              'random_state':42}):
    '''
    A function to model data using logistic regression.
    
    lr_params : a dictionary of hyperparameters to pass to logistic regression.
    '''
    if model == 'w2v_lda':
        clf = LogisticRegression(**lr_params)
    
    clf_fit = clf.fit(train_feat, train_label)
    y_pred = clf_fit.predict(test_feat)
    
    cnf_matrix = confusion_matrix(test_label, y_pred)
    
    return clf, clf_fit, y_pred, cnf_matrix
    

def random_undersample(df, random_state=42):
    '''
    A function to randomly undersample the majority class
    to create a balanced dataset.
    '''
    # Separate majority class
    df_major = df[df.label==0]
    
    # Separate minority class
    df_minor = df[df.label==1]
    
    # Downsample majority class
    df_major_undersampled = resample(df_major,
                                     replace=False,
                                     n_samples=len(df_minor),
                                     random_state=random_state)
    
    # Combine minority class with downsampled majority class
    df_undersampled = pd.concat([df_major_undersampled, df_minor])
    
    return df_undersampled
    

