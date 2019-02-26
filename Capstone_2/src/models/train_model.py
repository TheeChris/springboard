import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SVMSMOTE


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

def logistic_regression(vec_params, lr_params, train_feat, train_label, model,
                        test_feat, test_label, random_state=42):
    '''
    A function to model data using logistic regression with under- or over-sampling.S
    '''
    if model == 'svmsmote':
        pipe = make_pipeline(CountVectorizer(**vec_params),
                             SVMSMOTE(random_state=random_state),
                             LogisticRegression(**lr_params))       
    else:
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
    