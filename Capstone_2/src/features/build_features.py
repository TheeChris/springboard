import string
from nltk import word_tokenize

def clean_and_tokenize(text):
    '''
    tokenize the text by replacing punctuation and 
    numbers with spaces and lowercase all words
    '''
    
    punc_list = string.punctuation + string.digits
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.lower().translate(t)
    tokens = word_tokenize(text)
    return tokens