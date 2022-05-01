import re
from os.path import exists

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
import gensim.downloader as api

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = str(text).lower()                    # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)       # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)            # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)          # Remove ellipsis (and last word)
    #text = re.sub(r"(?<=\w)-(?=\w)", "", text)  # Remove dash between words
    text = re.sub(r'[^\w\-\s]', "", text)       # Remove punctuation

    return text

def get_model_wv(model='custom', retrain=False):
    path = './results/mbic-article-word2vec.model'

    if exists(path) and not retrain and model == 'custom':
        model = Word2Vec.load(path)
        return model.wv
    
    elif model != 'custom':
        return api.load(model)

    elif retrain:
        # import the article texts from MBIC, which are the corpus of our custom model
        df = pd.read_json('./mbic/labeled_dataset.json')

        # convert each sentence text to a list of individual words
        sentences = df['article'].tolist()
        corpus = []

        for item in sentences:
            if item != None:
                text = clean_text(item)
                tokens = text.split(' ')

                tokens = [t for t in tokens if not t in stopwords.words('english')]  # Remove stopwords
                tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
                tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens

                corpus.append(tokens)

        model = Word2Vec(corpus, vector_size=300)
        
        print(f'Saving newly-trained Word2Vec model in {path}')
        model.save(path)
        return model.wv

def prepare_keywords(text):
    """ Converts the given comma-separated string of keywords/phrases into cleaned up list of lists,
        where each list is either a single word or a list of words in a phrase 
        (with junk and stopwords removed)
    """
    tokens = []

    if text == None:
        return tokens
    
    text = text.split(',')
    lists = [x.split(' ') for x in text]

    for phrase in lists:
        phrase = [clean_text(t) for t in phrase]
        phrase = [t for t in phrase if not t in stopwords.words('english')]  # Remove stopwords
        phrase = ["" if t.isdigit() else t for t in phrase]  # Remove digits
        phrase = [t for t in phrase if len(t) > 1]  # Remove short tokens

        if phrase:
            tokens.append(phrase)

    return tokens

def keywords_to_vectors(words, wv, print_misses=True):
    """ Converts a comma-separated string of words/phrases to a word vector. Returns the aggregated vector, as well
        as a dict which reports for each word whether or not it was found in the Word2Vec model provided.

        Multi-word phrases provided by annotators (such as "assuage his ego") are averaged into a single word vector. 
        This isn't the BEST way to do this since you lose some semantic content of the phrase, but it works for now.
    """
    words = prepare_keywords(words)

    # empty word list
    if not words:
        return np.zeros(300), {}

    in_model = {}

    vecs = []
    # single words are converted to vectors
    # phrases are summed and averaged into vectors
    for item in words:
        # single word, like item == ['gun']
        if len(item) == 1:
            try:
                vecs.append(wv[item[0]])
                in_model[item[0]] = True
            except KeyError:
                if print_misses:
                    print(f'[single] "{item[0]}" not in word embedding model, skipping.')
                in_model[item[0]] = False
        # phrase, like item == ['democratic, 'socialist']
        else:
            phrase_vec = []

            for word in item:
                try:
                    phrase_vec.append(wv[word])
                    in_model[word] = True
                except KeyError:
                    if print_misses:
                        print(f'[phrase] "{word}" not in word embedding model, skipping.')
                    in_model[word] = False

            if phrase_vec:
                # average out the multiple vectors for the phrase into a single embedding
                phrase_vec = np.asarray(phrase_vec)
                vecs.append(phrase_vec.mean(axis=0))
   
    if len(vecs) == 0:
        return np.zeros(300), in_model

    # aggregate the vectors into a single vector.
    vecs = np.asarray(vecs)
    vec = vecs.mean(axis=0)

    assert vec.size == 300

    return vec, in_model