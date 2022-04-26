import pandas as pd
import random
import csv
import statistics
import re
import string

from gensim.models import Word2Vec
import numpy as np
from functools import reduce

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

from matplotlib import pyplot

# df_annotators = pd.read_csv('./mbic/annotators.csv')
df_annotations = pd.read_json('./mbic/annotations_cleaned.json')
df_labels = pd.read_json('./mbic/labeled_dataset.json')
df_agreement = pd.read_csv('./agreement.csv')

regex = r'[^\w\-\s]'

# nltk.download('stopwords')

# def compute_agreement():
#     """ Returns an ordered list ranking each of the annotators by their agreement with the final label 
#         Also outputs the results to agreement.csv to avoid having to rerun
#     """
#     agreement = {}
#     annotators = df_annotations['survey_record_id'].unique()
    
#     with open('./agreement.csv', 'w') as agreement_csv:
#         writer = csv.writer(agreement_csv)
#         csv_header = [
#             'annotator',
#             'num_annotations',
#             'num_agreed_bias',
#             'Label_bias_agreement',
#             'num_agreed_opinion',
#             'Label_opinion_agreement'
#         ]
#         writer.writerow(csv_header)

#         # for each unique annotator in the annotations
#         for annotator in annotators:
#             # grab that annotator's annotations in a dataframe
#             df_annotator_annotations = df_annotations.loc[df_annotations['survey_record_id'] == annotator]
#             agreed_bias = 0
#             agreed_opinion = 0

#             # look at each of the annotator's annotations and add to the count if it matches the true label
#             for _, row in df_annotator_annotations.iterrows():
#                 predicted_bias = row['label']
#                 actual_bias = df_labels.loc[df_labels['sentence'] == row['text']]['Label_bias'].item()

#                 if predicted_bias == actual_bias:
#                     agreed_bias += 1

#                 predicted_opinion = row['factual']
#                 actual_opinion = df_labels.loc[df_labels['sentence'] == row['text']]['Label_opinion'].item()

#                 if predicted_opinion == actual_opinion:
#                     agreed_opinion += 1

#             num_annotations = len(df_annotator_annotations.index)
#             # compute the annotator's agreement as predicted - actual over the total number of annotations
#             agreement[annotator] = {
#                 'Label_bias_agreement': (agreed_bias-num_annotations)/num_annotations,
#                 'Label_opinion_agreement': (agreed_opinion-num_annotations)/num_annotations
#             }
#             csv_row = [
#                 annotator,
#                 num_annotations,
#                 agreed_bias,
#                 agreement[annotator]['Label_bias_agreement'],
#                 agreed_opinion,
#                 agreement[annotator]['Label_opinion_agreement']
#             ]
#             writer.writerow(csv_row)
    
#     return agreement

def subset_agreement(n, annotators=[]):
    """ Returns the agreement of n annotators and the true label. Chooses randomly if a list of annotators is not provided. """
    agreement = {}
    individual_agreement = {}

    # no annotators provided; do a random sampling
    if len(annotators) == 0:
        annotators = df_annotations['survey_record_id'].unique()
        random_annotators = random.sample(list(annotators), n)
    # use the passed list of annotators
    else:
        random_annotators = annotators
        
    # hold the count of agreed bias and opinion labels, plus the total number of annotations for the group
    agreed_bias = 0
    agreed_opinion = 0
    num_annotations = 0

    # for each unique annotator in the annotations
    for annotator in random_annotators:
        # grab that annotator's annotations in a dataframe
        df_annotator_annotations = df_annotations.loc[df_annotations['survey_record_id'] == annotator]

        individual_agreed_bias = 0
        individual_agreed_opinion = 0

        # look at each of the annotator's annotations and add to the count if it matches the true label
        for _, row in df_annotator_annotations.iterrows():
            predicted_bias = row['label']
            actual_bias = df_labels.loc[df_labels['sentence'] == row['text']]['Label_bias'].item()

            if predicted_bias == actual_bias:
                individual_agreed_bias += 1
                agreed_bias += 1

            predicted_opinion = row['factual']
            actual_opinion = df_labels.loc[df_labels['sentence'] == row['text']]['Label_opinion'].item()

            if predicted_opinion == actual_opinion:
                individual_agreed_opinion += 1
                agreed_opinion += 1

        num_annotations_from_annotator = len(df_annotator_annotations.index)
        num_annotations += num_annotations_from_annotator

        individual_agreement[annotator] = (
            (individual_agreed_bias-num_annotations_from_annotator)/num_annotations_from_annotator, 
            (individual_agreed_opinion-num_annotations_from_annotator)/num_annotations_from_annotator
        )

    # compute the group's overall agreement as predicted - actual over the total number of annotations
    agreement = {
        'Label_bias_agreement': (agreed_bias-num_annotations)/num_annotations,
        'Label_opinion_agreement': (agreed_opinion-num_annotations)/num_annotations,
        'individual_agreement': individual_agreement
    }

    return agreement

def report_cluster_agreement(df):
    """ Reports the agreement of each cluster in the given dataframe """

    for cluster in df['cluster'].unique():
        annotators = df.loc[df['cluster'] == cluster]['survey_record_id'].unique()
        cluster_agreement = subset_agreement(0, annotators=annotators)

        # f = {
        #     'individual_agreement': {
        #         annotator: (label_bias, label_opinion)
        #     }
        # }

        individual_bias_agreements = [x for (x, _) in cluster_agreement['individual_agreement'].values()]
        individual_opinion_agreements = [y for (_, y) in cluster_agreement['individual_agreement'].values()]

        bias_variance = statistics.variance(individual_bias_agreements)
        opinion_variance = statistics.variance(individual_opinion_agreements)

        print(f'Cluster:\t\t{cluster}, N = {len(annotators)}')
        print(f'Bias label agreement:\t\t{cluster_agreement["Label_bias_agreement"]:.4f}')
        print(f'> Variance:\t\t{bias_variance:.4f}')
        print(f'Opinion label agreement:\t{cluster_agreement["Label_opinion_agreement"]:.4f}')
        print(f'> Variance:\t\t{opinion_variance:.4f}\n')

def n_random_subsets(N, n):
    """ Reports mean and variance agreement for N random subsets of n annotators 
        
        These values serve as a baseline. If a group of annotators is hand-picked based on
        some ancillary characteristic (rationale content or format), we should hope that they either 
        do better or worse than these values – that indicates that whatever we chose them based on may
        be a predictor of a higher or lower level of bias over random chance.
    """
    label_bias_agreement = []
    label_opinion_agreement = []

    for _ in range(0, N):
        agreement = subset_agreement(n)
        label_bias_agreement.append(agreement['Label_bias_agreement'])
        label_opinion_agreement.append(agreement['Label_opinion_agreement'])

    label_bias_variance = statistics.variance(label_bias_agreement)
    label_opinion_variance = statistics.variance(label_opinion_agreement)

    label_bias_mean = statistics.mean(label_bias_agreement)
    label_opinion_mean = statistics.mean(label_opinion_agreement)

    print(f'N = {N}, n = {n}')
    print('\tBias agreement\tOpinion agreement')
    print(f'Mean     {label_bias_mean:.4f}     \t{label_opinion_mean:.4f}')
    print(f'Variance {label_bias_variance:.4f} \t{label_opinion_variance:.4f}')

def clean_text(text):
    text = str(text).lower()                    # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)       # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)            # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)          # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)  # Remove dash between words
    text = re.sub(r'[^\w\-\s]', "", text)       # Remove punctuation

    return text

def apply_word2vec():
    # convert each sentence text to a list of individual words
    sentences = df_labels['article'].tolist()
    corpus = []

    for item in sentences:
        if item != None:
            text = clean_text(item)
            tokens = text.split(' ')

            tokens = [t for t in tokens if not t in stopwords.words('english')]  # Remove stopwords
            tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
            tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens

            corpus.append(tokens)

    return Word2Vec(corpus)

in_model = 0
notin_model = 0

def words_to_vectors(words, model, padding=12):
    """ Converts a comma-separated string of words/phrases to a padded list of word vectors.
        Padding length has been precomputed - the longest list of words/phrases provided by an annotator 
        in MBIC is 12 long, so all lists are padded to length 12 with zeroed out vectors to make the features 
        consistent for clustering.

        Multi-word phrases provided by annotators (such as "assuage his ego") are averaged into a single word vector. 
        This isn't the BEST way to do this since you lose some semantic content of the phrase, but it works for now.
    """
    # empty word list
    if not words:
        return np.zeros(100)

    global in_model, notin_model

    vecs = []
    # single words are converted to vectors
    # phrases are summed and averaged into vectors
    for item in words:
        # single word, like item == ['gun']
        if len(item) == 1:
            try:
                vecs.append(model.wv[item[0]])
                in_model += 1
            except KeyError:
                print(f'[single] "{item[0]}" not in word embedding model, skipping.')
                notin_model += 1
        # phrase, like item == ['democratic, 'socialist']
        else:
            phrase_vec = []

            for word in item:
                try:
                    phrase_vec.append(model.wv[word])
                    in_model += 1
                except KeyError:
                    print(f'[phrase] "{word}" not in word embedding model, skipping.')
                    notin_model += 1

            if phrase_vec:
                # average out the multiple vectors for the phrase into a single embedding
                phrase_vec = np.asarray(phrase_vec)
                vecs.append(phrase_vec.mean(axis=0))
   
    if len(vecs) == 0:
        vecs.append(np.zeros(100))

    # flatten the vectors into a single vector.
    vecs = np.asarray(vecs)
    vec = vecs.mean(axis=0)
    print(vec)
    assert len(vec) == 100
    return vec

# source: https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
def mbkmeans_clusters(X, k, mb, print_silhouette_values):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia: {km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_

def plot_clusters(df_clusters):
    # grab each unique cluster in the dataframe
    clusters = df_clusters['cluster'].unique()

    # project the high-dimensional word vectors to a lower-dimensional 2D space (for plotting)
    pca = PCA(2)
    svd_word_embeddings = pca.fit_transform(list(df_clusters['word_embeddings'].values))
    df_clusters['svd_x'] = [x for [x, _] in svd_word_embeddings]
    df_clusters['svd_y'] = [y for [_, y] in svd_word_embeddings]

    # plot each cluster separately in a different color
    for cluster in clusters:
        xs = df_clusters.loc[df_clusters['cluster'] == cluster]['svd_x'].values
        ys = df_clusters.loc[df_clusters['cluster'] == cluster]['svd_y'].values
        pyplot.scatter(xs, ys)
    
    pyplot.show()

def main():
    # train the word2vec model for generating word embeddings
    model = apply_word2vec()

    # clean up the words in the data frame by turning them into lists of lists and removing stop words
    df_annotations['words'] = df_annotations['words'].apply(lambda x : prepare_words(x))

    # map the bias keywords to word embeddings
    df_annotations['word_embeddings'] = df_annotations['words'].apply(lambda x : words_to_vectors(x, model))

    # report how many tokens we lost since the model didn't know about them
    print(f'rationale tokens known to model:\t{in_model}')
    print(f'rationale tokens unknown to model:\t{notin_model}')
    print(f'percent unknown:\t\t\t{(notin_model/(notin_model+in_model))*100:.2f}%')

    word_embeddings = list(df_annotations['word_embeddings'].values)
    num_clusters = 5
    num_batches = 500
    clustering, cluster_labels = mbkmeans_clusters(word_embeddings, num_clusters, num_batches, False)

    df_clusters = pd.DataFrame({
        'text': df_annotations['words'],
        'word_embeddings': df_annotations['word_embeddings'].values,
        'cluster': cluster_labels,
        'survey_record_id': df_annotations['survey_record_id'].values
    })

    report_cluster_agreement(df_clusters)
    plot_clusters(df_clusters)

def prepare_words(text):
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

main()

# TODO
# 1 improve the training and tokenization of words (more filters, remove stop words, remove junk)
# X 2 write function to calculate agreement w/ chosen groups (by cluster) and gold standard