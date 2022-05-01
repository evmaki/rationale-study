import pandas as pd
import random
import csv
import statistics
import re
import string

from os.path import exists

from gensim.models import Word2Vec
import gensim.downloader as api
import numpy as np

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

from matplotlib import pyplot
import matplotlib.mlab as mlab

# df_annotators = pd.read_csv('./mbic/annotators.csv')
df_annotations = pd.read_json('./mbic/annotations_cleaned.json')
df_ff_annotations = pd.read_json('./mbic/freeform/coded_consolidated_annotations.json')
df_labels = pd.read_json('./mbic/labeled_dataset.json')

# nltk.download('stopwords')

def compute_agreement(num_agreed, num_annotations):
    return (num_agreed-num_annotations)/num_annotations

def compute_annotator_agreement():
    """ Returns an ordered list ranking each of the annotators by their agreement with the final label 
        Also outputs the results to agreement.csv to avoid having to rerun
    """
    agreement = {}
    annotators = df_annotations['survey_record_id'].unique()
    
    with open('./results/agreement.csv', 'w') as agreement_csv:
        writer = csv.writer(agreement_csv)
        csv_header = [
            'annotator',
            'num_annotations',
            'num_agreed_bias',
            'Label_bias_agreement',
            'num_agreed_opinion',
            'Label_opinion_agreement'
        ]
        writer.writerow(csv_header)

        # for each unique annotator in the annotations
        for annotator in annotators:
            # grab that annotator's annotations in a dataframe
            df_annotator_annotations = df_annotations.loc[df_annotations['survey_record_id'] == annotator]
            agreed_bias = 0
            agreed_opinion = 0

            # look at each of the annotator's annotations and add to the count if it matches the true label
            for _, row in df_annotator_annotations.iterrows():
                predicted_bias = row['label']
                actual_bias = df_labels.loc[df_labels['sentence'] == row['text']]['Label_bias'].item()

                if predicted_bias == actual_bias:
                    agreed_bias += 1

                predicted_opinion = row['factual']
                actual_opinion = df_labels.loc[df_labels['sentence'] == row['text']]['Label_opinion'].item()

                if predicted_opinion == actual_opinion:
                    agreed_opinion += 1

            num_annotations = len(df_annotator_annotations.index)
            # compute the annotator's agreement as predicted - actual over the total number of annotations
            agreement[annotator] = {
                'Label_bias_agreement': compute_agreement(agreed_bias, num_annotations),
                'Label_opinion_agreement': compute_agreement(agreed_opinion, num_annotations)
            }
            csv_row = [
                annotator,
                num_annotations,
                agreed_bias,
                agreement[annotator]['Label_bias_agreement'],
                agreed_opinion,
                agreement[annotator]['Label_opinion_agreement']
            ]
            writer.writerow(csv_row)
    
    return agreement

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
            compute_agreement(individual_agreed_bias, num_annotations_from_annotator),
            compute_agreement(individual_agreed_opinion, num_annotations_from_annotator)
        )

    # compute the group's overall agreement as predicted - actual over the total number of annotations
    agreement = {
        'Label_bias_agreement': compute_agreement(agreed_bias, num_annotations),
        'Label_opinion_agreement': compute_agreement(agreed_opinion, num_annotations),
        'individual_agreement': individual_agreement
    }

    return agreement

def report_cluster_agreement(df):
    """ Reports the agreement of each cluster in the given dataframe """

    for cluster in df['cluster'].unique():
        annotators = df.loc[df['cluster'] == cluster]['survey_record_id'].unique()
        cluster_agreement = subset_agreement(0, annotators=annotators)

        individual_bias_agreements = [x for (x, _) in cluster_agreement['individual_agreement'].values()]
        individual_opinion_agreements = [y for (_, y) in cluster_agreement['individual_agreement'].values()]

        bias_variance = statistics.variance(individual_bias_agreements)
        opinion_variance = statistics.variance(individual_opinion_agreements)

        print(f'Cluster:\t\t{cluster}, N = {len(annotators)}')
        print(f'Agreement:\t\t{cluster_agreement["Label_bias_agreement"]:.4f}')
        print(f'> Variance:\t\t{bias_variance:.4f}\n')
        # print(f'Opinion label agreement:\t{cluster_agreement["Label_opinion_agreement"]:.4f}')
        # print(f'> Variance:\t\t{opinion_variance:.4f}\n')

        print(f'Random subsets')
        n_random_subsets(10, len(annotators))

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

    print(label_bias_agreement)
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
    #text = re.sub(r"(?<=\w)-(?=\w)", "", text)  # Remove dash between words
    text = re.sub(r'[^\w\-\s]', "", text)       # Remove punctuation

    return text

def train_word2vec(retrain=False):
    if exists('./results/mbic-article-word2vec.model') and not retrain:
        return Word2Vec.load('./results/mbic-article-word2vec.model')

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

    model = Word2Vec(corpus, vector_size=300)
    model.save('./results/mbic-article-word2vec.model')
    return model

in_model = 0
notin_model = 0

unknown_words = []

def words_to_vectors(words, wv):
    """ Converts a comma-separated string of words/phrases to a padded list of word vectors.
        Padding length has been precomputed - the longest list of words/phrases provided by an annotator 
        in MBIC is 12 long, so all lists are padded to length 12 with zeroed out vectors to make the features 
        consistent for clustering.

        Multi-word phrases provided by annotators (such as "assuage his ego") are averaged into a single word vector. 
        This isn't the BEST way to do this since you lose some semantic content of the phrase, but it works for now.
    """
    # empty word list
    if not words:
        return np.zeros(300)

    global in_model, notin_model, unknown_words

    vecs = []
    # single words are converted to vectors
    # phrases are summed and averaged into vectors
    for item in words:
        # single word, like item == ['gun']
        if len(item) == 1:
            try:
                vecs.append(wv[item[0]])
                in_model += 1
            except KeyError:
                print(f'[single] "{item[0]}" not in word embedding model, skipping.')
                
                if item[0] not in unknown_words:
                    unknown_words.append(item[0])
                notin_model += 1
        # phrase, like item == ['democratic, 'socialist']
        else:
            phrase_vec = []

            for word in item:
                try:
                    phrase_vec.append(wv[word])
                    in_model += 1
                except KeyError:
                    print(f'[phrase] "{word}" not in word embedding model, skipping.')
                    
                    if item not in unknown_words:
                        unknown_words.append(item)
                    notin_model += 1

            if phrase_vec:
                # average out the multiple vectors for the phrase into a single embedding
                phrase_vec = np.asarray(phrase_vec)
                vecs.append(phrase_vec.mean(axis=0))
   
    if len(vecs) == 0:
        vecs.append(np.zeros(300))

    # flatten the vectors into a single vector.
    vecs = np.asarray(vecs)
    vec = vecs.mean(axis=0)

    return vec

# source: https://dylancastillo.co/nlp-snippets-cluster-documents-using-word2vec/
def mbkmeans_clusters(X, k, mb):
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

    return km, km.labels_

def birch_clusters(X, k): 
    b = Birch(n_clusters=k).fit(X)
    return b, b.labels_

def plot_keyword_clusters(df_clusters, title=''):
    # grab each unique cluster in the dataframe
    clusters = df_clusters['cluster'].unique()

    # project the high-dimensional word vectors to a lower-dimensional space (for plotting)
    pca = PCA(3)
    svd_word_embeddings = pca.fit_transform(list(df_clusters['word_embeddings'].values))

    df_clusters['svd_x'] = [x for [x, _, _] in svd_word_embeddings]
    df_clusters['svd_y'] = [y for [_, y, _] in svd_word_embeddings]
    df_clusters['svd_z'] = [z for [_, _, z] in svd_word_embeddings]

    fig = pyplot.figure()
    ax = fig.add_subplot(projection='3d')

    # plot each cluster separately in a different color
    for cluster in clusters:
        xs = df_clusters.loc[df_clusters['cluster'] == cluster]['svd_x'].values
        ys = df_clusters.loc[df_clusters['cluster'] == cluster]['svd_y'].values
        zs = df_clusters.loc[df_clusters['cluster'] == cluster]['svd_z'].values

        ax.scatter(xs, ys, zs, label=f'{cluster}')
    
    pyplot.title(title)
    pyplot.legend()
    pyplot.show()

def plot_freeform_clusters(df_clusters, title=''):
    # grab each unique cluster in the dataframe
    clusters = df_clusters['cluster'].unique()

    df_clusters['x'] = [x for [x, _] in df_clusters['features']]
    df_clusters['y'] = [y for [_, y] in df_clusters['features']]
    # df_clusters['z'] = [z for [_, _, z] in df_clusters['features']]

    fig = pyplot.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()

    # plot each cluster separately in a different color
    for cluster in clusters:
        xs = df_clusters.loc[df_clusters['cluster'] == cluster]['x'].values
        ys = df_clusters.loc[df_clusters['cluster'] == cluster]['y'].values
        # zs = df_clusters.loc[df_clusters['cluster'] == cluster]['z'].values

        # ax.scatter(xs, ys, zs, label=f'{cluster}')
        ax.scatter(xs, ys, label=f'{cluster}')
    
    pyplot.title(title)
    pyplot.legend()
    pyplot.show()

def plot_freeform_histogram(df_clusters):
    fig, ax = pyplot.subplots()
    data = df_clusters['features'].apply(lambda x : str(x))

    print(data)

    N, bins, patches = ax.hist(data, edgecolor='white', linewidth=1)

    # for i in range(0,3):
    #     patches[i].set_facecolor('b')
    # for i in range(3,5):    
    #     patches[i].set_facecolor('r')
    # for i in range(5, len(patches)):
    #     patches[i].set_facecolor('black')

    pyplot.show()

def plot_cluster_histograms(df_clusters):
    for cluster in df_clusters['cluster'].unique():
        print(f'Plotting histogram of cluster {cluster} annotators...')
        plot_agreement_histogram(10, annotators=df_clusters.loc[df_clusters['cluster'] == cluster]['survey_record_id'].unique())

def report_cluster_keywords(wv, clustering, num_clusters, cluster_centers, num_words):
    """ Reports the most representative terms in the given clusters. This is a qualitative way
        of examining the types of words each cluster is formed around 
    """
    for i in range(num_clusters):
        tokens_per_cluster = ''
        most_representative = wv.most_similar(positive=[cluster_centers[i]], topn=num_words)
        
        for t in most_representative:
            tokens_per_cluster += f'{t[0]} '

        print(f'Cluster {i}: {tokens_per_cluster}')

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

def plot_agreement_histogram(num_bins=5, annotators=[]):
    if not exists('./results/agreement.csv'):
        compute_annotator_agreement()

    df = pd.read_csv('./results/agreement.csv')

    if len(annotators) == 0:
        n, bins, patches = pyplot.hist(df['Label_bias_agreement'].values, num_bins)

    else:
        xs = df[pd.DataFrame(df['annotator'].tolist()).isin(annotators).any(1).values]['Label_bias_agreement']
        n, bins, patches = pyplot.hist(xs.values, num_bins)

    pyplot.xlabel('agreement')
    pyplot.ylabel('annotator count')
    pyplot.title('Histogram of annotator agreement with gold standard label')

    pyplot.show()

def keyword_evals_main(custom_model=True):
    # train the word2vec model for generating word embeddings
    if custom_model:
        model = train_word2vec()
        wv = model.wv
    else:
        wv = api.load('word2vec-google-news-300')

    # clean up the words in the data frame by turning them into lists of lists and removing stop words
    df_annotations['words'] = df_annotations['words'].apply(lambda x : prepare_words(x))

    # map the bias keywords to word embeddings
    df_annotations['word_embeddings'] = df_annotations['words'].apply(lambda x : words_to_vectors(x, wv))

    # report how many tokens we lost since the model didn't know about them
    print(f'tokens known to model:\t\t{in_model}')
    print(f'tokens unknown to model:\t{notin_model}')
    print(f'percent unknown:\t\t{(notin_model/(notin_model+in_model))*100:.2f}%')

    def flatten(xs):
        result = []
        if isinstance(xs, (list, tuple)):
            for x in xs:
                result.extend(flatten(x))
        else:
            result.append(xs)
        return result

    flat_unknown_words = flatten(unknown_words)

    # get the annotators who used words unknown to model – what's their agreement?
    df_annotations['words'] = df_annotations['words'].apply(lambda x : flatten(x))
    unknown_word_annotators = df_annotations[pd.DataFrame(df_annotations['words'].tolist()).isin(flat_unknown_words).any(1).values]
    unknown_word_annotator_agreement = subset_agreement(0, annotators=unknown_word_annotators['survey_record_id'].unique())
    print(f'unknown word labeler agreement: {unknown_word_annotator_agreement["Label_bias_agreement"]}\n')

    word_embeddings = list(df_annotations['word_embeddings'].values)
    num_clusters = 3

    clustering, cluster_labels = birch_clusters(word_embeddings, num_clusters)

    df_clusters = pd.DataFrame({
        'text': df_annotations['words'],
        'word_embeddings': df_annotations['word_embeddings'].values,
        'cluster': cluster_labels,
        'survey_record_id': df_annotations['survey_record_id'].values
    })

    report_cluster_agreement(df_clusters)
    report_cluster_keywords(wv, clustering, num_clusters, clustering.subcluster_centers_, 10)

    plot_keyword_clusters(df_clusters, 'Clusters of keyword rationale content (Google News model)')
    # plot_cluster_histograms(df_clusters)

def freeform_subset_agreement(n):
    # strip punctuation, lowercase, and remove spaces from sentences in label df and annotation df so that we can
    # more reliably match on the example text in each to get the gold label
    df_labels['sentence'] = df_labels['sentence'].apply(lambda x : clean_text(x))
    df_labels['sentence'] = df_labels['sentence'].apply(lambda x : x.replace(' ', ''))

    df_ff_annotations['text'] = df_ff_annotations['text'].apply(lambda x : clean_text(x))
    df_ff_annotations['text'] = df_ff_annotations['text'].apply(lambda x : x.replace(' ', ''))

    agreement = {}
    individual_agreement = {}
    annotators = df_ff_annotations['workerId'].unique()
    random_annotators = random.sample(list(annotators), n)
    
    # hold the count of agreed bias labels, plus the total number of annotations for the group
    agreed_bias = 0
    num_annotations = 0
    # for each unique annotator in the annotations
    for annotator in random_annotators:
        # grab that annotator's annotations in a dataframe
        df_annotator_annotations = df_ff_annotations.loc[df_ff_annotations['workerId'] == annotator]
        individual_agreed_bias = 0
        # look at each of the annotator's annotations and add to the count if it matches the true label
        for _, row in df_annotator_annotations.iterrows():
            predicted_bias = row['label']

            actual_bias = df_labels.loc[df_labels['sentence'] == row['text']]['Label_bias'].item()

            if predicted_bias == actual_bias:
                individual_agreed_bias += 1
                agreed_bias += 1

        num_annotations_from_annotator = len(df_annotator_annotations.index)
        num_annotations += num_annotations_from_annotator

        individual_agreement[annotator] = (
            compute_agreement(individual_agreed_bias, num_annotations_from_annotator)            
        )

    # compute the group's overall agreement as predicted - actual over the total number of annotations
    agreement = {
        'Label_bias_agreement': compute_agreement(agreed_bias, num_annotations),
        'individual_agreement': individual_agreement
    }

    return agreement['Label_bias_agreement']

def report_freeform_cluster_agreement(df_clusters):
    clusters = df_clusters['cluster'].unique()

    # strip punctuation, lowercase, and remove spaces from sentences in label df and annotation df so that we can
    # more reliably match on the example text in each to get the gold label
    df_labels['sentence'] = df_labels['sentence'].apply(lambda x : clean_text(x))
    df_labels['sentence'] = df_labels['sentence'].apply(lambda x : x.replace(' ', ''))

    df_clusters['text'] = df_clusters['text'].apply(lambda x : clean_text(x))
    df_clusters['text'] = df_clusters['text'].apply(lambda x : x.replace(' ', ''))

    agreements = {}

    for cluster in clusters:
        annotations = df_clusters.loc[df_clusters['cluster'] == cluster]
        
        num_annotations = annotations.shape[0]
        num_agreed = 0
        agreements = []
        
        for _, annotation in annotations.iterrows():
            gold_label = df_labels.loc[df_labels['sentence'] == annotation['text']]['Label_bias']
            
            if gold_label.empty:
                print('Warning: no gold label found in labeled dataset. Strings don\'t match.')
            
            if annotation['label'] == gold_label.item():
                num_agreed += 1

        print(f'Cluster:\t\t{cluster}, N = {len(annotations["workerId"].unique())}')
        print(f'Agreement:\t\t{compute_agreement(num_agreed, num_annotations):.4f}')
        #print(f'> Variance:\t\t{bias_variance:.4f}\n')

        random_subset_agreements = [freeform_subset_agreement(len(annotations["workerId"].unique())) for _ in range(0, 10)]
        print(f'N = {len(annotations["workerId"].unique())}')
        print(f'Random subset: {statistics.mean(random_subset_agreements):.4f}\n')

def freeform_evals_main():
    # replaces NaNs with nulls
    df_ff_annotations['Class-1'] = df_ff_annotations['Class-1'].apply(lambda x : "null" if x == '' or x == None else x)
    df_ff_annotations['Class-2'] = df_ff_annotations['Class-2'].apply(lambda x : "null" if x == '' or x == None else x)

    # replaces "Unbiased" with "Non-biased" so both datasets are consistent
    df_ff_annotations['label'] = df_ff_annotations['label'].apply(lambda x : "Non-biased" if x == "Unbiased" else x)

    # map unique classes to numeric codes
    rationale_classes = dict((v,k) for k,v in enumerate(df_ff_annotations['Class-1'].unique()))

    print(f'Classes:     {rationale_classes}')
    print(f'Annotators:  {len(df_ff_annotations["workerId"].unique())}')
    print(f'Annotations: {df_ff_annotations.shape[0]}')

    print(f'Class-1')
    for rationale_class in rationale_classes.keys():
        print(f'\t{rationale_class:>16}:\t{len(df_ff_annotations.loc[df_ff_annotations["Class-1"] == rationale_class].values)}')
    
    print(f'Class-2')
    for rationale_class in rationale_classes.keys():
        print(f'\t{rationale_class:>16}:\t{len(df_ff_annotations.loc[df_ff_annotations["Class-2"] == rationale_class].values)}')

    # create a numerical feature vector for each annotation
    df_ff_annotations['Class-1'] = df_ff_annotations['Class-1'].apply(lambda x : rationale_classes[x])
    df_ff_annotations['Class-2'] = df_ff_annotations['Class-2'].apply(lambda x : rationale_classes[x])
    df_ff_annotations['rationale_length'] = df_ff_annotations['rationale'].apply(lambda x : len(x))
    # df_ff_annotations['features'] = df_ff_annotations[['Class-1', 'Class-2', 'rationale_length']].values.tolist()
    df_ff_annotations['features'] = df_ff_annotations[['Class-1', 'Class-2']].values.tolist()

    print(df_ff_annotations.head())

    num_clusters = 3

    clustering, cluster_labels = birch_clusters(list(df_ff_annotations['features'].values), num_clusters)

    df_clusters = pd.DataFrame({
        'features': df_ff_annotations['features'].values,
        'cluster': cluster_labels,
        'workerId': df_ff_annotations['workerId'].values,
        'text': df_ff_annotations['text'].values,
        'label': df_ff_annotations['label'].values
    })

    report_freeform_cluster_agreement(df_clusters)
    plot_freeform_clusters(df_clusters, 'Clusters of freeform rationale content')
    #plot_freeform_histogram(df_clusters)

freeform_evals_main()
# keyword_evals_main()
# plot_agreement_histogram(18)