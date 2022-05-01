import pandas as pd
import word2vec
import statistics

from sklearn.cluster import Birch
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

#
#   CLUSTERING
#

def birch_clusters(X, k): 
    """ Takes a list X of features and performs Birch clustering to find k clusters
    """
    print(f'Birch clustering on {len(X)} features for k = {k}')
    b = Birch(n_clusters=k).fit(X)
    return b, b.labels_

#
#   ERROR CALCULATION
#

def compute_error(num_agreed, num_annotations):
    return (num_agreed-num_annotations)/num_annotations

def random_subset_error(df, N, n):
    """ Computes the error for a size N subset of annotations from df for n runs
    """    
    errors = []

    # over n runs...
    for _ in range(0, n):
        # ...grab a size N subset of annotations (without replacement)
        df_subset = df.sample(n=N)
        
        # find the number of annotations in that subset that are equal to the gold label
        num_agreed = df_subset.loc[df_subset['label'] == df_subset['gold_label']].shape[0]
        
        # add the result for this run to the list of results
        errors.append(compute_error(num_agreed, N))

    return errors

def cluster_error(df):
    """ Computes the error for annotations in each cluster
    """
    errors = {}

    # for each cluster...
    for cluster in df['cluster'].unique():
        # ...grab the annotations from the current cluster
        df_cluster = df.loc[df['cluster'] == cluster]

        # find the number of annotations in that cluster that are equal to the gold label
        num_agreed = df_cluster.loc[df_cluster['label'] == df_cluster['gold_label']].shape[0]

        # add the result for this run to the dict which stores cluster -> error mappings
        errors[cluster] = compute_error(num_agreed, df_cluster.shape[0])

    return errors

def class_error(df):
    """ Computes the average error for each class-1 group of annotations
    """
    errors = {}

    # for each unique class...
    for rationale_class in df['class-1'].unique():
        # ...grab the group of annotations in that class
        df_group = df.loc[df['class-1'] == rationale_class]

        # find the number of annotations in that group that are equal to the gold label
        num_agreed = df_group.loc[df_group['label'] == df_group['gold_label']].shape[0]

        # add the result for this run to the dict which stores cluster -> error mappings
        errors[rationale_class] = compute_error(num_agreed, df_group.shape[0])

    return errors

#
# PLOTS
#

def plot_keyword_clusters(df, title=''):
    # grab each unique cluster in the dataframe
    clusters = df['cluster'].unique()

    # project the high-dimensional features to a lower-dimensional space (for plotting)
    pca = PCA(3)
    svd_features = pca.fit_transform(list(df['features'].values))

    df['svd_x'] = [x for [x, _, _] in svd_features]
    df['svd_y'] = [y for [_, y, _] in svd_features]
    df['svd_z'] = [z for [_, _, z] in svd_features]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # plot each cluster separately in a different color
    for cluster in clusters:
        xs = df.loc[df['cluster'] == cluster]['svd_x'].values
        ys = df.loc[df['cluster'] == cluster]['svd_y'].values
        zs = df.loc[df['cluster'] == cluster]['svd_z'].values

        ax.scatter(xs, ys, zs, label=f'{cluster}')
    
    plt.title(title)
    plt.legend()
    plt.show()

def plot_class_histogram(df, rationale_classes):
    codes = df['class-1'].unique()

    series = [(lambda x : df.loc[df['class-1'] == x].shape[0])(x) for x in codes]

    #fig = plt.figure(figsize=(4,4))

    plt.bar([rationale_classes[x] if rationale_classes[x] is not None else 'None' for x in codes], series)
    plt.xticks(rotation=45)
    plt.tight_layout(pad=2.0)
    plt.title('Count of annotations in by class')
    plt.show()

#
#   MAIN EVALS
#

def keyword_evaluations():
    df = pd.read_json('./prepared_mbic_annotations.json')

    # pick either the google model or our custom model trained the articles in MBIC
    word2vec_model = 'word2vec-google-news-300' # 'custom'
    print(f'Using Word2Vec model \"{word2vec_model}\"')

    # get the learned word embeddings from the word2vec model
    wv = word2vec.get_model_wv(word2vec_model)

    # compute the feature vector each annotation
    df['features'] = df['rationale-keywords'].apply(
        lambda x : word2vec.keywords_to_vectors(x, wv, print_misses=False)[0]
    )
    
    # do clustering and add the labels to the dataframe
    X = df['features'].tolist()
    k = 3

    clustering, cluster_labels = birch_clusters(X, k)
    df['cluster'] = cluster_labels

    # compute the error for each cluster
    cluster_errors = cluster_error(df)

    # compute and report the random error for subsets equal in size to each cluster
    for cluster in df['cluster'].unique():
        N = df.loc[df['cluster'] == cluster].shape[0]
        print('-'*100)
        print(f'Cluster: {cluster}, N = {N}')
        print(f'Mean: {cluster_errors[cluster]:.4f}')
        print('')

        n = 10
        print(f'Random subset error for N = {N}, n = {n} runs')
        random_errors = random_subset_error(df, N, n)
        print(f'Mean: {statistics.mean(random_errors):.4f}')

    plot_keyword_clusters(df)

def freeform_evaluations():
    df = pd.read_json('./prepared_free_annotations.json')

    # map each class to a numeric code
    rationale_codes = dict((v,k) for k,v in enumerate(df['class-1'].unique()))
    rationale_classes = dict((v,k) for k,v in rationale_codes.items())

    # map class of each annotation to the corresponding class code
    df['class-1'] = df['class-1'].apply(lambda x : rationale_codes[x])
    df['class-2'] = df['class-2'].apply(lambda x : rationale_codes[x])

    # compute the feature vector for each annotation
    # TODO more features?
    df['features'] = df[['class-1', 'class-1']].values.tolist()

    # compute the error for each class-1 group
    class_errors = class_error(df)

    # report error by class vs random error
    for rationale_code in df['class-1'].unique():
        N = df.loc[df['class-1'] == rationale_code].shape[0]
        print('-'*100)
        print(f'Class: {rationale_classes[rationale_code]}, N = {N}')
        print(f'Mean: {class_errors[rationale_code]:.4f}')
        print('')

        n = 10
        print(f'Random subset error for N = {N}, n = {n} runs')
        random_errors = random_subset_error(df, N, n)
        print(f'Mean: {statistics.mean(random_errors):.4f}')

    plot_class_histogram(df, rationale_classes)


# keyword_evaluations()
freeform_evaluations()