# samples a subset of the MBIC dataset for use with our HIT for INF385T course project
#
# - downloaded original MBIC dataset, 
# - separated into three sheets (based on article polarity), 
# - used https://kinoar.github.io/xlsx-to-json/ to convert to json
import json, random
from math import ceil

N = 150 # total number of random samples, split evenly between the three article polarities (right, left, center)
n = ceil(N/3)
types = ['left', 'center', 'right']

FILTER_MBIC_ANNOTATIONS = False # sets whether or not annotations from the MBIC study are filtered out


with open('./filtered_labeled_dataset.json') as f:
    mbic = json.load(f)
    random_subset = []

    for t in types:
        subset_t = mbic[t]

        # sample without replacement n times from the type subset of mbic
        for _ in range(0, n):
            rand = random.randint(0, len(subset_t)-1)
            sample = subset_t[rand]

            if FILTER_MBIC_ANNOTATIONS:
                for k in ['group_id', 'num_sent', 'Label_bias', 'Label_opinion', 'biased_words4']:
                    del sample[k]

            random_subset.append(sample)

            del subset_t[rand]

with open('./mbic_subset_dataset.manifest', 'w') as f:
    for item in random_subset:
        # replace newlines with html line breaks, replace unicode quotes with escape characters
        tmp = item['sentence'].replace('\n', '<br>')
        tmp2 = tmp.replace('"', r'\"')
        tmp3 = tmp2.replace('“', r'\"')
        article = tmp3.replace('”', r'\"')

        outstr = f'{{"source": "{article}", "type": "{item["type"]}"}}\n'
        f.write(outstr)