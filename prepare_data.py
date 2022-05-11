""" This script prepares the raw data from MBIC and our free-form rationale annotations
    into two consistent datasets with the following fields:

    annotator, text, text_hash, label, gold_label, [rationale-keywords OR rationale-freeform], [class-1, class-2]
"""

import pandas as pd
import re
import json
import hashlib

df_a_mbic = pd.read_json('./mbic/annotations_cleaned.json')
# df_a_free = pd.read_json('./mbic/freeform/coded_consolidated_annotations.json')
df_a_free = pd.read_json('./mbic/freeform/coded_consolidated_annotations_newclasses.json')

# use expert labels from MBIC/BABE as gold label
df_labels = pd.read_csv('./mbic/final_labels_SG1.csv', delimiter=';')

df_mbic_prepared = pd.DataFrame()
df_free_prepared = pd.DataFrame()

def hash_text(text):
    text = str(text).lower()                    # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)       # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)            # Remove multiple spaces in content
    text = text.strip(" ")                      # Strip single spaces
    text = re.sub(r"\w+…|…", "", text)          # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)  # Remove dash between words
    text = re.sub(r'[^\w\-\s]', "", text)       # Remove punctuation

    hasher = hashlib.md5(text.encode())

    return hasher.hexdigest()

# first generate a hash of each sentence in the labels and annotations so we can
# assign them a consistent identifier
df_labels['text_hash'] = df_labels['text'].apply(lambda x : hash_text(x))

# df_prepared needs the following:
# annotator, text, label, gold_label, rationale-keywords, rationale-freeform

df_mbic_prepared['annotator'] = df_a_mbic['survey_record_id']
df_mbic_prepared['text'] = df_a_mbic['text']
df_mbic_prepared['text_hash'] = df_a_mbic['text'].apply(lambda x : hash_text(x))
df_mbic_prepared['label'] = df_a_mbic['label']

def find_gold_labels(src_df, dest_df):
    """ Finds the gold label in src_df (the MBIC labels dataset) by matching the
        hashed text (text_hash column) in the dest_df. Returns a list of gold labels.
    """
    gold_labels = []

    for _, row in dest_df.iterrows():
        gold_label = src_df.loc[src_df['text_hash'] == row['text_hash']]['label_bias']

        if gold_label.any():
            gold_labels.append(gold_label.item())
        else:
            print(f'{row}')
            print(f'Warning: no match found. Prepared table may be misaligned.')
    
    return gold_labels

df_mbic_prepared['gold_label'] = find_gold_labels(df_labels, df_mbic_prepared)
df_mbic_prepared['rationale-keywords'] = df_a_mbic['words']

# save result to a json file
with open('./prepared_mbic_annotations.json', 'w') as f:
    parsed = json.loads(df_mbic_prepared.to_json(orient="records"))
    json.dump(parsed, f, indent=4)

# now prepare the dataframe for the freeform annotations
df_free_prepared['annotator'] = df_a_free['workerId']
df_free_prepared['text'] = df_a_free['text']
df_free_prepared['text_hash'] = df_a_free['text'].apply(lambda x : hash_text(x))
df_free_prepared['label'] = df_a_free['label'].apply(lambda x : "Non-biased" if x == "Unbiased" else x)

df_free_prepared['gold_label'] = find_gold_labels(df_labels, df_free_prepared)
df_free_prepared['rationale-freeform'] = df_a_free['rationale']
df_free_prepared['class-1'] = df_a_free['Class-1'].apply(lambda x : None if x == "" else x)
df_free_prepared['class-2'] = df_a_free['Class-2'].apply(lambda x : None if x == "" else x)

# save result to json file
with open('./prepared_free_annotations.json', 'w') as f:
    parsed = json.loads(df_free_prepared.to_json(orient="records"))
    json.dump(parsed, f, indent=4)