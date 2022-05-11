# Media Bias Including Annotator Characteristics (MBIC)
With the exception of the annotations in `/freeform/`, the data in this folder comes from a combination of [the original MBIC dataset](https://zenodo.org/record/4635121#.Ynv7npLMJQI) along with expert labels from the later [MBIC/BABE dataset](https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE), both published by Spinde et. al.

## final_labels_SG1.csv
The final gold labels from the MBIC/BABE dataset. These labels are from expert annotators.

## labeled_dataset.json
The final labels from the original MBIC dataset. These labels are from non-expert crowdworkers.

## filtered_labeled_dataset.json
The final labels from the original MBIC dataset, restructured into three top level elements "left", "right", and "center" so that we can more easily take random samplings with equal balance between each article polarity.

## annotations_cleaned.json
The raw annotations from the original MBIC dataset, with some data cleaning. Incomplete bias keywords that were missing characters are completed. Some minor differences (missing characters, UTF-8 vs. ASCII characters) have been made so that these annotations can be matched more easily to the later MBIC/BABE dataset which has expert annotations.

## sample_random_subset.py
Generates a balanced random subset of size N from `filtered_labeled_dataset.json`.

## mbic_subset_dataset.manifest
A random sampling of annotations from `filtered_labeled_dataset.json` for use in our free-form rationale HIT.

## /freeform/
The raw and consolidated annotations (plus data wrangling scripts) from our free-form rationale HIT.
