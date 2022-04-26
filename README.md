# rationale-study
Dataset, task design, and evaluation code for INF385T Human Computation project, studying effect of rationale on annotator bias.

## /mbic/sample_random_subset.py
Takes N random samples evenly distributed across three article types (left, center, right) in MBIC and saves it as a .manifest for use in SageMaker Ground Truth

## /mbic/filtered_labeled_dataset.json
A copy of the MBIC dataset, converted to JSON with articles organized by type (left, center, right)

## hit_template.html
Template of the custom task design for use on SageMaker Ground Truth
