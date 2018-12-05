# Utilizing Natural Language Processing on Unstructured Clinical Notes to Predict Hospital Readmissions

## The Problem

The Affordable Care Act established the Hospital Readmission Reduction Program to incentivize improved patient outcomes by lowering payments to hospitals with excessive readmission rates. Predicting unplanned readmission may help to identify those patients at higher risk for readmission, allowing healthcare professionals to provide focused attention on their particular risk factors. This would potentially help to improve patient care, increase reimbursement rates, and decrease excessive healthcare spending associated with preventable hospitalization.

## The Data

This data for this project is text-based and will be collected from the [MIMIC-III Critical Care Database](https://mimic.physionet.org). In particular, text data will be extracted from the following tables:

- ADMISSIONS: contains data on 58,976 patient admissions. This table will provide data on hospital readmissions, including the type of admission (elective vs emergency or urgent) and the time in between admissions.
- NOTEEVENTS: contains 2,083,180 patient notes. This is the text data that will be analyzed. As indicated by the difference in dataset size, there are multiple notes for each patient, which are categorized. I will attempt to use all of the notes for each patient admission by concatenating the text fields.

.This data will need to be cleaned for known errors (marked as such in NOTEEVENTS) and unknown errors (e.g. two different severity ratings for a single measurement as a result of templates). The datasets will also need to be reshaped and concatenated 

## The Approach

At least two natural language processing approaches will be used to predict readmission. The results of the two approached will be compared for accuracy, interpretability, and computational expense. First, a traditional approach such as term frequency vectorization with logistic regression will be used. A deep learning approach such as LSTM or ULMFiT will also be attempted to see if more recent advancements in NLP are more effective.

## Deliverables

I will provide the client with a report and slide deck including a description of the process, findings, and recommended next steps. I will also publish a blog post to outline the process and outcome of the project. All code and data used in the project will be provided in a GitHub repository. 