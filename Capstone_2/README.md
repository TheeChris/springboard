# Using natural language processing on clinical notes to predict hospital readmission

30-day hospital readmissions have been targeted as a key metric of patient care. In 2012, the Affordable Care Act initiated the Hospital Readmission Reduction Program (HRRP) to incentivize improved patient outcomes by financially penalizing hospitals with excessive readmission rates. According to the [American Hospital Association](https://www.aha.org/other-resources/2016-01-18-aha-fact-sheet-hospital-readmissions-reduction-program), in the first five years of the HRRP, hospitals experienced $1.9 billion in penalties. This project uses 283,208 clinical notes (nursing and discharge summary) on 35,779 patient admissions from the [MIMIC-III database](https://mimic.physionet.org). Various natural language processing models were assessed in their ability to predict all-cause 30-day readmission for all patients (excluding neonates), including word embeddings and pre-trained language models. A bag-of-words approach using a Random Forest proved to be the most accurate with a ROC-AUC of 0.7076 and Recall of 0.7145. Prediction accuracy levels could possibly be improved through further feature engineering and building a language model more specific to clinical notes.

**Data Source**

[MIMIC-III v1.4](https://mimic.physionet.org/)

Table of Contents
------------

1. [notebooks](notebooks)
   1. [DataPrep](notebooks/0.1-TheeChris-DataPrep.ipynb): collecting and cleaning data
   2. [DataExploration](notebooks/1.1-TheeChris-DataExploration.ipynb): examining trends in the data
   3. [Logistic Regression](notebooks/2.1-TheeChris-ModelLogisticReg.ipynb): bag-of-words models using logistic regressions
   4. [Word2Vec](notebooks/2.2_TheeChris_ModelWord2Vec.ipynb): word embeddings model
   5. [ULMFiT](notebooks/2.3_TheeChris_ModelULMFiT.ipynb): pre-trained language model
   6. [Random Forest](notebooks/2.4_TheeChris_ModelRandomForest.ipynb): bag-of-words models using random forest
   7. [SVM](notebooks/2.5_TheeChris_ModelSVM.ipynb): bag-of-words models using support vector machine
2. [reports](reports)
   1. [Figures](reports/figures): all saved plot outputs 
   2. [Final Report](reports/Capstone_2_Report.pdf): a summary of the project process and results
   3. [Milestone Report 1](reports/Capstone2_Milestone_Report.pdf): a summary of data preparation and exploration
   4. [Milestone Report 2](reports/Capstone2_Milestone_Report_2.pdf): a summary of the initial logistic regression model with over- and under-sampling
3. [src](src): source code for modules used in the project



**Latent Dirichlet Allocation Topic Modelling**

![LDA Topics](C:\Users\echri\Desktop\springboard\Capstone_2\reports\figures\lda_topic_distro.png)