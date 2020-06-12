#  Using Gastrointestinal Distress Reports to Predict Youth Anxiety Risk: Implications for Mental Health Literacy and Community Care

Paul A. Bloom, Ian Douglas, Michelle VanTieghem, Nim Tottenham, & Bridget Callaghan

## Repository for analysis on GI items predicting anxiety

All analysis code can be found in the scripts folder. Analyses largely follow the preregistration found at https://osf.io/687ky

Manuscript currently in prep!


## Scripts Roadmap

1a: Cleans data for LA/NYC cohorts

1b: Cleans HBN KSADS data

1c: Cleans HBN data for all measures, splits HBN into train/test sets

1d: Imputes missing data values for HBN training set

2a: Replication: runs SCARED-P models across all three cohorts (NYC, LA, HBN-train). 

2b: Robustness: runs SCARED-P, SCARED-C, and KSADS models in HBN training data

3a: Cross-validation and model selection of SCARED-P models in HBN training data

3b: Cross-validation and model selection of SCARED-C models in HBN training data

3c: Cross-validation and model selection of KSADS models in HBN training data

3d-e: Compiling info and plotting results across all cross-validation

4a-c: Validation of model performance in HBN test set with models trained on complete-case data and 3NN/9NN imputation, respectively. Validation is with SCARED-P, SCARED-C, and KSADS outcomes.

5a-c: Validation of model performance in combined NYC/LA test set with models trained on complete-case data and 3NN/9NN imputation, respectively. Validation here is just with SCARED-P as outcome. 

6: Comparisons and plots of validation summaries across models, measures, holdout sets, and pipelines

7: Modeling and prediction of many KSADS diagnoses in the HBN data (train + test)

8a-c: Comparing performance of nonlinear models (random forest, support vector with nonlinear kernal) with linear models (ridge and lasso regression) in iterative training/testing for prediction of SCARED-P, SCARED-C, and KSADS outcomes from GI symptoms, age, and sex

8d: Comparing predictions made by a sample set of models in 8a across model types

8e: Plotting the results of 8a-c

9: Analyses of demographic variables, and bivariate associations for several variable examined in the study

*Supporting functions found in cvFunction.R and helperFunctions.R*







**Questions or comments?**
Email paul.bloom@columbia.edu 

