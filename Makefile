all: paper/jama_results.docx

# Make paper pdf
paper/jama_results.docx: paper/jama_results.Rmd paper/supplemental_jama.docx
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('paper/jama_results.Rmd')"

# Convert plots to png and make supplemental
paper/supplemental_jama.docx: paper/supplemental_jama.Rmd scripts/1a_cleaning_LA_NYC.html scripts/2b_robustness.html scripts/3d_hbn_compile_cv.html scripts/3e_compare_cv_model_params_hbn.html scripts/6_compare_validation_summaries.html scripts/7_hbn_KSADS_many_diagnoses.html scripts/8_nonlinear_models/8e_plot_nonlinear_models.html scripts/9_demographic_analyses.html 
	cd scripts; bash pdf2png.sh
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('paper/supplemental_jama.Rmd')"

# Demographic analyses -- depend on data cleaning
scripts/9_demographic_analyses.html: scripts/9_demographic_analyses.Rmd scripts/1a_cleaning_LA_NYC.html scripts/1c_cleaning_splitting_hbn.html
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/9_demographic_analyses.Rmd')"

# Nonlinear models of HBN
scripts/8_nonlinear_models/8e_plot_nonlinear_models.html: scripts/1c_cleaning_splitting_hbn.html scripts/8_nonlinear_models/8e_plot_nonlinear_models.Rmd output/nonlinearRegressionScaredP.csv 
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/8_nonlinear_models/8e_plot_nonlinear_models.Rmd')"

output/nonlinearRegressionScaredP.csv: scripts/1c_cleaning_splitting_hbn.html scripts/8_nonlinear_models/8a_nonlinear_compare_SCAREDP.py scripts/8_nonlinear_models/8b_nonlinear_compare_SCAREDC.py scripts/8_nonlinear_models/8c_nonlinear_compare_KSADS.py 
	cd scripts/8_nonlinear_models/; python3 8a_nonlinear_compare_SCAREDP.py
	cd scripts/8_nonlinear_models/; python3 8b_nonlinear_compare_SCAREDC.py
	cd scripts/8_nonlinear_models/; python3 8c_nonlinear_compare_KSADS.py
	cd scripts/8_nonlinear_models/; jupyter nbconvert --to notebook --inplace --execute 8d_nonlinear_prediction_correlations.ipynb

# Explorations of many KSADS diagnoses
scripts/7_hbn_KSADS_many_diagnoses.html: scripts/7_hbn_KSADS_many_diagnoses.Rmd cleanData/clinicianConsensusDiagnoses.csv scripts/1b_cleaning_KSADS_hbn.html scripts/helperFunctions.R
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/7_hbn_KSADS_many_diagnoses.Rmd')"

# Comparing all validations -- depends on validation of both LA/NYC and HBN
scripts/6_compare_validation_summaries.html: scripts/6_compare_validation_summaries.Rmd scripts/4a_hbn_validation_complete.html scripts/4b_hbn_validation_3NN.html scripts/4c_hbn_validation_9NN.html scripts/5a_la_nyc_validation_complete.html scripts/5b_la_nyc_validation_3NN.html scripts/5c_la_nyc_validation_9NN.html
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/6_compare_validation_summaries.Rmd')"

# LA + NYC Validation -- depends on CV
scripts/5a_la_nyc_validation_complete.html: scripts/5a_la_nyc_validation_complete.Rmd scripts/3a_hbn_SCAREDP_cv.html scripts/3b_hbn_SCAREDC_cv.html scripts/3c_hbn_KSADS_cv.html scripts/1a_cleaning_LA_NYC.html
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/5a_la_nyc_validation_complete.Rmd')"

scripts/5b_la_nyc_validation_3NN.html: scripts/5b_la_nyc_validation_3NN.Rmd scripts/3a_hbn_SCAREDP_cv.html scripts/3b_hbn_SCAREDC_cv.html scripts/3c_hbn_KSADS_cv.html scripts/1a_cleaning_LA_NYC.html
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/5b_la_nyc_validation_3NN.Rmd')"

scripts/5c_la_nyc_validation_9NN.html: scripts/5c_la_nyc_validation_9NN.Rmd scripts/3a_hbn_SCAREDP_cv.html scripts/3b_hbn_SCAREDC_cv.html scripts/3c_hbn_KSADS_cv.html scripts/1a_cleaning_LA_NYC.html
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/5c_la_nyc_validation_9NN.Rmd')"

# HBN Validation - depends on CV
scripts/4a_hbn_validation_complete.html: scripts/4a_hbn_validation_complete.Rmd scripts/3a_hbn_SCAREDP_cv.html scripts/3b_hbn_SCAREDC_cv.html scripts/3c_hbn_KSADS_cv.html
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/4a_hbn_validation_complete.Rmd')"

scripts/4b_hbn_validation_3NN.html: scripts/4b_hbn_validation_3NN.Rmd scripts/3a_hbn_SCAREDP_cv.html scripts/3b_hbn_SCAREDC_cv.html scripts/3c_hbn_KSADS_cv.html
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/4b_hbn_validation_3NN.Rmd')"

scripts/4c_hbn_validation_9NN.html: scripts/4c_hbn_validation_9NN.Rmd scripts/3a_hbn_SCAREDP_cv.html scripts/3b_hbn_SCAREDC_cv.html scripts/3c_hbn_KSADS_cv.html
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/4c_hbn_validation_9NN.Rmd')"

# CV for model selection/fitting models to training set for validation
scripts/3a_hbn_SCAREDP_cv.html: scripts/3a_hbn_SCAREDP_cv.Rmd scripts/1c_cleaning_splitting_hbn.html scripts/cvFunction.R
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/3a_hbn_SCAREDP_cv.Rmd')"

scripts/3b_hbn_SCAREDC_cv.html: scripts/3b_hbn_SCAREDC_cv.Rmd scripts/1c_cleaning_splitting_hbn.html scripts/cvFunction.R
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/3b_hbn_SCAREDC_cv.Rmd')"

scripts/3c_hbn_KSADS_cv.html: scripts/3c_hbn_KSADS_cv.Rmd  scripts/1c_cleaning_splitting_hbn.html scripts/cvFunction.R
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/3c_hbn_KSADS_cv.Rmd')"

# 3D and 3E depend on the 3A-C being run already
scripts/3d_hbn_compile_cv.html: scripts/3d_hbn_compile_cv.Rmd scripts/3a_hbn_SCAREDP_cv.html scripts/3b_hbn_SCAREDC_cv.html scripts/3c_hbn_KSADS_cv.html
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/3d_hbn_compile_cv.Rmd')"

scripts/3e_compare_cv_model_params_hbn.html: scripts/3e_compare_cv_model_params_hbn.Rmd scripts/3a_hbn_SCAREDP_cv.html scripts/3b_hbn_SCAREDC_cv.html scripts/3c_hbn_KSADS_cv.html
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/3e_compare_cv_model_params_hbn.Rmd')"

# Replication and Robustness Analyses
scripts/2b_robustness.html: scripts/2b_robustness.Rmd scripts/2a_replication_SCAREDP.html scripts/1c_cleaning_splitting_hbn.html scripts/1a_cleaning_LA_NYC.html scripts/helperFunctions.R
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/2b_robustness.Rmd')"

scripts/2a_replication_SCAREDP.html: scripts/2a_replication_SCAREDP.Rmd scripts/1c_cleaning_splitting_hbn.html scripts/1a_cleaning_LA_NYC.html scripts/helperFunctions.R
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/2a_replication_SCAREDP.Rmd')"

# Data cleaning
scripts/1c_cleaning_splitting_hbn.html: scripts/1c_cleaning_splitting_hbn.Rmd scripts/1b_cleaning_KSADS_hbn.html rawData/hbnRawData.csv scripts/helperFunctions.R
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/1c_cleaning_splitting_hbn.Rmd')"

scripts/1b_cleaning_KSADS_hbn.html: scripts/1b_cleaning_KSADS_hbn.Rmd rawData/clinicianDiagnoses.csv
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/1b_cleaning_KSADS_hbn.Rmd')"

scripts/1a_cleaning_LA_NYC.html: scripts/1a_cleaning_LA_NYC.Rmd rawData/CBCL_ELFK.csv rawData/ELFK_demo.csv rawData/SCARED_ELFK.csv rawData/GIANXSx_SB.csv rawData/SB_scared_individual_items.csv rawData/sbMaster.csv
	Rscript -e "library(rmarkdown); Sys.setenv(RSTUDIO_PANDOC='/Applications/RStudio.app/Contents/MacOS/pandoc'); rmarkdown::render('scripts/1a_cleaning_LA_NYC.Rmd')"

