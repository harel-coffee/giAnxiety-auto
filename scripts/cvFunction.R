# Paul A. Bloom
# October 2019
# Big ugly function to cross-validate a bunch of models at once!

# inputData = the input dataframe for conducting cross-validation -- must have variable names as specified by the model formula objects
# outcomeColumn = a string matching the column name of the the outcome variable in inputData being predicted by each model
# metric = model scoring metric, can be (rMSE, crossEntropy, or q2)
# numFolds = number of folds of cross validation to use (i.e. 10-fold)
# cvRounds = number of independently-seeded iterations of k-fold cross-validation to do
# sumFormulas, indivFormulas, noGiFormulas, interceptFormula = model formulae objects, regression model formula specification for each modelto be cross-validated. Each one is a list object 
# modType = The function `lm` or `glm`, specifying whether lm or glm (linear vs. logistic regressions)
cvManyLinearModels = function(inputData, outcomeColumn, metric, numFolds, cvRounds, 
                              sumFormulas, indivFormulas, noGiFormulas, interceptFormula, modType){
  # Some checks for input errors
  stopifnot(
    outcomeColumn %in% names(inputData),
    metric %in% c('rMSE', 'crossEntropy', 'q2'),
    all(map_lgl(list(sumFormulas, indivFormulas, noGiFormulas, interceptFormula), function(x) {
      map_lgl(x, function(xx) {all(is_formula(xx))})})), # check that the elements of each formulae list are all formulae
    is_function(modType)
  )
  
  for (j in 1:cvRounds){
    # set the seed differently each time. This means that if this CV function is run multiple times for the same data, we should get the same results
    set.seed(j)

    # Split the data
    folds = caret::createFolds(inputData[[outcomeColumn]], k = numFolds) # outputs a list, length k, with the indices of each fold
    cvFrame = data.frame(cvFold = 1:numFolds) # create a data frame containing a row for each of the k folds
    cvFrame$cvRound = j # create a column to indicate which cv round this is (long format)
    
    # Do k-fold cross-val for each model
    for (i in 1:numFolds){
      # Set up train/test set for each fold
      innerTrain = inputData[-folds[[i]],]
      innerTest = inputData[folds[[i]],]
      
      # Fit each type of model on the i-th train data, and store the result in a new column of cvFrame, in the i-th row.
      # Each model type (sum, indiv, noGi, intercept) are contained in their own column, each row pertains to the identical i-th data partition
      # (Ensure models are fit using the function specified in modType)

      # Models/Predictions for each model
      if (identical(modType, lm)){ # if models are linear regression
        # Summed models
        cvFrame$modSum1[i] = list(do.call(modType, args = list(formula = sumFormulas[[1]], data = innerTrain)))
        cvFrame$modSum2[i] = list(do.call(modType, args = list(formula = sumFormulas[[2]], data = innerTrain)))
        cvFrame$modSum3[i] = list(do.call(modType, args = list(formula = sumFormulas[[3]], data = innerTrain)))
        cvFrame$modSum4[i] = list(do.call(modType, args = list(formula = sumFormulas[[4]], data = innerTrain)))
        
        # Individual Item Models
        cvFrame$modIndiv1[i] = list(do.call(modType, args = list(formula = indivFormulas[[1]], data = innerTrain)))
        cvFrame$modIndiv2[i] = list(do.call(modType, args = list(formula = indivFormulas[[2]], data = innerTrain)))
        cvFrame$modIndiv3[i] = list(do.call(modType, args = list(formula = indivFormulas[[3]], data = innerTrain)))
        cvFrame$modIndiv4[i] = list(do.call(modType, args = list(formula = indivFormulas[[4]], data = innerTrain)))
        
        # No GI Models
        cvFrame$modNoGi1[i] = list(do.call(modType, args = list(formula = noGiFormulas[[1]], data = innerTrain)))
        cvFrame$modNoGi2[i] = list(do.call(modType, args = list(formula = noGiFormulas[[2]], data = innerTrain)))
        cvFrame$modNoGi3[i] = list(do.call(modType, args = list(formula = noGiFormulas[[3]], data = innerTrain)))
        cvFrame$modNoGi4[i] = list(do.call(modType, args = list(formula = noGiFormulas[[4]], data = innerTrain)))
        
        # Intercept Only
        cvFrame$modInt[i] = list(do.call(modType, args = list(formula = interceptFormula[[1]], data = innerTrain)))
        
        # Now generate the predictions from each of these models, and store them as a new column within the test data (in the i-th fold's row))
        # Preds:
        innerTest$modSum1Preds = as.vector(predict(cvFrame$modSum1[i], newdata = innerTest))[[1]]
        innerTest$modSum2Preds = as.vector(predict(cvFrame$modSum2[i], newdata = innerTest))[[1]]
        innerTest$modSum3Preds = as.vector(predict(cvFrame$modSum3[i], newdata = innerTest))[[1]]
        innerTest$modSum4Preds = as.vector(predict(cvFrame$modSum4[i], newdata = innerTest))[[1]]
        innerTest$modIndiv1Preds = as.vector(predict(cvFrame$modIndiv1[i], newdata = innerTest))[[1]]
        innerTest$modIndiv2Preds = as.vector(predict(cvFrame$modIndiv2[i], newdata = innerTest))[[1]]
        innerTest$modIndiv3Preds = as.vector(predict(cvFrame$modIndiv3[i], newdata = innerTest))[[1]]
        innerTest$modIndiv4Preds = as.vector(predict(cvFrame$modIndiv4[i], newdata = innerTest))[[1]]
        innerTest$modNoGi1Preds = as.vector(predict(cvFrame$modNoGi1[i], newdata = innerTest))[[1]]
        innerTest$modNoGi2Preds = as.vector(predict(cvFrame$modNoGi2[i], newdata = innerTest))[[1]]
        innerTest$modNoGi3Preds = as.vector(predict(cvFrame$modNoGi3[i], newdata = innerTest))[[1]]
        innerTest$modNoGi4Preds = as.vector(predict(cvFrame$modNoGi4[i], newdata = innerTest))[[1]]
        innerTest$modIntPreds = as.vector(predict(cvFrame$modInt[i], newdata = innerTest))[[1]]
      } else if (identical(modType, glm)){ 
        
        # Repeat the code for this process identically, for the case when the models are logistic regressions
        
        # Summed models
        cvFrame$modSum1[i] = list(do.call(modType, args = list(formula = sumFormulas[[1]], data = innerTrain, family = binomial)))
        cvFrame$modSum2[i] = list(do.call(modType, args = list(formula = sumFormulas[[2]], data = innerTrain, family = binomial)))
        cvFrame$modSum3[i] = list(do.call(modType, args = list(formula = sumFormulas[[3]], data = innerTrain, family = binomial)))
        cvFrame$modSum4[i] = list(do.call(modType, args = list(formula = sumFormulas[[4]], data = innerTrain, family = binomial)))
        
        # Individual Item Models
        cvFrame$modIndiv1[i] = list(do.call(modType, args = list(formula = indivFormulas[[1]], data = innerTrain, family = binomial)))
        cvFrame$modIndiv2[i] = list(do.call(modType, args = list(formula = indivFormulas[[2]], data = innerTrain, family = binomial)))
        cvFrame$modIndiv3[i] = list(do.call(modType, args = list(formula = indivFormulas[[3]], data = innerTrain, family = binomial)))
        cvFrame$modIndiv4[i] = list(do.call(modType, args = list(formula = indivFormulas[[4]], data = innerTrain, family = binomial)))
        
        # No GI Models
        cvFrame$modNoGi1[i] = list(do.call(modType, args = list(formula = noGiFormulas[[1]], data = innerTrain, family = binomial)))
        cvFrame$modNoGi2[i] = list(do.call(modType, args = list(formula = noGiFormulas[[2]], data = innerTrain, family = binomial)))
        cvFrame$modNoGi3[i] = list(do.call(modType, args = list(formula = noGiFormulas[[3]], data = innerTrain, family = binomial)))
        cvFrame$modNoGi4[i] = list(do.call(modType, args = list(formula = noGiFormulas[[4]], data = innerTrain, family = binomial)))
        
        # Intercept Only
        cvFrame$modInt[i] = list(do.call(modType, args = list(formula = interceptFormula[[1]], data = innerTrain, family = binomial)))
        
        # Preds:
        innerTest$modSum1Preds = as.vector(predict(cvFrame$modSum1[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modSum2Preds = as.vector(predict(cvFrame$modSum2[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modSum3Preds = as.vector(predict(cvFrame$modSum3[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modSum4Preds = as.vector(predict(cvFrame$modSum4[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modIndiv1Preds = as.vector(predict(cvFrame$modIndiv1[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modIndiv2Preds = as.vector(predict(cvFrame$modIndiv2[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modIndiv3Preds = as.vector(predict(cvFrame$modIndiv3[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modIndiv4Preds = as.vector(predict(cvFrame$modIndiv4[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modNoGi1Preds = as.vector(predict(cvFrame$modNoGi1[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modNoGi2Preds = as.vector(predict(cvFrame$modNoGi2[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modNoGi3Preds = as.vector(predict(cvFrame$modNoGi3[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modNoGi4Preds = as.vector(predict(cvFrame$modNoGi4[i], newdata = innerTest, type ='response'))[[1]]
        innerTest$modIntPreds = as.vector(predict(cvFrame$modInt[i], newdata = innerTest, type ='response'))[[1]]
      }
      
      # Now, store metric values into output dataframe, one entry for each model/formula's column in the i-th row
      
      # If the metric isn't cross-entropy -- can just output metric into output scoring frame (cvFrame)
      if (!identical(metric, crossEntropy)){
        cvFrame$modSum1Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modSum1Preds))
        cvFrame$modSum2Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modSum2Preds))
        cvFrame$modSum3Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modSum3Preds))
        cvFrame$modSum4Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modSum4Preds))
        cvFrame$modIndiv1Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modIndiv1Preds))
        cvFrame$modIndiv2Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modIndiv2Preds))
        cvFrame$modIndiv3Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modIndiv3Preds))
        cvFrame$modIndiv4Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modIndiv4Preds))
        cvFrame$modNoGi1Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modNoGi1Preds))
        cvFrame$modNoGi2Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modNoGi2Preds))
        cvFrame$modNoGi3Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modNoGi3Preds))
        cvFrame$modNoGi4Score[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modNoGi4Preds))
        cvFrame$modIntScore[i] =  do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modIntPreds))
        
        # Repeat for the case when metric is crossEntropy with a slight modification:
        
      } else if (identical(metric, crossEntropy)){ # If metric is crossEntropy, need to take mean across the output vector before putting into cvFrame
        cvFrame$modSum1Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modSum1Preds)))
        cvFrame$modSum2Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modSum2Preds)))
        cvFrame$modSum3Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modSum3Preds)))
        cvFrame$modSum4Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modSum4Preds)))
        cvFrame$modIndiv1Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modIndiv1Preds)))
        cvFrame$modIndiv2Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modIndiv2Preds)))
        cvFrame$modIndiv3Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modIndiv3Preds)))
        cvFrame$modIndiv4Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modIndiv4Preds)))
        cvFrame$modNoGi1Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modNoGi1Preds)))
        cvFrame$modNoGi2Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modNoGi2Preds)))
        cvFrame$modNoGi3Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modNoGi3Preds)))
        cvFrame$modNoGi4Score[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modNoGi4Preds)))
        cvFrame$modIntScore[i] =  mean(do.call(metric, args = list(y = innerTest[[outcomeColumn]], yHat = innerTest$modIntPreds)))
      }
    }
    # If this is the first iteration of cv, set up cvIndivResults output.
    # This will be a data frame in which to store the results from each cross validation round (to later aggregate/summarize)
    if (j == 1){
      # extract the columns with the output metrics scored on test data
      cvIndivResults = dplyr::select(cvFrame, contains('Score'), cvFold, cvRound) 
    }else{ # Otherwise, for subsequent iterations, add additional cv results to the cvIndivResults dataframe via rbind()
      cvIndivResults = rbind(cvIndivResults, dplyr::select(cvFrame, contains('Score'), cvFold, cvRound))
    }

  }
  # Summarize CV results across all folds in all rounds for each model
  cvSummary = cvIndivResults %>%
    tidyr::gather(., key = 'model', value = 'Score', -contains('cv')) %>%
    group_by(model) %>%
    summarise(median = median(Score),
              # record the distribution across the j cvRounds
              lwr95 = quantile(Score, probs= .025),
              upr95 = quantile(Score, probs = .975),
              lwr80 = quantile(Score, probs= .1),
              upr80 = quantile(Score, probs= .9)) %>%
    mutate(.,
           ModelSet = case_when(
             grepl('Sum', model) ~ 'GI Sum Score',
             grepl('Indiv', model) ~ 'GI Indiv. Items',
             grepl('NoGi', model) ~ 'No GI Term',
             grepl('Int', model) ~ 'Intercept Only')) %>%
    ungroup() %>%
    group_by(ModelSet)
  
  # Make a ranking variable based on median model performance for each cross-validated model within each formulation
  if (identical(metric, q2)){
    cvSummary = mutate(cvSummary, rank = rank(median),
           rank = ifelse(rank == max(rank), 1, 0))
  }else{
    cvSummary = mutate(cvSummary, rank = rank(median),
                       rank = ifelse(rank == min(rank), 1, 0))
  }
  
  # a dataframe with best model information (within each formulation) for final models on training set/testing set
  bestModels = cvSummary %>% 
    group_by(ModelSet) %>%
    filter(rank == 1) %>%
    mutate(., modelNum = parse_number(model))
  
  # rename model names
  cvSummary$model = dplyr::recode(cvSummary$model, 
                                  'modSum1Score'= 'No Interactions', 
                                  'modSum2Score'= 'Age*GI',
                                  'modSum3Score'= 'Sex*GI',
                                  'modSum4Score'= 'Age*Sex*GI',
                                  'modIndiv1Score'= 'No Interactions', 
                                  'modIndiv2Score'= 'Age*GI',
                                  'modIndiv3Score'= 'Sex*GI',
                                  'modIndiv4Score'= 'Age*Sex*GI',
                                  'modNoGi1Score'= 'Age', 
                                  'modNoGi2Score'= 'Sex',
                                  'modNoGi3Score'= 'Age + Sex',
                                  'modNoGi4Score'= 'Age*Sex',
                                  'modIntScore' = 'Intercept Only')
  
  # Output list of 3 dataframes: each cv model result, the cv summary, and best model info
  return(list('CV Indiv Results' = cvIndivResults, 'CV Summary' = cvSummary, 'Selected Models' = bestModels))
}
