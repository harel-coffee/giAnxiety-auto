# Paul A. Bloom
# October 2019
# Helper Functions for "Predicting Mental Health Risk In Primary Healthcare Settings: A Rapid Gastrointestinal Screener for Mental Health Outcomes in Youth"



# crossEntropy() ----------------------------------------------------------

# Vectorized function for calculating cross entropy (Log Loss) for binary classification
# yHat = model output probability
# y = true class
crossEntropy = Vectorize(function(yHat, y){
  if (y == 1){
    return (-log(yHat))
  }
  else{
    return (-log(1-yHat))
  }
})


# rMSE() ------------------------------------------------------------------

# function for calculating RMSE
# yHat = model output 
# y = true outcome
rMSE = function(yHat, y){
  residuals = y - yHat
  output = sqrt(mean(residuals^2))
}


# nMSE() ------------------------------------------------------------------


# function for caltulating normalized MSE
# yHat = model output
# y = true outcome
# y_mean = mean of true outcomes
nMSE = function(yHat, y, y_mean){
  residualsModel = y - yHat
  residualsMean = y - y_mean
  modelMSE = mean(residualsModel^2)
  meanMSE = mean(residualsMean^2)
  return(modelMSE/meanMSE)
}



# Q2 ----------------------------------------------------------------------

# function for calculating the q^2 metric (predictive squared correlation coefficient)
# yHat = model output 
# y = true outcome
q2 = function(yHat, y){
  residualsModel = y - yHat
  residualsMean = y - mean(y)
  modelMSE = mean(residualsModel^2)
  meanMSE = mean(residualsMean^2)
  return(1 - (modelMSE/meanMSE))
}

# removeRowsAllNA() -------------------------------------------------------

# Function to remove all rows from a dataframe where all of the columns are NA on a specified range
# dat = input dataframe
# columns = columns to check for NAs (other columns are ignored)
removeRowsAllNA <- function(dat, columns) {
  dat = dat %>% filter(rowSums(is.na(.)) != ncol(.[columns]))
  return(dat)
}


# percentBetter() ---------------------------------------------------------


# This function calculates the percentage for which column1 is higher than column2 (vectorized)
# Depending on the metric (i.e. if lower is better), 1-percentBetter() might be needed
percentBetter = function(column1, column2){
  if (length(column1) == length(column2)){
    diff = column1 - column2
    return(sum(diff > 0)/length(column1))
  }
  else{
    return('Error, col lengths unequal!')
  }
}

# permute() ---------------------------------------------------------------

# Function to run a permutation test for each input model -- returns a dataframe of permuted results (true outcomes shuffled) for each set of predictions
# numPerms = number of iterations of permutations to run
# trueOutcome = true outome values vector
# predCols = a named list of vectors of predicted values (length should match trueOutcome)
# metric = model evaluation metric to calculate ('rmse', 'crossEntropy', 'auc', or 'q2')

permute = function(numPerms, trueOutcome, predCols, metric){
  permFrame = tibble(index = 1:numPerms)
  for (i in 1:numPerms){
      # set the seed each iteration of the loop -- ensuring this function should give the same results if run multiple times with the same data
      set.seed(i)
    
      # shuffle the outcome
      sampleIndices = sample(1:length(trueOutcome), size = length(trueOutcome), replace=FALSE)
      permOutcome = trueOutcome[sampleIndices]
      
      # for each set of predictions, get model performance metric
      for (j in 1:length(predCols)){
        if (metric == 'rmse'){
          permFrame[i, j + 1] = rMSE(predCols[[j]], permOutcome)
        }
        else if (metric == 'crossEntropy'){
          permFrame[i, j + 1] = mean(crossEntropy(predCols[[j]], permOutcome))
        }
        else if (metric == 'auc'){
          permFrame[i, j + 1] = pROC::auc(permOutcome, predCols[[j]])
        }
        else if (metric == 'q2'){
          permFrame[i, j + 1] = q2(yHat = predCols[[j]], y = permOutcome)
        }
      }
  }
  # rename output dataframe with the names of the predCols list items
  names(permFrame)[-1] = names(predCols)
  return(permFrame)
}


# bestPermute() -----------------------------------------------------------

# From the output of the permute() function -- finds the permuted distribution with the best median, uses that for all as baseline for comparisons
# This is so we can compare all models for a given outcome to the same permuted null
# The output dataframe has all permuted distributions set to the best one (highest median performance).
# permDataFrame = a dataframe of permuted results, as output by the permute() function
# metric = scoring metric used ('auc', 'q2', 'crossEntropy', or 'rmse')


bestPermute = function(permDataFrame, metric){
  maxFuns = c('auc', 'q2')
  minFuns = c('crossEntropy', 'rmse')
  
  # if 'best' means metric should be maximized
  if (metric %in% maxFuns){
    bestMedianIndex = which.max(summarise_all(dplyr::select(permDataFrame, sum, indiv, noGi), .funs = median)) + 1
  }
  # if 'best' means metric should be minimized
  else if (metric %in% minFuns){
    bestMedianIndex = which.min(summarise_all(dplyr::select(permDataFrame, sum, indiv, noGi), .funs = median)) + 1
  }
  # Replace each permuted column with the column with the best median performance
  # This seems a little weird to *replace* here, but it helps set up for plotting the same permuted distribution for each model formulation
  permDataFrame = mutate(permDataFrame, 
                         sum = pull(permDataFrame, bestMedianIndex),
                         indiv = pull(permDataFrame, bestMedianIndex),
                         noGi = pull(permDataFrame, bestMedianIndex),
                         intercept = pull(permDataFrame, bestMedianIndex))
  return(permDataFrame)
}

# boots() -----------------------------------------------------------------


# Function to run a bootstrapping for each input model
# Returns a dataframe of all bootstrap results for each model
# numBoots = number of iterations of bootstrapping to run
# trueOutcome = vector of true outcomes
# predCols = a named list of vectors of predicted values (length should match trueOutcome)
# metric = scoring metric used ('auc', 'q2', 'crossEntropy', or 'rmse')

boots = function(numBoots, trueOutcome, predCols, metric){
  bootFrame = tibble(index = 1:numBoots)
  for (i in 1:numBoots){
    # set the seed each iteration of the loop -- ensuring this function should give the same results if run multiple times with the same data
    set.seed(i)
    
    # Create a bootstrap resampling of the outcomes
    sampleIndices = sample(1:length(trueOutcome), size = length(trueOutcome), replace=TRUE)
    bootOutcome = trueOutcome[sampleIndices]
    
    # For each vector of predictions, get the same set of resampled items, and calculate model performance metric
    for (j in 1:length(predCols)){
      bootPreds = predCols[[j]][sampleIndices]
      if (metric == 'rmse'){
        bootFrame[i, j + 1] = rMSE(bootPreds, bootOutcome)
      }
      else if (metric == 'crossEntropy'){
        bootFrame[i, j + 1] = mean(crossEntropy(bootPreds, bootOutcome))
      }
      else if (metric == 'q2'){
        bootFrame[i, j + 1] = q2(yHat = bootPreds, y = bootOutcome)
      }
      else if (metric == 'auc'){
        bootFrame[i, j + 1] = pROC::auc(bootOutcome, bootPreds)
      }
    }
  }
  # Rename columns based on predCols
  names(bootFrame)[-1] = names(predCols)
  return(bootFrame)
}


# makePreds() -------------------------------------------------------------

# Function to make predictions on held out data for each model
# testData = dataframe of test set features -- needs to contain exactly the same variables as used in the model
# modelsFrame = dataframe of rstanarm models (linear or logistic regressions) with model objects in the modObject column. row order in modelsFrame should be GI Summed, GI Indiv. Items, No-GI, Intercept-Only
# modelType = linear or logistic regression models ('linear', or 'logistic')
makePreds = function(testData, modelsFrame, modelType){
  numCols = ncol(testData)
  
  # For each model, use posterior_linpred() to get posterior predictions, then take the median prediction for each participant
  for (i in 1:nrow(modelsFrame)){
  	if (modelType == 'linear'){
    	predictions = data.frame(posterior_linpred(modelsFrame$modObject[[i]], newdata = testData))%>%
    		summarise_all(list(median = median)) %>% t() %>% as.vector()
  	}
  	else if (modelType == 'logistic'){
  		predictions = data.frame(posterior_linpred(modelsFrame$modObject[[i]], transform = TRUE, newdata = testData))%>%
    		summarise_all(list(median = median)) %>% t() %>% as.vector()
  	}
    # add new column to testData with predictions
    testData[,numCols + i] = predictions
  }
  # rename columns of predictions
  names(testData)[-(1:numCols)] = c('sumPreds', 'indivPreds', 'noGiPreds', 'interceptPreds')
  return(testData)
}


# GeomFlatViolin() --------------------------------------------------------

# Not my function, but putting it here in case the github link breaks
GeomFlatViolin <-
  ggproto("GeomFlatViolin", Geom,
          setup_data = function(data, params) {
            data$width <- data$width %||%
              params$width %||% (resolution(data$x, FALSE) * 0.9)
            
            # ymin, ymax, xmin, and xmax define the bounding rectangle for each group
            data %>%
              group_by(group) %>%
              mutate(ymin = min(y),
                     ymax = max(y),
                     xmin = x,
                     xmax = x + width / 2)
            
          },
          
          draw_group = function(data, panel_scales, coord) {
            # Find the points for the line to go all the way around
            data <- transform(data, xminv = x,
                              xmaxv = x + violinwidth * (xmax - x))
            
            # Make sure it's sorted properly to draw the outline
            newdata <- rbind(plyr::arrange(transform(data, x = xminv), y),
                             plyr::arrange(transform(data, x = xmaxv), -y))
            
            # Close the polygon: set first and last point the same
            # Needed for coord_polar and such
            newdata <- rbind(newdata, newdata[1,])
            
            ggplot2:::ggname("geom_flat_violin", GeomPolygon$draw_panel(newdata, panel_scales, coord))
          },
          
          draw_key = draw_key_polygon,
          
          default_aes = aes(weight = 1, colour = "grey20", fill = "white", size = 0.5,
                            alpha = NA, linetype = "solid"),
          
          required_aes = c("x", "y")
)

geom_flat_violin <- function(mapping = NULL, data = NULL, stat = "ydensity",
                             position = "dodge", trim = TRUE, scale = "area",
                             show.legend = NA, inherit.aes = TRUE, ...) {
  layer(
    data = data,
    mapping = mapping,
    stat = stat,
    geom = GeomFlatViolin,
    position = position,
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(
      trim = trim,
      scale = scale,
      ...
    )
  )
}