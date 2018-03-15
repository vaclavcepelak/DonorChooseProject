#################################################################################
# DonorChoose project ###########################################################
#################################################################################


# Environment ===================================================================
rm(list = ls())
cat('\014')

library(data.table)
library(caret)
library(futile.logger)
library(lubridate)
library(MLmetrics)
library(openxlsx)
library(RANN)
library(slam)

source('scripts/helpers.R')


load('./outputs/fullData.rda')


# Model =========================================================================

flog.info('GBM modelling')
flog.info('Prepare data...')

trainset <- fullData[sample == 'train', !'sample', with = FALSE]
testset <- fullData[sample == 'test', !'sample', with = FALSE]

# Feature selection =============================================================

flog.info('Feature selection...')

# all relevant features (remove ids etc.)
features <- names(trainset)[c(-1, -2, -3, -5)]

# features with non-zero variance
features <- features[apply(trainset[, features, with = FALSE], 2, var) > 0]

# target name
target <- 'project_is_approved'
# Feature selection using random forest
cv <- trainControl(method = "cv", 
                   number = 3, 
                   classProbs = TRUE,
                   summaryFunction = f_mcc,
                   sampling = "smote")

Grid <- expand.grid(.mtry = 22)

feature_selection <- caret::train(trainset[, features, with = FALSE],
                                  trainset[[target]],
                                  method = 'rf',
                                  metric = 'MCC', # or use Kappa
                                  trControl = cv,
                                  tuneGrid = Grid,
                                  ntree = 200,
                                  maxnodes = 8,
                                  nodesize = 200)

# variable importance extraction
vi <- caret::varImp(feature_selection)
vi <- data.table(vi$importance, keep.rownames = TRUE)

# select features for further modelling
selected_features <- vi[Overall >= 1, rn]

# save selected features
save(selected_features, file = './outputs/selected_features.rda', compress = 'bzip2')
# load('./outputs/selected_features.rda')

# GBM ###########################################################################

flog.info('GBM first step...')

# Crossvalidation
cv <- trainControl(method = "cv", 
                   number = 4, 
                   classProbs = TRUE,
                   summaryFunction = twoClassSummary,
                   sampling = "smote") # possible to try "up" instead of smote



# Modelling process =============================================================

# Initial start

# Candidate parameters
Grid <- expand.grid(n.trees = seq(10, 4000, 10), # number of trees to fit (number of iterations)
                    shrinkage = c(0.001, 0.005), # learning rate
                    n.minobsinnode = 60, # number of observations in terminal node
                    interaction.depth = 14) # variable interaction - set relatively high

# tunning order
# 1. fix shrinkage = 0.1, n.minobsinnode = 100, interaction.depth = 5 and iterate over n.trees seq(1, 100, 1)
# 2. tune n.minobsinnode = c(40, 60, 80, 100, 120, 140) and interaction.depth = c(4, 6, 8, 10, 12, 14)
# 3. decrease shrinkage c(0.01, 0.05, 0.1) and increase n.trees to seq(1, 600, 1) 

# Fit GBM
flog.info('Begin modelling GBM')
set.seed(2911)
gbm_fit <- train(trainset[, selected_features, with = FALSE],
                 trainset[[target]],
                 metric = 'auc',
                 method = 'gbm', 
                 tuneGrid = Grid, 
                 trControl = cv,
                 bag.fraction = 0.8)

flog.info('Modelling finished')

# Confusion matrices for train and test
gbm_cm_train <- confusionMatrix(table(predict(gbm_fit), trainset[[target]]))
gbm_cm_test <- confusionMatrix(table(predict(gbm_fit, newdata = testset), testset[[target]]))

# MCC metric added
flog.info(paste0('MCC on train: ', round(f_mccf(y_pred = predict(gbm_fit), 
                                                y_true = trainset[[target]]), 3), '
                 MCC on test: ', round(f_mccf(y_pred = predict(gbm_fit, newdata = testset), 
                                              y_true = testset[[target]]), 3)))

# ROC Curve =====================================================================

# predict probability scores for test
gbm_probs <- predict(gbm_fit, testset, type = "prob")

# ROC data
gbm_ROC <- pROC::roc(predictor = gbm_probs$Approved,
                     response = testset[[target]])

# print AUC
flog.info(paste0('Area under the curve: ', round(gbm_ROC$auc, 4)))

# simple ROC plot
# plot(gbm_ROC, main = "GBM ROC", xlim = c(1, 0), ylim = c(0, 1))

# ROC curve plot by ggplot2
gbm_ROC_coords <- data.table(t(pROC::coords(gbm_ROC, "all", ret=c("specificity", "sensitivity"))))

df_plot <-  ggplot(gbm_ROC_coords, 
                   aes(x = specificity, y = sensitivity)) + 
  geom_line(size = 1.5, color = '#00c3dd') + 
  geom_abline(intercept = 1, slope = 1, size = 1, color = '#dd3654', linetype = 3) + 
  scale_x_reverse() + 
  ggtitle('ROC curve') +
  theme(plot.title = element_text(size = 35, face = "bold"),
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 12)) +
  xlab('Specificity') +
  ylab('Sensitivity')

# export to png
png(filename = paste0('./outputs/roc_', gsub('-', '', Sys.Date()), '.png'),
    width = 650, height = 600)
df_plot
dev.off()

# Save the model to .rda
save(gbm_fit, file = paste0('./outputs/gbm_', gsub('-', '', Sys.Date()), '.rda'), compress = 'bzip2')

