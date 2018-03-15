#################################################################################
# DonorChoose project ###########################################################
#################################################################################


# Environment ===================================================================
rm(list = ls())
cat('\014')

library(data.table)
library(caret)


load('./outputs/gbm_20180314.rda')
load('./outputs/submission.rda')

predicted <- submission[, .(id)]
predicted[, project_is_approved := predict(gbm_fit, newdata = submission, type = 'prob')[, 'Approved']]

fwrite(predicted, file = './outputs/submission.csv')

