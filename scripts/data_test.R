#################################################################################
# DonorChoose project ###########################################################
# Test data preparation #########################################################
#################################################################################


# Environment ==================================================================
rm(list = ls())
cat('\014')

setwd("C:/Users/aLook/Documents/Machine_Learning/DonorChooseProject")

library(data.table)
library(futile.logger)
library(lubridate)
library(openxlsx)
library(RANN)
library(readr)
library(slam)
library(stringi)
library(tidyverse)
library(tidytext)
library(tm)
library(topicmodels)

source('scripts/helpers.R')

load('./outputs/topics_lda.rda')
load('./outputs/resources_lda.rda')

ADD_STOP_WORDS <- 'student'

# Read data ====================================================================

flog.info('Loading data from csv...')
submission <- data.table(read_csv('https://www.dropbox.com/s/ssw9rl335flls45/test.csv?dl=1'))

# # select subset for testing purposes
# set.seed(2911)
# submission <- submission[sample(nrow(submission), 100),]

resources_submission <- data.table(read_csv('https://www.dropbox.com/s/wpd4gfgqa569ppg/resources.csv?dl=1'))
resources_submission <- resources_submission[id %in% submission$id,]


# Clean data ===================================================================

flog.info('Cleaning data...')
submission <- clean_donor_data(submission, is_train = FALSE)
submission <- generate_features_donor(submission)


# Data for text mining =========================================================

# reshape essay data
flog.info('Reshaping the topics data...')
submission_topics <- reshape_topics(submission, sample = FALSE)

# create a document-term matrix
flog.info('Creating the document-term matrix...')
submission_dtm <- data_to_dtm(submission_topics, ADD_STOP_WORDS)


# Statistics ===================================================================
flog.info('Appending word stats...')
submission_topics <- append_word_stats(submission_topics, submission_dtm)

# Sentiment analysis ===========================================================
flog.info('Performing the sentiment analysis...')
submission_topics <- append_sentiment(submission_topics, submission_dtm)

# Topic models ==============================================================

flog.info('Assigning the topics...')
submission_dtm <- submission_dtm[row_sums(submission_dtm) > 0,]

# predict topics based on the lda object
submission_topics <- append_topics(submission_topics, submission_dtm, topics_lda)


# Aggregate topics data to features =========================================

flog.info('Aggregating the topics features...')
submission_topics <- aggregate_topics(submission_topics, is_train = FALSE)

# merge to the submission data
setkey(submission, id)
setkey(submission_topics, id)
submission <- merge(submission, submission_topics, all.x = TRUE)

rm(submission_topics)

# Resources data features =======================================================

# clean resources data
resources_submission <- clean_donor_resources(resources_submission)

# transform to document-term matrix
resources_submission_dtm <- data_to_dtm(resources_submission, doc_var = 'resource_id', text_var = 'description')
# remove empty rows
resources_submission_dtm <- resources_submission_dtm[row_sums(resources_submission_dtm) > 0,]

# append topic data
resources_submission <- append_topics(resources_submission, 
                                      resources_submission_dtm, 
                                      resources_lda, 
                                      'RC',
                                      key = 'resource_id')

rm(resources_submission_dtm)

# Aggregate resources data to features =========================================

flog.info('Aggregate resources data...')
resources_submission <- aggregate_resources(resources_submission, is_train = FALSE)

setkey(submission, id)
setkey(resources_submission, id)
submission <- merge(submission, resources_submission, all.x = TRUE)
rm(resources_submission)

# Save data and outputs ========================================================
save(submission, file = './outputs/submission.rda', compress = 'bzip2')