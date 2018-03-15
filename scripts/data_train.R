#################################################################################
# DonorChoose project ###########################################################
# Train data preparation ########################################################
#################################################################################


# Environment ==================================================================
rm(list = ls())
cat('\014')

setwd("C:/Users/aLook/Documents/Machine_Learning/DonorChooseProject")

library(caret)
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

# Parameters ===================================================================

N_ROWS <- 30000 # number of rows for modelling
N_TOPICS_ESSAYS <- 75 # number of topics for essay data
N_TOPICS_RESOURCES <- 30 # number of topics for resources data

# # Testing parameters
# N_ROWS <- 100 # number of rows for modelling
# N_TOPICS_ESSAYS <- 2 # number of topics for essay data
# N_TOPICS_RESOURCES <- 2 # number of topics for resources data


TRAIN_PROB <- 0.75 # fraction of train
SEED <- 2911 # random seed for reproducibility

ADD_STOP_WORDS <- 'student'

# Read data ====================================================================
# http://programminghistorian.github.io/ph-submissions/lessons/published/basic-text-processing-in-r

resources <- data.table(read_csv('https://www.dropbox.com/s/wpd4gfgqa569ppg/resources.csv?dl=1'))
fullData <- data.table(read_csv('https://www.dropbox.com/s/56kqhbaz8b55nfc/train.csv?dl=1', 
                                locale = locale(encoding = 'utf8')))

# Clean data ===================================================================

flog.info('Data cleaning...')

# for computational purposes select only # rows according to N_ROWS
set.seed(SEED)
fullData <- fullData[sample(nrow(fullData), N_ROWS),]

# dtto in resources data
resources <- resources[id %in% fullData$id,]

fullData <- clean_donor_data(fullData)
fullData <- generate_features_donor(fullData)

# train test split
set.seed(SEED)

fullData[, sample := factor(sample(c('train', 'test'), 
                                   nrow(fullData), 
                                   replace = TRUE,
                                   prob = c(TRAIN_PROB, 1 - TRAIN_PROB)))]

sampleData <- fullData[, .(id, sample)]
setkey(sampleData, id)

# Data for text mining =========================================================

flog.info('Create corpora of essays...')

# https://www.tidytextmining.com/topicmodeling.html
# http://www.dataperspective.info/2013/10/topic-modeling-in-r.html

topics <- reshape_topics(fullData)

# create overall + train corpus
topics_dtm <- data_to_dtm(topics, ADD_STOP_WORDS)
topics_dtm_train <- data_to_dtm(topics[sample == 'train'], ADD_STOP_WORDS)

# Statistics ===================================================================

flog.info('Compute word statistics...')
topics <- append_word_stats(topics, topics_dtm)


# Sentiment analysis ===========================================================
# https://www.tidytextmining.com/sentiment.html

flog.info('Sentiment analysis of essays...')

topics <- append_sentiment(topics, topics_dtm)

# Topic models ==============================================================

flog.info('Topic model for essays...')

# create tf-idf matrix
topics_tfidf <- tapply(topics_dtm_train$v / row_sums(topics_dtm_train)[topics_dtm_train$i], topics_dtm_train$j, mean) * log2(nDocs(topics_dtm_train)/col_sums(topics_dtm_train > 0))

# exclude terms with low tf-idf
topics_dtm_train <- topics_dtm_train[, topics_tfidf >= 0.1]
rm(topics_tfidf)

topics_dtm_train <- topics_dtm_train[row_sums(topics_dtm_train) > 0,]


# 50 topics selected 
flog.info('Run LDA...')
topics_lda <- LDA(topics_dtm_train, k = N_TOPICS_ESSAYS, control = list(seed = SEED))
rm(topics_dtm_train)
flog.info('LDA finished...')

# beta matrix - words per topic
generated_topics <- tidy(topics_lda, matrix = "beta")

# overall beta (all topics)
overall <- tidy(topics_dtm) %>%
  dplyr::group_by(term) %>%
  dplyr::summarise(count = sum(count)) %>%
  dplyr::filter(count >= 50) %>% # consider only words with total frequency at least 10
  dplyr::mutate(beta = count / sum(count)) %>%
  dplyr::distinct(term, beta) %>%
  dplyr::rename(beta_overall = beta)

# compute log ratio (beta in topic / overall beta)
generated_topics <- dplyr::inner_join(generated_topics, overall, by = 'term') %>%
  dplyr::mutate(log_ratio = log2(beta / beta_overall))

# select top 10 terms per each topic by LOG RATIO
generated_top_terms_log <- generated_topics %>%
  group_by(topic) %>%
  top_n(10, log_ratio) %>%
  ungroup() %>%
  arrange(topic, -log_ratio)

# select top 10 terms per each topic by FREQ
generated_top_terms_freq <- generated_topics %>%
  group_by(topic) %>%
  filter(log_ratio >= 2) %>% # only words with reasonable log ratio
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)


rm(generated_topics)

# export results to xlsx
wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, 'TM_TOP_TERMS_LOG')
openxlsx::writeData(wb, 'TM_TOP_TERMS_LOG', generated_top_terms_log)
openxlsx::addWorksheet(wb, 'TM_TOP_TERMS_FREQ')
openxlsx::writeData(wb, 'TM_TOP_TERMS_FREQ', generated_top_terms_freq)

rm(overall, generated_top_terms_log, generated_top_terms_freq)


# predict topics in the topics data
topics <- append_topics(topics, topics_dtm, topics_lda)

# save topics_lda
flog.info('Saving topics_lda...')
save(topics_lda, file = './outputs/topics_lda.rda', compress = 'bzip2')
rm(topics_lda, topics_dtm)

# Aggregate topics data to features =========================================

flog.info('Aggregate topics data...')

topics <- aggregate_topics(topics)

setkey(fullData, id)
fullData <- merge(fullData, topics, all.x = TRUE)

# remove unnecessary variables
fullData <- fullData[, !c('teacher_id', 'project_submitted_datetime', 'project_title', paste0('project_essay_', 1:4), 'project_resource_summary'), with = FALSE]

# remove topics
rm(topics)

# Save full data and ============================================================
flog.info('Save the data...')
save(fullData, file = './outputs/fullData.rda', compress = 'bzip2')
rm(fullData)


# Resources data features =======================================================

flog.info('Clean resources data...')
resources <- clean_donor_resources(resources, sampleData)
rm(sampleData)

flog.info('Create resources corpora...')
resources_dtm <- data_to_dtm(resources, doc_var = 'resource_id', text_var = 'description')
resources_dtm <- resources_dtm[row_sums(resources_dtm) > 0,]



# train sample dtm for resources
resources_dtm_train <- data_to_dtm(resources[sample == 'train'], doc_var = 'resource_id', text_var = 'description')

# create tf-idf matrix
resources_tfidf <- tapply(resources_dtm_train$v / row_sums(resources_dtm_train)[resources_dtm_train$i], resources_dtm_train$j, mean) * log2(nDocs(resources_dtm_train)/col_sums(resources_dtm_train > 0))

# exclude terms with low tf-idf
resources_dtm_train <- resources_dtm_train[, resources_tfidf >= 0.1]
resources_dtm_train <- resources_dtm_train[row_sums(resources_dtm_train) > 0,]


# 20 resources topics selected 
flog.info('Run LDA on resources...')
resources_lda <- LDA(resources_dtm_train, k = N_TOPICS_RESOURCES, control = list(seed = SEED))
rm(resources_dtm_train, resources_tfidf)
flog.info('LDA finished...')

# save.image(file = paste0('./outputs/image_', format(Sys.Date(), '%Y%m%d'), '.rda'),
#            compress = 'bzip2')

# load('C:/Users/aLook/Dropbox/DonorsChooseProject/temp_outputs/image_20180312.rda')


# beta matrix - words per topic
generated_resources <- tidy(resources_lda, matrix = "beta")

# overall beta (all resources)
overall <- tidy(resources_dtm) %>%
  dplyr::group_by(term) %>%
  dplyr::summarise(count = sum(count)) %>%
  dplyr::filter(count >= 50) %>% # consider only words with total frequency at least 10
  dplyr::mutate(beta = count / sum(count)) %>%
  dplyr::distinct(term, beta) %>%
  dplyr::rename(beta_overall = beta)

# compute log ratio (beta in topic / overall beta)
generated_resources <- dplyr::inner_join(generated_resources, overall, by = 'term') %>%
  dplyr::mutate(log_ratio = log2(beta / beta_overall))

# select top 10 terms per each topic by LOG RATIO
resources_top_terms_log <- generated_resources %>%
  group_by(topic) %>%
  top_n(10, log_ratio) %>%
  ungroup() %>%
  arrange(topic, -log_ratio)

# select top 10 terms per each topic by FREQ
resources_top_terms_freq <- generated_resources %>%
  group_by(topic) %>%
  filter(log_ratio >= 2) %>% # only words with reasonable log ratio
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

rm(generated_resources)

# export results to xlsx
openxlsx::addWorksheet(wb, 'RES_TOP_TERMS_LOG')
openxlsx::writeData(wb, 'RES_TOP_TERMS_LOG', resources_top_terms_log)
openxlsx::addWorksheet(wb, 'RES_TOP_TERMS_FREQ')
openxlsx::writeData(wb, 'RES_TOP_TERMS_FREQ', resources_top_terms_freq)
openxlsx::saveWorkbook(wb, file = paste0('./outputs/', format(Sys.Date(), '%Y%m%d'), '_outputs.xlsx'), overwrite = TRUE)

rm(overall, resources_top_terms_log, resources_top_terms_freq)

debugonce(append_topics)
resources <- append_topics(resources, resources_dtm, resources_lda, 'RC', key = 'resource_id')

# save resources_lda
flog.info('Saving resources_lda...')
save(resources_lda, file = './outputs/resources_lda.rda', compress = 'bzip2')
rm(resources_lda, resources_dtm)

# Aggregate resources data to features =========================================

flog.info('Aggregate resources data...')

# debugonce(aggregate_resources)
resources <- aggregate_resources(resources)
setkey(resources, id)

# Merge to full data and save again =============================================
flog.info('Save the data...')
load('./outputs/fullData.rda')

fullData <- merge(fullData, resources, all.x = TRUE)

# clean data for modelling
fullData <- clean_for_modelling(fullData)

save(fullData, file = './outputs/fullData.rda', compress = 'bzip2')
