#################################################################################
# DonorChoose project ###########################################################
#################################################################################


# Environment ==================================================================
rm(list = ls())
cat('\014')

library(data.table)
library(caret)
library(readr)
library(lubridate)
library(openxlsx)
library(RANN)
library(slam)
library(stringi)
library(tidytext)
library(tm)
library(topicmodels)

# http://programminghistorian.github.io/ph-submissions/lessons/published/basic-text-processing-in-r
# Read data
# resources <- data.table(read_csv('https://www.dropbox.com/s/wpd4gfgqa569ppg/resources.csv?dl=1'))
# sample_submission <- fread('https://www.dropbox.com/s/4jec2zd80ti90ax/sample_submission.csv?dl=1')
fullData <- data.table(read_csv('https://www.dropbox.com/s/56kqhbaz8b55nfc/train.csv?dl=1', 
                                locale = locale(encoding = 'utf8')))
# test <- data.table(read_csv('https://www.dropbox.com/s/ssw9rl335flls45/test.csv?dl=1'))

source('scripts/helpers.R')

# Clean data ===================================================================

# for computational purposes select only 10000 rows 
set.seed(2911)
fullData <- fullData[sample(nrow(fullData), 10000),]

fullData <- clean_donor_data(fullData)
fullData <- generate_features_donor(fullData)

### train test split
PROB <- 0.75
set.seed(1129)

fullData[, sample := factor(sample(c('train', 'test'), 
                                   nrow(fullData), 
                                   replace = TRUE,
                                   prob = c(PROB, 1 - PROB)))]


# Data for text mining =========================================================

# https://www.tidytextmining.com/topicmodeling.html
# http://www.dataperspective.info/2013/10/topic-modeling-in-r.html

topics <- data.table::melt(fullData,
                           id.vars = c('id', 'sample'),
                           measure.vars = paste0('project_essay_', 1:4),
                           variable.name = 'essay',
                           value.name = 'text')[!is.na(text)]

topics[, document := as.character(1:nrow(topics))]

topics_dtm <- VectorSource(topics$text) %>% 
  VCorpus() %>% 
  DocumentTermMatrix(control = list(removePunctuation = TRUE,
                                    removeNumbers = TRUE,
                                    stopwords = TRUE))

# train sample dtm
topics_dtm_train <- VectorSource(topics$text[topics$sample == 'train']) %>% 
  VCorpus() %>% 
  DocumentTermMatrix(control = list(removePunctuation = TRUE,
                                    removeNumbers = TRUE,
                                    stopwords = TRUE))

# Statistics ===================================================================

# get total number of words per document
word_stats <- data.table(tidy(topics_dtm), key = 'document')[, .(n_types = sum(count),
                                                                 n_tokens = .N,
                                                                 mean_freq = mean(count),
                                                                 mean_char = mean(nchar(term))), 
                                                             by = document]


# merge to topics
setkey(topics, document)
topics <- merge(topics, word_stats, all.x = TRUE)


# Sentiment analysis ===========================================================
# https://www.tidytextmining.com/sentiment.html

# get sentiment library
sentiment_nrc <- data.table(sentiments, key = 'word')[lexicon == 'nrc'] # & !sentiment %in% c('positive', 'negative'), .(word, sentiment)]
sentiment_nrc[, sentiment := factor(sentiment)]

# merge to sentiment data
sentiment_data <- data.table(inner_join(tidy(topics_dtm), sentiment_nrc, by = c('term' = 'word')))
setkey(sentiment_data, document, term)

# aggregate counts
sentiment_data <- sentiment_data[, .N, by = .(document, sentiment)]
setkey(sentiment_data, document)

# get total number of words per document
word_counts <- data.table(tidy(topics_dtm), key = 'document')[, .(word_count = sum(count)), by = document]
sentiment_data <- merge(sentiment_data, word_counts)

# compute relative emotion scores
sentiment_data[, N := N / word_count]

# emotions to columns
sentiment_data <- data.table::dcast(sentiment_data,
                                    document ~ sentiment,
                                    value.var = 'N',
                                    fill = 0)

# merge to topics
setkey(topics, document)
topics <- merge(topics, sentiment_data)

# Topic models ==============================================================

# create tf-idf matrix
topics_tfidf <- tapply(topics_dtm_train$v / row_sums(topics_dtm_train)[topics_dtm_train$i], topics_dtm_train$j, mean) * log2(nDocs(topics_dtm_train)/col_sums(topics_dtm_train > 0))

# exclude terms with low tf-idf
topics_dtm_train <- topics_dtm_train[, topics_tfidf >= 0.1]
topics_dtm_train <- topics_dtm_train[row_sums(topics_dtm_train) > 0,]

# Deciding best K value using Log-likelihood method - not done because time consuming
# best.model <- lapply(c(10, 20, 50, 100), function(d){LDA(topics_dtm, d)})
# best.model.logLik <- as.data.frame(as.matrix(lapply(best.model, logLik)))


# 50 topics selected 
topics_lda <- LDA(topics_dtm_train, k = 50, control = list(seed = 1234))

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


# export results to xlsx
wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, 'TM_TOP_TERMS_LOG')
openxlsx::writeData(wb, 'TM_TOP_TERMS_LOG', generated_top_terms_log)
openxlsx::addWorksheet(wb, 'TM_TOP_TERMS_FREQ')
openxlsx::writeData(wb, 'TM_TOP_TERMS_FREQ', generated_top_terms_freq)
openxlsx::saveWorkbook(wb, file = paste0('./outputs/', format(Sys.Date(), '%Y%m%d'), '_outputs.xlsx'), overwrite = TRUE)

# gamma matrix - topics per document
# topic_scores <- data.table(tidy(topics_lda, matrix = "gamma"), key = 'document')
modelled_topics <- data.table(posterior(topics_lda, topics_dtm)$topics, 
                              keep.rownames = TRUE)

# k should be large enough so that each document is assigned to any topic with reasonable probability
# summary(apply(modelled_topics[, -1], 1, max))

# save.image(paste0('./outputs/', format(Sys.Date(), '%Y%m%d'), '_image.rda'))
# load('./outputs/20180008image.rda')

setnames(modelled_topics, 
         names(modelled_topics)[-1],
         paste0('T', formatC(as.numeric(names(modelled_topics)[-1]), width = 2, flag = 0)))

setnames(modelled_topics, 'rn', 'document')

# compute 5 main topics probability
modelled_topics[, paste0('Tmax', 1:5) := data.table(t(apply(modelled_topics[, -1], 1, function(x) sort(x, decreasing = TRUE)[1:5])))]

# summary(modelled_topics$Tmax1)
# summary(modelled_topics$Tmax2)
# summary(modelled_topics$Tmax3)
# summary(modelled_topics$Tmax4)
# summary(modelled_topics$Tmax5)

# merge to topics
setkey(modelled_topics, document)
topics <- merge(topics, modelled_topics, all.x = TRUE)


# Aggregate topics data to features =========================================

topics_agg <- data.table::melt(topics,
                               id_vars = c('document', 'id', 'sample', 'essay', 'text'),
                               variable.name = 'feature',
                               value.name = 'value')

topics_agg <- topics_agg[, .(MEAN = mean(value),
                             MIN = min(value),
                             MAX = max(value)), by = .(id, sample, feature)]

topics_agg <- data.table::dcast(topics_agg,
                                id ~ feature,
                                value.var = c('MEAN', 'MIN', 'MAX'))

setkey(fullData, id)
fullData <- merge(fullData, topics_agg, all.x = TRUE)

fullData <- fullData[, !c('teacher_id', 'project_submitted_datetime', 'project_title', paste0('project_essay_', 1:4), 'project_resource_summary')]


save.image(paste0('./outputs/', format(Sys.Date(), '%Y%m%d'), '_image.rda'))
load('./outputs/20180308_image.rda')
# fullData[, project_is_approved := factor(project_is_approved,
#                                          levels = c(1, 0),
#                                          labels = c('Approved', 'Not_approved'))]

# Model =========================================================================

dummies <- dummyVars(~ teacher_prefix + project_grade_category + project_subject_categories, fullData, sep = '_')

fullData <- cbind(fullData[, !c('teacher_prefix', 'project_grade_category', 'project_subject_categories'), with = FALSE], 
                  data.table(predict(dummies, fullData)))

setnames(fullData, names(fullData), make.names(names(fullData)))

for (i in names(fullData)) {
  if (is.numeric(fullData[[i]]) & any(is.na(fullData[[i]]))) {
    fullData[is.na(get(i)), (i) := 0]
  }
}

trainset <- fullData[sample == 'train', !'sample', with = FALSE]
testset <- fullData[sample == 'test', !'sample', with = FALSE]

# Feature selection ==============================================================

# all relevant features
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


### GBM ##############################################################################################

# Crossvalidation
cv <- trainControl(method = "cv", 
                   number = 4, 
                   classProbs = TRUE,
                   summaryFunction = f_mcc,
                   sampling = "smote") # possible to try "up" instead of smote



### Modelling process ================================================================================

# Initial start

# Candidate parameters
Grid <- expand.grid(n.trees = 400, # number of trees to fit (number of iterations)
                    shrinkage = 0.01, # learning rate
                    n.minobsinnode = 100, # numbr of observations in terminal node
                    interaction.depth = 6) # variable interaction - set relatively high

# tunning order
# 1. fix shrinkage = 0.1, n.minobsinnode = 100, interaction.depth = 5 and iterate over n.trees (40, 60, 80, 100)
# 2. tune n.minobsinnode = c(40, 60, 80, 100, 120, 140) and interaction.depth = c(4,6,8,10,12,14)
# 3. decrease shrinkage 0.01 and increase n.trees to cca. 400 

# Fit GBM
flog.info('Begin modelling GBM')
set.seed(SEED)
gbm_fit <- train(trainset[, selected_features, with = FALSE],
                 trainset[[target]],
                 preProcess = 'knnImpute', # impute missing values for socdem variables
                 metric = 'MCC', # or use Kappa
                 method = 'gbm', 
                 tuneGrid = Grid, 
                 trControl = cv,
                 bag.fraction = 0.8)

flog.info('Finished')

