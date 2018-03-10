# clean the donor data
clean_donor_data <- function(data, is_train = TRUE) {
  if (!'data.table' %in% class(data)) data <- data.table(data)
  
  # set keys
  setkey(data, id, teacher_id)
  
  # turn variables to factors
  data[, id := factor(id)]
  data[, teacher_id := factor(teacher_id)]
  data[, teacher_prefix := factor(teacher_prefix)]
  data[, school_state := factor(school_state)]
  data[, project_submitted_date_num := as.numeric(project_submitted_datetime)]
  
  data[, project_grade_category := factor(project_grade_category)]
  data[, project_subject_categories := factor(project_subject_categories)]
  data[, project_subject_subcategories := factor(project_subject_subcategories)]
  
  if (is_train == TRUE) {
    data[, project_is_approved := factor(project_is_approved,
                                         levels = c(1, 0),
                                         labels = c('Approved', 'Not_approved'))]
  }
  
  # clean texts
  data[, project_essay_1 := iconv(gsub("\\\\r?\\\\n|\\\\r", " ", project_essay_1), 'UTF-8', 'ASCII', sub = '')]
  data[, project_essay_2 := iconv(gsub("\\\\r?\\\\n|\\\\r", " ", project_essay_2), 'UTF-8', 'ASCII', sub = '')]
  data[, project_essay_3 := iconv(gsub("\\\\r?\\\\n|\\\\r", " ", project_essay_3), 'UTF-8', 'ASCII', sub = '')]
  data[, project_essay_4 := iconv(gsub("\\\\r?\\\\n|\\r", " ", project_essay_4), 'UTF-8', 'ASCII', sub = '')]
  data[, project_resource_summary := iconv(gsub("\\\\r?\\\\n|\\\\r", " ", project_resource_summary), 'UTF-8', 'ASCII', sub = '')]
  
  return(data)
}

clean_donor_resources <- function(data, sample_data = NULL) {
  # clean data
  data[, id := factor(id)]
  data[, description := iconv(gsub("\\\\r?\\\\n|\\\\r", " ", description), 'UTF-8', 'ASCII', sub = '')]
  
  # merge train/test split
  setkey(data, id)
  
  # add resource id
  data[, resource_id := factor(1:nrow(data))]
  setkey(data, resource_id, id)
  
  return(data)
}


generate_features_donor <- function(data) {
  if (!'data.table' %in% class(data)) data <- data.table(data)
  
  # submission day/year
  data[, project_submitted_date_yday := yday(project_submitted_datetime)]
  data[, project_submitted_date_year := year(project_submitted_datetime)]
  
  # length of essays
  data[, E1_nchar := ifelse(is.na(project_essay_1), 0, nchar(project_essay_1))]
  data[, E2_nchar := ifelse(is.na(project_essay_2), 0, nchar(project_essay_2))]
  data[, E3_nchar := ifelse(is.na(project_essay_3), 0, nchar(project_essay_3))]
  data[, E4_nchar := ifelse(is.na(project_essay_4), 0, nchar(project_essay_4))]
  data[, ES_nchar := ifelse(is.na(project_resource_summary), 0, nchar(project_resource_summary))]
  
  # additional transformations
  data[, E_nchar_sum := apply(data[, .(E1_nchar, E2_nchar, E3_nchar, E4_nchar, ES_nchar)], 1, sum)]
  data[, E_nchar_mean := apply(data[, .(E1_nchar, E2_nchar, E3_nchar, E4_nchar, ES_nchar)], 1, function(x) mean(x[x > 0]))]
  data[, E_nchar_min := apply(data[, .(E1_nchar, E2_nchar, E3_nchar, E4_nchar, ES_nchar)], 1, min)]
  data[, E_nchar_max := apply(data[, .(E1_nchar, E2_nchar, E3_nchar, E4_nchar, ES_nchar)], 1, max)]
  data[, E_nchar_sd := apply(data[, .(E1_nchar, E2_nchar, E3_nchar, E4_nchar, ES_nchar)], 1, sd)]
  
  return(data)
}




# Wrapper function around the mccf function to be used in caret::train
f_mcc <- function(data, lev = NULL, model = NULL) {
  mcc_val <- f_mccf(y_pred = data$pred, y_true = data$obs, positive = lev[1])
  c(MCC = mcc_val)
}

# Function to calculate the matthews correlation coefficient
f_mccf <- function(y_true, y_pred, positive = NULL){
  
  Confusion_DF <- MLmetrics::ConfusionDF(y_pred, y_true)
  if (is.null(positive) == TRUE) 
    positive <- as.character(Confusion_DF[1, 1])
  
  TP <- as.integer(subset(Confusion_DF, y_true == positive & 
                            y_pred == positive)["Freq"])
  FP <- as.integer(sum(subset(Confusion_DF, y_true != positive & 
                                y_pred == positive)["Freq"]))
  TN <- as.integer(subset(Confusion_DF, y_true != positive & 
                            y_pred != positive)["Freq"])
  FN <- as.integer(sum(subset(Confusion_DF, y_true == positive & 
                                y_pred != positive)["Freq"]))
  
  denom <- as.double(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
  if (any((TP + FP) == 0, (TP + FN) == 0, (TN + FP) == 0, (TN + FN) == 0)){ 
    denom <- 1
  }
  
  mcc <- ((TP * TN) - (FP * FN))/sqrt(denom)
  
  return(mcc)
}



reshape_topics <- function(data, sample = TRUE) {
  
  if (sample == TRUE) id_vars <- c('id', 'sample') else id_vars <- 'id'

  # melt data by project essay
  topics <- data.table::melt(data,
                             id.vars = id_vars,
                             measure.vars = paste0('project_essay_', 1:4),
                             variable.name = 'essay',
                             value.name = 'text')[!is.na(text)]
  
  # add document number
  topics[, document := as.character(1:nrow(topics))]
  
  return(topics)
}


data_to_dtm <- function(data, 
                        add_stop_words = NULL, 
                        doc_var = 'document',
                        text_var = 'text') {
  library(data.table)
  
  data <- data[, c(doc_var, text_var), with = FALSE]
  setnames(data, old = c(doc_var, text_var), new = c('document', 'text'))
  data <- tidytext::unnest_tokens(data, word, text)
  
  if (is.null(add_stop_words)) {
    stopwords <- stop_words
  } else {
    stopwords <- add_row(stop_words, word = add_stop_words, lexicon = rep('SMART', length(add_stop_words)))
  }
  
  data <- data.table(anti_join(data, stopwords) %>%
    count(document, word, sort = TRUE) %>%
    ungroup(), key = 'document')
  
  data_dtm <- cast_dtm(data, document, word, n)
  
  return(data_dtm)
}

append_word_stats <- function(topics_data, topics_dtm) {
  
  # get total number of words per document
  word_stats <- data.table(tidy(topics_dtm), key = 'document')[, .(n_types = sum(count),
                                                                   n_tokens = .N,
                                                                   mean_freq = mean(count),
                                                                   mean_char = mean(nchar(term))), 
                                                               by = document]
  
  
  # merge to topics
  setkey(topics_data, document)
  topics <- merge(topics_data, word_stats, all.x = TRUE)
  return(topics_data)
}



append_sentiment <- function(topics_data, 
                             topics_dtm,
                             sentiment_library = data.table(tidytext::sentiments, key = 'word')[lexicon == 'nrc', .(word, sentiment)]) {
  
  # factor sentiment library
  sentiment_library[, sentiment := factor(sentiment)]
  
  # merge to sentiment data
  sentiment_data <- data.table(inner_join(tidy(topics_dtm), sentiment_library, by = c('term' = 'word')))
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
  setkey(topics_data, document)
  topics_data <- merge(topics_data, sentiment_data, all.x = TRUE)
  
  return(topics_data)
}



append_topics <- function(topics_data, 
                          topics_dtm,
                          topics_lda,
                          prefix = 'T',
                          key = 'document') {
  
  # gamma matrix - topics per document
  modelled_topics <- data.table(posterior(topics_lda, topics_dtm)$topics, 
                                keep.rownames = TRUE)
  
  
  setnames(modelled_topics, 
           names(modelled_topics)[-1],
           paste0(prefix, formatC(as.numeric(names(modelled_topics)[-1]), width = 2, flag = 0)))
  
  setnames(modelled_topics, 'rn', key)
  
  # compute 5 main topics probability
  modelled_topics[, paste0(prefix, 'max', 1:5) := data.table(t(apply(modelled_topics[, -1], 1, function(x) sort(x, decreasing = TRUE)[1:5])))]

  # merge to topics
  setkeyv(modelled_topics, key)
  topics_data <- merge(topics_data, modelled_topics, all.x = TRUE)
  
  return(topics_data)
}


aggregate_topics <- function(topics_data, is_train = TRUE) {
  
  topics_agg <- data.table::melt(topics_data,
                                 id_vars = c('document', 'id', 'sample', 'essay', 'text'),
                                 variable.name = 'feature',
                                 value.name = 'value')
  
  if (is_train == TRUE) {
    keys <- c('id', 'sample', 'feature')
  } else {
    keys <- c('id', 'feature')
  }
  
  topics_agg <- topics_agg[, .(MEAN = mean(value),
                               MIN = min(value),
                               MAX = max(value)), keyby = keys]
  
  topics_agg <- data.table::dcast(topics_agg,
                                  id ~ feature,
                                  value.var = c('MEAN', 'MIN', 'MAX'))
  
  
  return(topics_agg)
}


aggregate_resources <- function(resources_data, is_train = TRUE) {
  
  resources_agg <- data.table::melt(resources_data,
                                    id_vars = c('resource_id', 'id', 'sample', 'description', 'text'),
                                    variable.name = 'feature',
                                    value.name = 'value')
  
  resources_agg[is.na(value), value := 0]
  
  if (is_train == TRUE) {
    keys <- c('id', 'sample', 'feature')
  } else {
    keys <- c('id', 'feature')
  }
  
  resources_agg <- resources_agg[, .(SUM = sum(value),
                                     MEAN = mean(value),
                                     SD = ifelse(is.na(sd(value)), 0, sd(value)),
                                     MIN = min(value),
                                     MAX = max(value),
                                     N_ITEMS = .N), keyby = keys]
  
  resources_agg <- data.table::dcast(resources_agg,
                                     id ~ feature,
                                     value.var = c('SUM', 'MEAN', 'SD', 'MIN', 'MAX', 'N_ITEMS'),
                                     fill = 0)
  
  
  names(resources_agg)[names(resources_agg) == 'N_ITEMS_quantity'] <- 'N_ITEMS'
  
  resources_agg[, names(resources_agg)[grepl('N_ITEMS_', names(resources_agg))] := NULL]
  
  
  
  return(resources_agg)
}


clean_for_modelling <- function(data, 
                                vars = c('teacher_prefix', 'project_grade_category', 'project_subject_categories')) {
  dummy_fml <- as.formula(paste0('~ ', paste(vars, collapse = ' + ')))
  dummies <- dummyVars(dummy_fml, data, sep = '_')
  
  data <- cbind(data[, !vars, with = FALSE], data.table(predict(dummies, data)))
  
  setnames(data, names(data), make.names(names(data)))
  
  for (i in names(data)) {
    if (is.numeric(data[[i]]) & any(is.na(data[[i]]))) {
      fullData[is.na(get(i)), (i) := 0]
    }
  }
  
  return(data)
}
