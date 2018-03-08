# clean the donor data
clean_donor_data <- function(data) {
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
  
  data[, project_is_approved := factor(project_is_approved,
                                       levels = c(1, 0),
                                       labels = c('Approved', 'Not_approved'))]
  
  # clean texts
  data[, project_essay_1 := iconv(gsub("\\\\r?\\\\n|\\\\r", " ", project_essay_1), 'UTF-8', 'ASCII', sub = '')]
  data[, project_essay_2 := iconv(gsub("\\\\r?\\\\n|\\\\r", " ", project_essay_2), 'UTF-8', 'ASCII', sub = '')]
  data[, project_essay_3 := iconv(gsub("\\\\r?\\\\n|\\\\r", " ", project_essay_3), 'UTF-8', 'ASCII', sub = '')]
  data[, project_essay_4 := iconv(gsub("\\\\r?\\\\n|\\r", " ", project_essay_4), 'UTF-8', 'ASCII', sub = '')]
  data[, project_resource_summary := iconv(gsub("\\\\r?\\\\n|\\\\r", " ", project_resource_summary), 'UTF-8', 'ASCII', sub = '')]
  
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
