# Data: randomForest, DecisionTrees, naiveBayes, svm
# Data complicated: regression
# Train, Test: knn
# talvez: xgboost, adaboost

# Libraries
library(magrittr)
library(tibble)
library(kknn)
library(dplyr)
library(tidyr)
library(purrr)
library(caret)
library(randomForest)
library(stringr)

dataset = as_tibble(iris)

# Input params
rf = list(
           fun = "randomForest",
           package = "randomForest",
           importance = c(TRUE, FALSE),
           ntree = c(100, 1000),
           mtry = c(2, 4)
          )

data = list(
             data = dataset,
             target = "Species",
             validation = "kfold_3",
             metric = "accuracy"
)

# = = = = = = = = = = = = = = = # ----
#     Create expressions df     # ----
# = = = = = = = = = = = = = = = # ----

# Control params or fun
control <- function(obj){
  names = names(obj)
  control_index = names %in% c("fun", "package")
  
  fun_list = obj[control_index]
  param_list = obj[!control_index]
  
  control = list(model = fun_list, param = param_list)
  
  return(control)
}

# Put '' in characteres or transform to character
to_character <- function(obj){
  if(is.character(obj)){
    obj = paste0("'", obj, "'")
  } else{
    obj = as.character(obj)
  }
}

# Put "=" in params
put_simbols <- function(obj){
  for(i in names(obj)){
    name = i
    str = obj[[name]]
    
    obj[[name]] = paste(name, "=", str)
  }
  return(obj)
}

# Dataframe's combinations
create_expression <- function(obj, data){
  obj_param = obj[['param']]
  obj_model = obj[['model']]
  
  if("formula" %in% names(data)){
    formula = data[['formula']]
  } else{
    formula = paste0(data[['target']], "~.")
  }
  
  df_combinations = expand.grid(obj_param) %>%
    as_tibble()

  
  
  df_expressions = df_combinations %>%
    unite(col = params, sep = ", ") %>%
    dplyr::mutate(fun = unlist(obj_model['fun']),
                  package = unlist(obj_model['package']),
                  formula = formula,
                  data = data[['dataset']]) %>%
    dplyr::mutate(expression = paste0(package, 
                                      "::", 
                                      fun,
                                      "(",
                                      formula,
                                      ", data = data, ", 
                                      params, 
                                      ")"),
                  ID = 1:n()) %>%
    split(.$ID) %>%
    map(~pull(., expression))
  
  df_combinations = df_combinations %>%
    dplyr::mutate(fun = unlist(obj_model['fun']),
                  N = 1:n()) %>%
    tidyr::nest(params = -c(fun, N)) %>%
    dplyr::select(-N)
  

  return(list(df_control = df_combinations,
              df_expressions = df_expressions))
}

# = = = = = = = = = = = = = = = # ----
#   Algorithms for validation    # ----
# = = = = = = = = = = = = = = = # ----

# We will use a S3 Class to create the cases for validations

# Defining the class and call the main function
split_by_validation <- function(data){
  method = data[['validation']]
  
  if(grepl(x = method, pattern = 'kfold')){
    class(data) = "kfold"
  }
  execute_validation(data)
}

# Execute by the class (method of validation)
execute_validation <- function(data){
  df = data[['data']]
  method = data[['validation']]
  UseMethod("execute_validation")
}

# Default
execute_validation.default = function(data){
  print("Please, choose a valid method.")
}

# K-Fold
execute_validation.kfold  = function(data){
  folds = str_split(string = method, pattern = "_")[[1]][2]
  if(is.na(folds)){
    folds = 10
  } else{
    folds = as.numeric(folds)  
  }
  
  len <- nrow(df)
  index <- cut(seq(1, len), 
               breaks = folds, 
               labels = FALSE) %>%
    sample()
  
  new_data <- df %>% 
    dplyr::mutate(fold = index) 
  
  list_data = list()
  for(i in 1:folds){
    train = new_data %>%
      dplyr::filter(fold != i) %>%
      dplyr::select(-fold)
    
    test = new_data %>%
      dplyr::filter(fold == i) %>%
      dplyr::select(-fold)
    
    list_data[[i]] = list(train = train, 
                          test = test)
  }
  
  return(list_data)
  
}

# Bootstrap
# Data split
# Repeated k-fold CV
# Leave one out CV

# = = = = = = = = = = = = = = = # ----
#     Metrics for error         # ----
# = = = = = = = = = = = = = = = # ----

use_metric <- function(pred, real, metric){
  data = list(pred = pred, real = real)
  
  if(grepl(x = metric, pattern = 'accuracy')){
    class(data) = "accuracy"
  }
  execute_metric(data)
}
# Execute by the class (metric)
execute_metric <- function(data){
  UseMethod("execute_metric")
}

# Default
execute_metric.default = function(data){
  print("Please, choose a valid metric for evaluate.")
}

# Accuracy
execute_metric.accuracy = function(data){
  error = mean(data[['pred']] == data[['real']])
  return(error)
}



# = = = = = = = = = = = = = = = # ----
#      Apply the model          # ----
# = = = = = = = = = = = = = = = # ----

# Evaluate the expressions and call the error function
eval_models <- function(dataset, expression, metric){
  data = dataset[['train']]
  test = dataset[['test']]
  
  my_expression = paste("fit <- ", expression)
  eval(parse(text = my_expression))
  
  pred = predict(fit, test)
  
  target_after_parent = str_split(string = expression, 
                                  pattern = "\\(")[[1]][2]
  target_name = str_split(target_after_parent, 
                          pattern = "~")[[1]][1]

  error = use_metric(pred = pred, 
                     real = test[[target_name]], 
                     metric = metric)
  return(error)
}

# Make some information using errors of the folds
resume_errors <- function(error){
  error_mean = mean(error)
  error_var = var(error)
  error_median = median(error) 
  
  out = tibble(mean = error_mean, var = error_var, median = error_median)
  return(out)
}

map_expressions <- function(expression, dataset, metric){
  dataset %>%
    map(~eval_models(., expression, metric))
}

  
# = = = = = = = = = = = # ----
#     Orquestrador      # ----
# = = = = = = = = = = = # ----
orchestrator <- function(model, data){
  
  list_control = control(model)
  names = names(list_control[['param']])
  list_control[['param']] = list_control[['param']] %>%
    map(~to_character(.))
  list_control[['param']] = put_simbols(list_control[['param']])
  
  combinations = create_expression(list_control, data)
  
  expressions = combinations[['df_expressions']]
  control = combinations[['df_control']] %>%
    dplyr::mutate(ID = 1:n())
  
  data_model = split_by_validation(data = data)
  
  expr_data = list(expr = expressions, data = data_model)
  
  errors = expr_data[['expr']] %>%
    map(~map_expressions(., expr_data[['data']], metric = data[['metric']])) %>%
    map(~unlist(.)) %>% 
    map(~resume_errors(.))
  
  
  error_df = errors[[1]]
  for(i in 2:length(errors)){
    error_df = rbind(error_df, errors[[i]])
  }
  
  error_df = error_df %>%
    dplyr::mutate(ID = 1:n())
  
  my_df = control %>% 
    left_join(error_df, by = 'ID') %>%
    dplyr::select(-ID)
  
  return(my_df)
}

obj = orchestrator(model = rf, data = data)
