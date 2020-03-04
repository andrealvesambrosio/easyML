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

dataset = as_tibble(iris)

# Input params
rf = list(
           fun = "randomForest",
           package = "randomForest",
           importance = c(TRUE, FALSE),
           ntree = c(100, 250, 500, 750, 1000),
           mtry = c(2, 4)
          )

data = list(
             data = dataset,
             target = "Species",
             validation = "kfold_7"
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

###########################

# Dataframe's combinations
create_expression <- function(obj, data){
  obj_param = obj[['param']]
  obj_model = obj[['model']]
  
  if("formula" %in% names(data)){
    formula = data[['formula']]
  } else{
    formula = paste0(data[['target']], "~.")
  }
  df = expand.grid(obj_param) %>%
    as_tibble() %>%
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
  
  return(df)
}

# = = = = = = = = = = = = = = = # ----
#   Algorithm for validation    # ----
# = = = = = = = = = = = = = = = # ----

# Choose the method
split_by_validation <- function(method){
  method = data[['validation']]
  df = data[['data']]
  
  # K-fold
  if(grepl(x = method, pattern = 'kfold')){
    fold = str_split(string = method, pattern = "_")[[1]][2]
    if(is.na(fold)){
      fold = 10
    } else{
      fold = as.numeric(fold)  
    }
    data_list = kfold(df, fold)
  }
  
  return(data_list)
}

# Methods = = = = = = = =

# K-fold Cross Validation
kfold <- function(x, folds){
  len <- nrow(x)
  index <- cut(seq(1, len), 
               breaks = folds, 
               labels = FALSE) %>%
    sample()
  
  new_data <- x %>% 
    dplyr::mutate(fold = index) 
  
  list_data = list()
  for(i in 1:folds){
    train = new_data %>%
      dplyr::filter(fold != i)
    
    test = new_data %>%
      dplyr::filter(fold == i)
    
    list_data[[i]] = list(train = train, 
                          test = test)
  }
  
  return(list_data)
  
}

# Bootstrap
# Data split
# Repeated k-fold CV
# Leave one out CV

# = = = = = = = = = = = # ----
#     Orquestrador      # ----
# = = = = = = = = = = = # ----
orchestrator <- function(model, data){
  
  list_control = control(model)
  names = names(list_control[['param']])
  list_control[['param']] = list_control[['param']] %>%
    map(~to_character(.))
  list_control[['param']] = put_simbols(list_control[['param']])
  
  expression = create_expression(list_control, data)
  
  data_model = split_by_validation(data[['validation']])
  
  output = list(expr = expression, data = data_model)
  return(output)
}

obj = orchestrator(model = rf, data = data)


# ===========================================================
list_data <- list(list(treino = "Treino 1", teste = "Teste 1"),
                  list(treino = "Treino 2", teste = "Teste 2"),
                  list(treino = "Treino 3", teste = "Teste 3"))

list_expressions <- list("expression 1", "expression 2")


my_function <- function(exp, df){
  df %>%
    map(~second_function(., exp))
}

second_function <- function(df, exp){
  paste(df$teste, df$treino, exp)
}

list_expressions %>%
  map(~my_function(., list_data))
#