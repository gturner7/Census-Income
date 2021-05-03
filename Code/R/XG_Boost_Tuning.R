#import packages
library(xgboost)
library(mltools)
library(data.table)
library(DescTools)
library(dplyr)
library(caret)

#read in dataset
data<-read.csv('Final_Data_No_Outliers.csv')

#select training rows 80% training and 20% testing
set.seed(100)
trainrows<-sample(nrow(data),nrow(data)*.8)

#pull out response variable
labels<-data$PEARNVAL
newdata.no.pearnval<-subset(data,select=-PEARNVAL)

#build training data
train<-as.matrix(newdata.no.pearnval[trainrows,])
train.labels<-as.matrix(labels[trainrows])

#build testing data
test<-as.matrix(newdata.no.pearnval[-trainrows,])
test.labels<-as.matrix(labels[-trainrows])

#make data xgb objects
xgtrain<-xgb.DMatrix(data=train, label=train.labels)
xgtest<-xgb.DMatrix(data=test, label =test.labels)

#regression on training data
model<-xgboost(data=xgtrain,nround=20,verbose=2)

#predict with test data - results etc. 
defaultpredictions<-predict(model,xgtest)
defaultRMSE<-RMSE(x=defaultpredictions, ref=test.labels)
defaultxglm<-lm(defaultpredictions~test.labels)
defaultr2<-summary(defaultxglm)$r.squared



#NEW TUNING TESTS

#controls the tuning param search, 
#Here it is searching for nrounds and the learning rate
tune_grid <- expand.grid(
  nrounds = seq(from=20,to=100,by=20),
  eta = seq(from=.1, to=.2, by=.025),
  max_depth = 8,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = .75
)

#specifies some conditions for validation
tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 3, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = TRUE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

#this code executes the tune search - may take a long time depending on search
xgb_tune <- caret::train(
  x = train,
  y = as.double(train.labels),
  trControl = tune_control,
  tuneGrid = tune_grid,
  method = "xgbTree",
  verbose = TRUE,
  objective = "reg:squarederror"
)

#pull out best tuning params
params<-as.list(xgb_tune$bestTune)

#execute model with new params
testmodel<-xgboost(data=xgtrain,verbose=2, params = params)

#plot search w/helper function
tuneplot <- function(x, probs = .9) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
    theme_bw()
}

#plotting call
tuneplot(xgb_tune)

#predict with new tuning params
predictions<-predict(testmodel,xgtest)
newRMSE<-RMSE(x=predictions, ref=test.labels)

xglm<-lm(predictions~test.labels)
newr2<-summary(xglm)$r.squared

data.table(newRMSE,defaultRMSE)
data.table(newr2,defaultr2)
