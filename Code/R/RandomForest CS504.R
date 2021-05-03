data<-read.csv('Final_Data_With_Outliers.csv')
library(DescTools)
library(MASS)
library(randomForest)

#partition data
set.seed(100)
trainrows<-sample(nrow(data),nrow(data)*.8, replace = FALSE)
traindata<-data[trainrows,]
testdata<-data[-trainrows,]

#run rf
rf<-randomForest(x=traindata[,-41],y=traindata[,41],ntree=100,mtry=5,do.trace=1)
varImpPlot(rf, main="Random Forest Top 10 Variables", n.var=10)

#predict
rfpredictions<-predict(rf, newdata = testdata)

#calculate statistics
RMSE(rfpredictions,testdata[,41])

summary(lm(testdata[,41]~rfpredictions))


