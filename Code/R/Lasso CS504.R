#Lasso regression sample

library(DescTools)
library(glmnet)

x <- read.csv('Final_Data_No_Outliers.csv')
y <- x[,41]
x <- x[,-41]
nr <- nrow(x)
trainingSize <- nr * .8 # 80% training size

#sets a random seed
set.seed(100)
train <- sample(1:nr,trainingSize)

#partition and label data
x.train <- x[train,]
y.train <- y[train]
head(x.train)
head(y.train)

x.test <- x[-train,] 
y.test <- y[-train]
head(x.test)

#Show variable plot 
lasso.mod <- glmnet(x[train,],y[train], alpha=1)
plot(lasso.mod,las=1)

#show tuning parameter plot (lambda)
cv.out <- cv.glmnet(data.matrix(x[train,]), y[train] ,alpha=1)
plot(cv.out)
title("Lasso Tuning and Variable Selection", line = 2.5)

#lambda value for most powerful regression
lambda.min <- cv.out$lambda.min

#predict values using best lambda
lasso.predBest <- predict(lasso.mod, s=lambda.min, newx=as.matrix(x.test))

#Display RMSE and r2
RMSE(lasso.predBest,y.test)
summary(lm(lasso.predBest~y.test))
