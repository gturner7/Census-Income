library(DescTools)

#import datasets
data<-read.csv('pppub20.csv')
hh<-read.csv('hhpub20.csv')

#set options
options(scipen = 100)

#create subset of "hh" to merge with main data
hh_subset<-hh[,c('H_IDNUM','GESTFIPS','GTCO')]

#calculate new column in data for merging key value
data$'H_IDNUM'<-data$'PERIDNUM'%/%100

#check for matching
min(hh_subset$H_IDNUM)
min(data$H_IDNUM)

#check for duplicated ID numbers, there are many due to recording methodology
id_table<-table(data$PERIDNUM)

#merge datasets
data_with_outliers<-merge(data,hh_subset,by='H_IDNUM')

#export new combined data
write.csv(data_with_outliers, file='2020_Data_With_Outliers.csv')

#remove outliers
quantiles<-data.frame(quantile(data_with_outliers$PEARNVAL))
iqr<-quantiles['75%',]-quantiles['25%',]
maxy<-quantiles['75%',]+iqr*1.5
miny<-quantiles['25%',]-iqr*1.5
#miny has no effect

data_without_outliers<-subset(data_with_outliers, PEARNVAL>= miny & PEARNVAL<=maxy)
summary(data_without_outliers$PEARNVAL)
#summary shows values from -9,999:100,000

#export data w/out outliers
write.csv(data_without_outliers, file='2020_Data_Without_Outliers.csv')
