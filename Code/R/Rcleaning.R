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
new<-merge(data,hh_subset,by='H_IDNUM')

#export new combined data
write.csv(new, file='Hydrogen_Data.csv')
