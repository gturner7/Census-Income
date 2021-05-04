Correlation Matrix and plotting Sample Code - 
correlatedData <-cor(datafraem[column], dataframe[column], method="pearson/spearman")  #this creates a correlations to plot in the next command 
Corrplot(correlatedData, method="number", tl.cex=.5, number.cex=.4 #plot the corrleated data with smaller numbers and axis labels ot fit on the screen 
         Scatterplot Sample Code - 
           options(scipen=50)    # this tells R that we are not using scientific notation, but integers  
         clswkrAndSalaryOUTLIERS<-ds_Outliers_ORGNAMES[c("A_CLSWKR", "PEARNVAL")] #uses those 2 pertinent columns from the dataframe 
         clswNames<-c("None", "Private", "FED GOV", "State GOV", "Local GOV", "Self-INC", "Self-NoINC", "No Pay", "NVR WRKD") # uses the Class wof wokers from the dataset to be used ina future step as the X Axis 
plot(classWorkerandSalaryNOOUTLIERS, xlab="", ylab="Salary") # plots the scatterplot 
axis(1,at=0:8, labels=clswNames, las=2) # adds the Classes of workers as the X axis 
