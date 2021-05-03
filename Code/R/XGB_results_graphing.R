library(ggplot2)

plotme<-data.frame(predictions,test.labels)

error=abs(predictions-test.labels)


ggplot(plotme)+aes(x=test.labels,y=predictions, colour = error)+
  geom_point(alpha=.2, shape=16)+
  scale_colour_gradientn(colours=c('black','red'), name ='Error',
                         limits=c(0,20000),oob = scales::squish)+ 
  labs(x='Actual', y='Prediction', title='Actual vs. Predicted Income Error')+
  theme(plot.title = element_text(hjust = 0.5))

error2=(predictions-test.labels)

ggplot(as.data.frame(error2)) + aes(x=error2) + geom_histogram(binwidth=8000) + 
  xlim(c(-40000,40000)) + labs(title='Modeling Error Distribution')