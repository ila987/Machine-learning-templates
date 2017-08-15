#SVR
#import data 
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]


#let's split between train and test set
#library(caTools)
#set.seed(123)
##true goes to training, false to test
#split= sample.split(dataset$Profit, SplitRatio = 0.8)
#training_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

#fitting SVR
library(e1071)
regressor = svm(formula = Salary ~., 
                data = dataset, 
                type = 'eps-regression')

#predict a new result with SVR
y_pred = predict(regressor, data.frame(Level=6.5))

#visualize SVR regression 
library(ggplot2)
ggplot()+ 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'blue') + 
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), colour = 'blue') + 
  ggtitle('Truth vs bluff polynomial regression') + 
  xlab ('level') + 
  ylab('salary')





