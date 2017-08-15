#Decision tree regression

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

#fitting Decision tree
library(rpart)
regressor = rpart(formula= Salary~., 
                  data = dataset, 
                  control = rpart.control(minsplit = 1))

#predict a new result with decision tree
y_pred = predict(regressor, data.frame(Level=6.5))

#visualize decision tree regression (for higher resolution because decision tree is not continuous)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot()+ 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'blue') + 
  geom_line(aes(x = x_grid , y = predict(regressor, newdata = data.frame(Level= x_grid))), colour = 'blue') + 
  ggtitle('Truth vs bluff decision tree regression') + 
  xlab ('level') + 
  ylab('salary')





