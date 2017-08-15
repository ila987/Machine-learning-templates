#POLYNOMIAL REGRESSION
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

#fitting linear regression
lin_reg = lm(formula = Salary ~., data = dataset)
summary(lin_reg)

#fitting polinomial regression
#add poly level to the dataset
dataset$Level2= dataset$Level^2 
poly_reg = lm(formula = Salary ~., data = dataset)
summary(poly_reg)

#visualize linear regression
library(ggplot2)
ggplot()+ 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') + 
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), colour = 'blue') + 
  ggtitle('Truth vs bluff linear regression') + 
  xlab ('level') + 
  ylab('salary')


#visualize polynomial regression 
library(ggplot2)
ggplot()+ 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'blue') + 
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), colour = 'blue') + 
  ggtitle('Truth vs bluff polynomial regression') + 
  xlab ('level') + 
  ylab('salary')

#predict a new result with linear regression
y_pred = predict(lin_reg, data.frame(Level=6.5))

#predict a new result with poly regression
y_pred = predict(lin_reg, data.frame(Level=6.5,
                                     Level2=6.5^2,
                                     Level3=6.5^3,
                                     Level4=6.5^4))

