#SIMPLE LINEAR REGRESSION
#split between train and test set, fit linear regression to the training
#predict and visualize the results 

#import data 
dataset = read.csv('Salary_Data.csv')

#let's split between train and test set
library(caTools)
set.seed(123)
#true goes to training, false to test
split= sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#fitting linear regression to training data
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)
summary(regressor)

#Predict test set results
y_pred = predict(regressor, newdata = test_set)

#visualizing training set results
library(ggplot2)
ggplot() + geom_point(aes(x=training_set$YearsExperience, y = training_set$Salary), 
                      color = 'red') + 
        geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), 
                  color = 'blue')+
 ggtitle ('Salary vs Experience (training set) ') + 
  xlab("Years of experience") + 
  ylab("Salary")

#visualizing test set results
library(ggplot2)
ggplot() + geom_point(aes(x=test_set$YearsExperience, y = test_set$Salary), 
                      color = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), 
            color = 'blue')+
  ggtitle ('Salary vs Experience (training set) ') + 
  xlab("Years of experience") + 
  ylab("Salary")