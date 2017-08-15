##MULTIPLE LINEAR REGRESSION
#import and split data between train and test set, fit multiple regression, 
#predict test set. Build best model using backward elimination 

#import data 
dataset = read.csv('50_Startups.csv')

#handle categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))

#let's split between train and test set
library(caTools)
set.seed(123)
#true goes to training, false to test
split= sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#fitting multiple regression
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = training_set)
summary(regressor)

#Predict test set results
y_pred = predict(regressor, newdata = test_set)

#building the optimal model using backward elimination
#fitting multiple regression
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = dataset)
summary(regressor)
#remove var with higher p value
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, data = dataset)
summary(regressor)
#remove var with higher p value
regressor = lm(formula = Profit ~ R.D.Spend  + Marketing.Spend, data = dataset)
summary(regressor)
