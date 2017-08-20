#Artificial neural network

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

#Encoding the categorical variables as factors 
# Encoding categorical data
dataset$Geography = as.numeric(factor(dataset$Geography,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                           levels = c('Female', 'Male'),
                           labels = c(1, 2)))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting classifier to the Training set
#fitting ANN to the training set
library(h2o)
#initialize to use all cores available 
h2o.init(nthreads = -1)

classifier = h2o.deeplearning(y = 'Exited', training_frame = as.h2o(training_set), 
                              activation = 'Rectifier', 
                              hidden = c(6,6), 
                              epochs = 100, 
                              train_samples_per_iteration = -2)

# Predicting the Test set results
prob_pred = h2o.predict(classifier,  newdata = as.h2o(test_set[-11]))
y_pred = (prob_pred > 0.5)
#from enviroment var to vector
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)

h2o.shutdown()
