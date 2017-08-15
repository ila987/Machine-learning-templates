# Data Preprocessing
#take care of missing data, by column, using the mean

# Importing the dataset
dataset = read.csv('Data.csv')

# Taking care of missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, dataset$Country, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, dataset$Country, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)