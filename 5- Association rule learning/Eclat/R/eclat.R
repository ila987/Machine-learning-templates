#ECLAT
#it's a simpler version of apriori

#data preprocessing
dataset = read.csv('Market_Basket_Optimisation.csv', header= FALSE)

#Create a sparse matrix
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN)

#training eclat on dataset
#support: products bought at least 3 times per day 3*7 /7500
rules = eclat(data = dataset, parameter = list(support= 0.004, minlen = 2))

#visualizing the results of most frequently purchased
#sort rules by support
inspect(sort(rules, by = 'support')[1:10])
