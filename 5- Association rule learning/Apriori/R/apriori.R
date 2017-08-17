#APRIORI

#data preprocessing
dataset = read.csv('Market_Basket_Optimisation.csv', header= FALSE)

#Create a sparse matrix
library(arules)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN)

#training apriori on dataset
#support: products bought at least 3 times per day 3*7 /7500
rules = apriori(data = dataset, parameter = list(support= 0.003, confidence = 0.2))

#visualizing the results
#sort rules by life
inspect(sort(rules, by = 'lift')[1:10])
