# K-MEANS CLUSTERING

dataset <- read.csv('Mall_Customers.csv')
X<- dataset[4:5]

#define how many clusters we need with elbows 
set.seed(6)
wcss <- vector()
for (i in 1:10)
  wcss[i]<- sum(kmeans(X,i)$withinss)
plot(1:10, wcss, type='b', main = paste('Cluster of clients'), 
     xlab  = 'number of cluster',
     ylab ='wcss')

#fit k-means with the optimal number of cluster
kmeans<- kmeans(X,5, iter.max= 300, nstart=10)
 
#visualize the clusters
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar= FALSE,
         span=  TRUE,
         main = paste ('Cluster of clients'),
         xlab = 'Annual income',
         ylab = 'spending score')