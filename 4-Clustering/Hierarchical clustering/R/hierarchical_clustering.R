#HIERARCHICAL CLUSTERING


dataset <- read.csv('Mall_Customers.csv')

X<- dataset[4:5]

#define how many clusters we need with dendrogram 
dendrogram= hclust(dist(X, method = 'euclidean'), method = 'ward.D' )
plot(dendrogram, 
     main = paste('Dengrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distances')

#fit hierarchical clustering the optimal number of cluster
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D' )
y_hc =  cutree(hc, k= 5)

#visualize the clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar= FALSE,
         span=  TRUE,
         main = paste ('Cluster of clients'),
         xlab = 'Annual income',
         ylab = 'spending score')