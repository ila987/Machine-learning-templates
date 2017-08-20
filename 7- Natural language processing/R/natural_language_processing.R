#NATURAL LANGUAGE PROCESSING

#importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

#cleaning the texts
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

#creating bag of words
dtm = DocumentTermMatrix(corpus)
#remove unfrequent words 
dtm = removeSparseTerms(dtm, 0.999)
#transform matrix to dataframe to use random forest
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Fitting random forest classification to the Training set
# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y= training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692], type = 'class')

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
