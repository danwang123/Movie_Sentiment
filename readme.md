
###Introduction:

Sentiment analysis is a well-known task in the realm of natural language processing. The objective of sentiment analysis is to determine the polarity of the given a set of texts, which can be understood as predict the attitude
of the user towards a topic with a short paragraph or sentence given by the user. There are great numbers of sentiment analysis application scenarios in real life, including Tweets on hot topics, review on products, movies, books, and comments on news event.

In this project, I implement several di↵erent natural language processing(NLP) methods to conduct the sentiment analysis on movie reviews in order to compare di↵erent semantic representation methods and classifiers. I compare
di↵erent representation for text in two ways: Bag of Words (BOW) and Word Vectors (Word2Vec). After extracting the features from each review, I apply Random Forest (RF), Logistic Regression(LR) as long as Support Vector Machine (SVM) classifier on it and predict reviewers attitudes towards the movie:positive and negative.

###EDA:

The sentiment of reviews is binary, meaning the IMDB rating < 5 results.in a sentiment score of 0, and rating>=7 have a sentiment score of 1. No individual movie has more than 30 reviews. The dominant token size of each
individual review is about 200 and Fig. 1 shows that on average there are 267.9 tokens for each review. The 25,000 review labeled training set does not include any of the same movies as the 25,000 review test set. In addition,
there are another 50,000 IMDB reviews provided without any rating labels.
###![](https://github.com/danwang123/Movie_Sentiment/blob/master/Images/review_length.jpeg?raw=true)


###Flowchart:

###![](https://github.com/danwang123/Movie_Sentiment/blob/master/Images/framework.jpeg?raw=true)
