from __future__ import division

import os
import pandas as pd
import numpy as np
# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup  
# remove punctuation and numbers;
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 


if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )

    print 'The first review is:'
    print train["review"][0]
    
    print train.shape
    print train.columns.values

    # Initialize the BeautifulSoup object on a single movie review     
    example1 = BeautifulSoup(train["review"][0])  

    # Print the raw review and then the output of get_text(), for comparison
    print train["review"][0]
    print example1.get_text()

    #nltk.download()  # Download text data sets, including stop words
    print stopwords.words("english") 

    clean_review = review_to_words( train["review"][0] )
    print clean_review

    # Get the number of reviews based on the dataframe column size
    num_reviews = train["review"].size

    # Initialize an empty list to hold the clean reviews
    print "Cleaning and parsing the training set movie reviews...\n"
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length of the movie review list 
    for i in xrange( 0, num_reviews ):
        # If the index is evenly divisible by 1000, print a message
        if( (i+1)%1000 == 0 ):
            print "Review %d of %d\n" % ( i+1, num_reviews ) 
        # Call our function for each one, and add the result to the list of clean reviews
        clean_train_reviews.append( review_to_words( train["review"][i] ) )

    print "Creating the bag of words...\n"

    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.  
    # vectorizer = CountVectorizer(analyzer = "word",   \
    #                              tokenizer = None,    \
    #                              preprocessor = None, \
    #                              stop_words = None,   \
    #                              max_features = 5000) 
    vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
    # 

    """
    fit_transform() does two functions: First, it fits the model and learns the vocabulary; 
    second, it transforms our training data into feature vectors. 
    The input to fit_transform should be a list of  strings.
    """
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()

    print train_data_features.shape

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it appears in the training set
    for tag, count in zip(vocab, dist):
        print count, tag

    print "Training the random forest..."
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier()
    param = {'n_estimators': [10,20,50,100,200]}
    print "Training RF with CV = 10"
    forest = GridSearchCV(forest, param, cv=5)
    forest = forest.fit( train_data_features, train["sentiment"] )
    # forest = RandomForestClassifier(n_estimators = 100) 

    # print "Supervised Learning - Naive Bayes"
    # nb_model = MultinomialNB(alpha = 0.01)
    # nb_model = nb_model.fit(train_data_features, train["sentiment"]) # using BOW as feaures and the given labels as repsonse variables

    # print "---------------------------------"
    # print " "
    # print "Predicting on test data: "

    # # Read the test data
    # test = pd.read_csv("/home/meng/Dropbox/class/cs5852015fall/projects/data/testData.tsv", header=0, delimiter="\t", quoting=3)

    # Verify that there are 25,000 rows and 2 columns
    print test.shape

    # Create an empty list and append the clean reviews one by one
    num_reviews = len(test["review"])
    clean_test_reviews = [] 

    print "Cleaning and parsing the test set movie reviews...\n"
    for i in xrange(0,num_reviews):
        if( (i+1) % 1000 == 0 ):
            print "Review %d of %d\n" % (i+1, num_reviews)
        clean_review = review_to_words( test["review"][i] )
        clean_test_reviews.append( clean_review )

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    # result = nb_model.predict(test_data_features)
    result = forest.predict(test_data_features)

    print "Optimized parameters:", forest.best_estimator_
    print "Best CV score:", forest.best_score_

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

    # Use pandas to write the comma-separated output file
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model_RF_CF_CV.csv'), index=False, quoting=3 )



