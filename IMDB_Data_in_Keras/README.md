# Mini Project: Using Keras to analyze IMDB Movie Data 

In this project, we will analyze a dataset from IMDB and use it to predict the sentiment analysis of a review.

The dataset
This lab uses a dataset of 25,000 IMDB reviews. Each review comes with a label. A label of 0 is given to a negative review, and a label of 1 is given to a positive review. The goal of this lab is to create a model that will predict the sentiment of a review, based on the words in the review. You can see more information about this dataset in the Keras website.

Now, the input already comes preprocessed for us for convenience. Each review is encoded as a sequence of indexes, corresponding to the words in the review. The words are ordered by frequency, so the integer 1 corresponds to the most frequent word ("the"), the integer 2 to the second most frequent word, etc. By convention, the integer 0 corresponds to unknown words.

Then, the sentence is turned into a vector by simply concatenating these integers. For instance, if the sentence is "To be or not to be." and the indices of the words are as follows:

"to": 5
"be": 8
"or": 21
"not": 3
Then the sentence gets encoded as the vector [5,8,21,3,5,8].
