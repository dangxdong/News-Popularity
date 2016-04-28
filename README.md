# News-Popularity

Source code for a course project in data mining.

The project stems from a parent study (http://link.springer.com/chapter/10.1007%2F978-3-319-23485-4_53), to predict the popularity of online news, with extracted features from both content and meatadata.

Stacked models are tried on top of logistic regression, KNN, Naive Bayes, random forest and ANN at level-0. At level-1, the average of the level-0 predictions in probabilities is used as a baseline. Then logistic regression, non-negative-coefficient linear regression and ANN are used and evaluated.

The project is a team work of four. I contributed the wrapper functions in it.

The entry files are '0.data preparation.r' for data preparation, exlploration and initial analyses, and '1.stackingNotesNew.r' for modelling. 

The data file can be obtained from http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity.
