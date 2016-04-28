
## This is the stacking note for group K.
# first, set your working directory to the folder where you put all the data and r files.
# in my case it is:
#    setwd("C:/Users/.../OnlineNewsPopularity")
     
#### I strongly recommend that we use the same train and test sets throughout our project:
# set the predefined train and test sets.

# read in the data and save the lable column into a separate vector:
newspop = read.csv("OnlineNewsPopularity_categorical_lable.csv")
lbscut = as.factor(newspop$lbscut)

library(caTools)
set.seed(3031)       # don't change this, so we are always using the same train + test sets
split = sample.split(lbscut, SplitRatio = 0.7)
train = subset(newspop, split == T)
test = subset(newspop, split == F)
train_lbs = subset(lbscut, split == T)
test_lbs = subset(lbscut, split == F)

# 
# Do a five-fold cross-validation test, for every different method.
# 
# Using the prepared r functions to get a list of the splitted data frames:
source('doNfoldsplit.r')
source('getCVsets.r')

Nfold = 5
cvlists = getCVsets(train, train$lbscut, n=Nfold, seed=3030)  
# to store the sub-train and cv sets (5 pairs for 5-fold)
# The seed number can be changed so repeated works can generate some deviation, 
# based on which the stability of each model is evaluated.

# to help sort the combined predictions on the train set.  
# all_idx = cvlists$all_idx

# Load these packages. Install them if you haven't done so.

library(randomForest)
library(class)
library(e1071)
library(caret)
library(klaR)  # needed for NB
library(pROC)  # needed for training NB with caret
library(h2o)   # used for ANN
library(psych)  # needed for NB, use PCA to get uncorrelated variables.
library(nnls)  # a multi-response linear regression with non-negative-coefficient constraint

# some helper functions I have written: Pease keep them in the same folder as the data files.

source("calclogloss.r")
source("getperformance.r")   # will need the library "ROCR" installed
source('doRF.r')
source('doLogiReg.r')
source('doKNN.r')
source('doNB.r')
source('doANNh2o.r')     # need the library "h2o" installed
source('pred.transform.r')

##  1. do logistic regression with "doLogiReg.r" 
source('doLogiReg.r')
glm_pred_list = doLogiReg(cvlists, test=test)  # there will be warnings, don't worry
# get predictions
glm_pred_train = glm_pred_list$glm_pred_train
glm_pred_test = glm_pred_list$glm_pred_test
#  see performances:
calclogloss(glm_pred_test, test$lbscut)      # logloss=0.6317896
getperformance(glm_pred_test, test$lbscut, getplot=F, simple=F)  #auc=0.6988656, acc05=0.6475865, acc=0.64827

# For getperformance(), 
# use getplot=c(T, F) to control if plotting the ROC curve
# and
# use simple=c(T, F) to control if only the AUC, threshold and best accuracy are returned, 
# or all the metrics are returned including the confusion matrix, recall, precision and specificity.

# Check performance on the training set to see if there is any overfitting
calclogloss(glm_pred_train, train$lbscut)   # logloss=0.6307993
getperformance(glm_pred_train, train$lbscut, getplot=F, simple=F)
# AUC=0.7035839,  accubest=0.6530336


# 2. similarly, made "doKNN.r" 

source('doKNN.r')

#  First, do normalization on the whole set, which is essential when doing KNN.

# first, normalize (scale) the whole dataset including the final test set.
normalize = function(x) { return ((x - min(x)) / (max(x) - min(x))) }
news_norm = as.data.frame(lapply(newspop[1:57], normalize))
news_norm = cbind(news_norm, lbscut=newspop$lbscut)

# split into the same sets of rows as before:
# remember we did this. do it again with the same seed value 3031 if "split" is removed.
# set.seed(3031)
# split = sample.split(lbscut, SplitRatio = 0.7)
train_norm = subset(news_norm, split == T)
test_norm = subset(news_norm, split == F)

# split the train_norm set as before
Nfold = 5
cvlists_norm = getCVsets(train_norm, train$lbscut, n=Nfold, seed=3030)

# do the modeling on the normalized sets:  
knn_pred_list = doKNN(cvlists_norm, test=test_norm)

# and get the predictions:
knn_pred_train = knn_pred_list$knn_pred_train
knn_pred_test = knn_pred_list$knn_pred_test

# see performance on test
calclogloss(knn_pred_test, test$lbscut)      # logloss=0.6483027
getperformance(knn_pred_test, test$lbscut, getplot=F, simple=F) 
#auc=0.6707504, acc05=0.6228108, accubest=0.6240068

# and check the prediction on train set:
calclogloss(knn_pred_train, train$lbscut)      # logloss= 0.6476359
getperformance(knn_pred_train, train$lbscut) #auc=0.6740216, accubest=0.6305884


# Conclusion: the performance is slightly worse than the wrong way! 
# maybe just noise, in fact they are equally well due to homogenerity of the data.

# 3. make "doNB.r"
library(caret)
library(klaR)

source('doNB.r')

## special notes for NB: the best practice is to do PCA, and use all extracted 
## features instead of the original ones, 

## Because Naive Bayes works ideally when the predictors are independent (orthogonal) from each other.
## Do PCA:
library(psych)
fit_pca = principal(newspop[,1:57], nfactors=57, rotate="varimax")
newspop_pca = data.frame(fit_pca$scores)
newspop_pca = cbind(newspop_pca, lbscut=newspop$lbscut)
#  summary(fit_pca$loadings)
#  plot(fit_pca$values,type="lines")
train_pca = subset(newspop_pca, split == T)
test_pca = subset(newspop_pca, split == F)

# split the train_pca set as before
Nfold = 5
cvlists_pca = getCVsets(train_pca, train$lbscut, n=Nfold, seed=3030)

# Now do NB again with the PCA data:
source('doNB.r')
nb_pred_list = doNB(cvlists_pca, test=test_pca)
nb_pred_train = nb_pred_list$nb_pred_train
nb_pred_test = nb_pred_list$nb_pred_test

calclogloss(nb_pred_test, test$lbscut)      # logloss= 1.048274 (0.9421386 after transform)
getperformance(nb_pred_test, test$lbscut, getplot=F, simple=F) #auc=0.6912445, acc=0.6402392

# check the prediction on the train set
calclogloss(nb_pred_train, train$lbscut)      # logloss= 1.048274
getperformance(nb_pred_train, train$lbscut, getplot=F) #auc=0.6912445, acc=0.6402392
summary(nb_pred_test)

#### Transforming the predictions from Naive Bayes,
#### because its predictions are higly skewed.
###  letting them scaled with best cutoff to 0.5.

source('pred.transform.r')
cut.nb = getperformance(nb_pred_train, train$lbscut, getplot=F)$bestcutoff
nb_pred_train = pred.transform(nb_pred_train, cut.nb)

# use the same cutoff as in train set for test set
# because we are supposed not to know the true outcome of test set 
# before doing the final modelling.
nb_pred_test = pred.transform(nb_pred_test, cut.nb)
# So both nb_pred_train and nb_pred_test are comparable with others in distribution.


#4.  do randomForest:
source('doRF.r')
rf_pred_list = doRF(cvlists, test=test)

# get the predictions (as probabilities) for the train and for the test set.
rf_pred_train = rf_pred_list$rf_pred_train
rf_pred_test = rf_pred_list$rf_pred_test

### see how well the prediction is on the test set:
#  get the logloss value, the smaller the better
calclogloss(rf_pred_test, test$lbscut)       # logloss=0.6136792
getperformance(rf_pred_test, test$lbscut, getplot=F, simple=F) 
#auc=0.7208129, acc=0.6615976

# check prediction on the train set:
getperformance(rf_pred_train, train$lbscut, getplot=F, simple=F)

#### 5.  ANN modeling with h2o

library(h2o)

h2o.init(ip = 'localhost', port = 54321, max_mem_size = '4g')
source('doANNh2o.r')
ann_pred_list = doANNh2o(cvlists, test=test)

h2o.removeAll()
h2o.shutdown(prompt = F)

ann_pred_train = ann_pred_list$ann_pred_train
ann_pred_test = ann_pred_list$ann_pred_test

calclogloss(ann_pred_test, test$lbscut)      # logloss= 0.6197815
getperformance(ann_pred_test, test$lbscut, getplot=F, simple=F) # AUC=0.7119782, accubest=0.6532251

calclogloss(ann_pred_train, train$lbscut)      # logloss= 0.6224054
getperformance(ann_pred_train, train$lbscut) # AUC=0.7113221, accubest=0.6559262



##### After doing all the level-0 predictions, we are going to the level-1

##  the predictions of the level-0 models are taken as the input data for level-1
# so they are the input of the level-1 model.

### finally all the predictions from the five different models are put together

predtraindf = data.frame(logi_pred=glm_pred_train, knn_pred=knn_pred_train,
                         nb_pred= nb_pred_train, rf_pred=rf_pred_train,
                         ann_pred=ann_pred_train, lbscut=train_lbs)

predtestdf = data.frame(logi_pred=glm_pred_test, knn_pred=knn_pred_test,
                        nb_pred=nb_pred_test, rf_pred=rf_pred_test,
                        ann_pred=ann_pred_test, lbscut=test_lbs)

summary(predtraindf); summary(predtestdf)

###### Do the Level-1 stacked model!!! ####

#### 2.0 simply get the average:

avg_pred_test = rowMeans(predtestdf[,1:5])
calclogloss(avg_pred_test, test$lbscut)   # logloss=0.6181755
getperformance(avg_pred_test, test$lbscut, getplot=F, simple=F)  
# AUC=0.7155402, accubest=0.6555318

#### 2.1 get weighted average, weighed by the accuracies of the models at level-0

getperformance(predtestdf[,1], test$lbscut, getplot=F, simple=T)$accuracy.best
getperformance(predtestdf[,2], test$lbscut, getplot=F, simple=T)$accuracy.best
getperformance(predtestdf[,3], test$lbscut, getplot=F, simple=T)$accuracy.best
getperformance(predtestdf[,4], test$lbscut, getplot=F, simple=T)$accuracy.best
getperformance(predtestdf[,5], test$lbscut, getplot=F, simple=T)$accuracy.best
# c(0.64827, 0.6240068, 0.6402392, 0.6615976, 0.6532251)

weightsacc = c(0.64827, 0.6240068, 0.6402392, 0.6615976, 0.6532251)
weightsacc = weightsacc / sum(weightsacc)

avg_pred_test_w = predtestdf[,1] * weightsacc[1] + predtestdf[,2] * weightsacc[2] +
                predtestdf[,3] * weightsacc[3] + predtestdf[,4] * weightsacc[4] +
                predtestdf[,5] * weightsacc[5]
calclogloss(avg_pred_test_w, test$lbscut)   # logloss=0.6179406
getperformance(avg_pred_test_w, test$lbscut, getplot=F, simple=F)  
# AUC=0.7158187, accubest=0.6563007

## Note: using weighed average gives roughly the same performance as unweighted.

#### 3.0 use logistic regression at the new level

#### 3.1 simply using the whole train set to train the model, 
####          and predict on the test set.

glm2 = glm(lbscut ~ ., data=predtraindf, family="binomial")
glm_pred_test_2 = predict(glm2, predtestdf, type="response")
summary(glm_pred_test_2)
# check the result of performance:
calclogloss(glm_pred_test_2, test$lbscut)   # logloss=0.6098538
getperformance(glm_pred_test_2, test$lbscut, getplot=F, simple=F) 
# AUC=0.7258583, accubest=0.6622811

# check prediction on train set:
glm_pred_train_2 = predict(glm2, predtraindf, type="response")
calclogloss(glm_pred_train_2,train$lbscut)      # logloss=0.6080605
getperformance(glm_pred_train_2, train$lbscut, getplot=F)  
# AUC=0.7296785, accubest=0.6696935

# conclusion: better than the averaged result.

### 3.2 do the five fold logistic regression :

Nfold = 5
cvlists_stack = getCVsets(predtraindf, train$lbscut, n=Nfold, seed=3030)
glm_pred_list_stack = doLogiReg(cvlists_stack, test=predtestdf)
glm_pred_train_3 = glm_pred_list_stack$glm_pred_train
glm_pred_test_3 = glm_pred_list_stack$glm_pred_test
#  see performances:
calclogloss(glm_pred_test_3, test$lbscut)      # # logloss=0.6098538
getperformance(glm_pred_test_3, test$lbscut, getplot=F, simple=F)  
#auc=0.7258581, accubest=0.6622811
### Conclusion: 
# doing 5-fold is almost the same as modeling on the whole training set at level-1 stage. 

# check for overfitting by looking at the prediction on the train set:
calclogloss(glm_pred_train_3,train$lbscut)      # # logloss=0.6082222
getperformance(glm_pred_train_3, train$lbscut, getplot=F)  
# AUC=0.7294717, accubest=0.6696935
# Actually this is just close to the reported accuracy in the original study.


### 3.3 
### do a non-negative-coefficient least square regression (or say, linear regression
# with non-negative coefficient and zero intercept):
library(nnls)
A = as.matrix(predtraindf[,c(1,2,3,4,5)])  # write as this to ease experiments
b = as.integer(as.character(predtraindf[,6]))

Model_nnls = nnls(A, b)
coef_nnls = coef(Model_nnls)
coef_nnls
# [1] 0.05611497 0.00000000 0.07817344 0.60764680 0.26846571
# it means that KNN did not contribute to anything with this level-1 model.

nnls_pred_test = predtestdf[,1] * coef_nnls[1] + predtestdf[,2] * coef_nnls[2] +
    predtestdf[,3] * coef_nnls[3] + predtestdf[,4] * coef_nnls[4] +
    predtestdf[,5] * coef_nnls[5]
summary(nnls_pred_test)
# check the result of performance:
calclogloss(nnls_pred_test, test$lbscut)   # logloss=0.6099097
getperformance(nnls_pred_test, test$lbscut, getplot=F, simple=F) 
# AUC=0.725878, accubest=0.6613413

# check prediction on train set:
nnls_pred_train = predtraindf[,1] * coef_nnls[1] + predtraindf[,2] * coef_nnls[2] +
    predtraindf[,3] * coef_nnls[3] + predtraindf[,4] * coef_nnls[4] +
    predtraindf[,5] * coef_nnls[5]
summary(nnls_pred_train)
calclogloss(nnls_pred_train,train$lbscut)      # logloss=0.6089907
getperformance(nnls_pred_train, train$lbscut, getplot=F, simple=F)  
# AUC=0.729164, accubest=0.6700597
# This is the closest to the original paper. Maybe they didn't to a test set,
# but do training and prediction both on the whole set, which is not good practice.

# conclusion: 
# non-negative regression is equaly well as logistic regression.

##  3.4  because nnls showed KNN is least useful, try to exclude it from logistic regression:
glm3 = glm(lbscut ~ ., data=predtraindf[,c(1,2, 4,5,6)], family="binomial")
glm_pred_test_3 = predict(glm3, predtestdf, type="response")
summary(glm_pred_test_3)
# check the result of performance:
calclogloss(glm_pred_test_3, test$lbscut)   # logloss=0.6098342
getperformance(glm_pred_test_3, test$lbscut, getplot=F, simple=F) 
# AUC=0.7258646, accubest=0.6627082

##
##  4.0   try neural network as the stacked (level-1) model:

##  4.1  just train on the whole train set
library(h2o)
h2o.init(ip = 'localhost', port = 54321, max_mem_size = '4g')

predtraindf.h2o = as.h2o(predtraindf)
predtestdf.h2o = as.h2o(predtestdf)

# to overcome randomness, do three to ten times and take average:
h2o_pred_test_stack = rep(0, dim(test)[1])
for (ii in 1:3) {
    model_stack_ann =
        h2o.deeplearning(x = 1:5,  # column numbers for predictors
                         y = 6,   # column number for label
                         training_frame = predtraindf.h2o, # data in H2O format
                         # here there's no validation frame:  validation_frame = cv1.h2o,
                         activation = "Maxout", # algorithm
                         input_dropout_ratio = 0, # % of inputs dropout
                         hidden_dropout_ratios = c(0, 0), # % for nodes dropout
                         hidden = c(5, 2), # one layer of 60 nodes
                         momentum_stable = 0.99,
                         # nesterov_accelerated_gradient = T, don't have to use it for speed
                         epochs = 5) # after test, 10 epochs are enough
    pred_test.h2o = h2o.predict(model_stack_ann, predtestdf.h2o)
    pred_test.df = as.data.frame(pred_test.h2o[,3]) # only get the third column to save time
    h2o_pred_test_stack_temp = pred_test.df$p1
    h2o_pred_test_stack = h2o_pred_test_stack + h2o_pred_test_stack_temp
}
h2o_pred_test_stack = h2o_pred_test_stack / 3

calclogloss(h2o_pred_test_stack,test$lbscut)      # logloss=0.6095819
getperformance(h2o_pred_test_stack, test$lbscut, getplot=F, simple=F)  
# AUC=0.7028985, accubest=0.6457924
summary(h2o_pred_test)

h2o.removeAll()

##  4.2  A 5-fold CV to build ANN: not better than a single training model.
#          source('doANNh2o.stack.r')
#          ann_pred_list_stack = doANNh2o.stack(cvlists_stack, all_idx, test=predtestdf)
#          
#          ann_pred_train_stack = ann_pred_list_stack$ann_pred_train
#          ann_pred_test_stack =  ann_pred_list_stack$ann_pred_test
#          
#          calclogloss(ann_pred_test_stack,test$lbscut)      # logloss=0.6385109
#          getperformance(ann_pred_test_stack, test$lbscut, getplot=F, simple=F) 
#          # AUC=0.7061422, accubest=0.6517727
# Not significantly better than doing just one whole model.
# check overfitting with predictions on the train set:
# calclogloss(ann_pred_train_stack,train$lbscut)      # logloss=0.6912312
# getperformance(ann_pred_train_stack, train$lbscut, getplot=F) 
# AUC=0.6049915, accubest=0.5888104
## actually with underfitting on the train set !

h2o.shutdown(prompt = F)

# save workspace:
save.image("stackingSessionNew3030.RData")
