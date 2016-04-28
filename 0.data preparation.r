library(data.table)

newspop = fread("OnlineNewsPopularity.csv")
newspop = as.data.frame(newspop)

newsurl = newspop$url
timedelta = newspop$timedelta
newspop = newspop[3:61]

cormatrix = cor(newspop[1:58])
cormatrix = data.frame(cormatrix)

namesX = names(cormatrix)   # 58 predictor variables

corrnames = c()
for (i in 1:57) {
    corrnames1 = namesX[which(cormatrix[,i] > 0.99)]
    idx = which(corrnames1==namesX[i])+1
    if (idx <= length(corrnames1)){
        corrnames1 = corrnames1[idx:length(corrnames1)]
        corrnames = c(corrnames, corrnames1)
    }
}
corrnames = unique(corrnames)

# It turned out "n_non_stop_words" and "n_non_stop_unique_tokens" are 
# highly correlated with "n_unique_tokens" (cor > 0.99). Consider to remove the two variables.

# but check the data closer, find outliers (wrong records)
plot(newspop$n_unique_tokens, newspop$n_non_stop_words)

summary(newspop$n_unique_tokens)
newspop$n_unique_tokens[newspop$n_unique_tokens >1]
which(newspop$n_unique_tokens >1) 
# turned out the row 31038 is problamatic , it has n_unique_tokens=701 
# n_non_stop_unique_tokens = 650, and n_non_stop_words=1042, 
#  which should all have been between 0 and 1.

# fix it:
newspop$n_unique_tokens[31038] = 0.701
newspop$n_non_stop_words[31038] = 1
newspop$n_non_stop_unique_tokens[31038] = 0.65

# Then check correlation again!!
cormatrix = cor(newspop[1:58])
cormatrix = data.frame(cormatrix)
namesX = names(cormatrix)  
corrnames=c()
for (i in 1:57) {
    corrnames1 = namesX[which(cormatrix[,i] > 0.9)]
    idx = which(corrnames1==namesX[i])+1
    if (idx <= length(corrnames1)){
        corrnames1 = corrnames1[idx:length(corrnames1)]
        corrnames = c(corrnames, corrnames1)
    }
}
corrnames = unique(corrnames)

# Now with the threshold 0.9, we have "n_non_stop_unique_tokens" highly correlated 
# with "n__unique_tokens"
# check it
plot(newspop$n_unique_tokens, newspop$n_non_stop_unique_tokens)
# confirmed

# also have "average_token_length" highly correlated 
# with "n_non_stop_words". Double check!
plot(newspop$n_non_stop_words, newspop$average_token_length)
summary(newspop$n_non_stop_words)
# because n_non_stop_words has most values as 1 and very few as 0.
# this is strictly correlated to average_token_length.
# consider to remove n_non_stop_words .

# also have "kw_avg_min" highly correlated 
# with "kw_max_min". Double check!
plot(newspop$kw_max_min, newspop$kw_avg_min)
# confirmed.

# to be conservative, do not remove any of the highly correlated coloumns,
# because they may provide extra information.

# And as a good practice for recoding categorical features, 
#  we only need six columns for the weekday features. Removed one of the seven columns
newspop$weekday_is_sunday = NULL

newspopdata = data.frame(newsurl, timedelta, newspop)

write.csv(newspopdata, "OnlineNewsPopularity_prepared.csv", row.names=F)
# Later use this csv file for further study. use the columns 3:59 as predictor features
# the last column "shares" is the outcome to be predicted.
# it is again a count number problem, we are already experienced with it!



# So the data preparation stage finally only fixed one row of outlier values,
# and removed one weekday column.

# the data frame, newspop, now has 57 predictor features and one outcome column.

# two extra columns, the newsurl and the timedelta, are stored as separate vectors
# only to be used finally to combine the results.


summary(newspop[,2])
hist(newspop[,57])


# do a simple poisson regression
glm1 = glm(shares ~ ., data=newspop, family="poisson")
summary(glm1)

pred1 = predict(glm1, type="response")

summary(pred1)
source("calcRMSE.r")
source("calcRMSLE.r")


table(newspop$shares>=1400, pred1>=1400)
# accuracy == (21104+190)/(21104+190+50+18300) == 0.537


# check the performance of the prediction
calcRMSE(pred1, newspop$shares)   # 11547.31
calcRMSLE(pred1, newspop$shares)  # 1.051596
plot(pred1, newspop$shares)

# linear regression
glm2 = glm(shares ~ ., data=newspop, family="gaussian")
summary(glm2)

pred2 = predict(glm2, type="response")
pred2[pred2<0] = 0
calcRMSE(pred2, newspop$shares)   # 11491.65
calcRMSLE(pred2, newspop$shares)  # 1.095848 
plot(pred2, newspop$shares)

hist(newspop$shares, breaks = 200)

table(newspop$shares>20000)

hist(newspop$shares[newspop$shares<20000], breaks = 100)
summary(newspop$shares)

lbs = newspop$shares

# transform the label column to a binary categorical variable.
# becaue the original article paper did the work this way.
lbscut = lbs
lbscut[lbscut<=1400] = 0
lbscut[lbscut>1400] = 1

newspop1 = newspop[1:57]
newspop1$lbscut = as.integer(lbscut)
newspop1$lbscut = as.factor(newspop1$lbscut)
# so we have the new data set newspop1, which has all the predictors
# and the label column in 0 / 1 format. 0 is unpopular, 1 is popular.


# As the original paper did, we should also remove the rows with timedelta <=21

newspop1 = newspop1[timedelta>21, ]

# so the new data set has only 39016 rows. The original paper had only 39000 rows,
# but we don't know which 16 rows to further remove.  So we just keep them.

# save the prepared new set:
write.csv(newspop1, file="OnlineNewsPopularity_categorical_lable.csv", row.names=F)

# do a logistic regression:
glm3 = glm(lbscut ~ ., data=newspop1, family="binomial")
summary(glm3)

pred3 = predict(glm3, type="response")

tbl1 = table(pred3 > 0.5, newspop1$lbscut)

source('getperformance.r')

getperformance(pred3, newspop1$lbscut)


#               0     1
#     FALSE 13361  7131
#     TRUE   6403 12121
# accuracy is (13361+12121)/(6403+7131+13361+12121) == 0.6531167.
# the original article paper has got accuracy of 0.73 with an advanced method.
# and 0.62 - 0.67 for other methods like KNN, SVM, ...


# because the original paper uses AUC to measure the model, we will also use AUC

# note here the AUC is on the training set itself. We should do cross-validation
# and get the AUC on the validation or test data, to be compared with the results
# in the original paper.
library(ROCR)

getperformance(pred3, newspop1$lbscut)

ROCRpred1 = prediction(pred3, newspop1$lbscut)
ROCRperf1 = performance(ROCRpred1, "tpr", "fpr")
# Get a plotting to see how ROC is like
plot(ROCRperf1, colorize=TRUE)

# Get the AUC value:
auc1 = as.numeric(performance(ROCRpred1, "auc")@y.values)
auc1    # 0.7049

# Find the threshold to divide the first and second levels
# create a data frame from the performence data
cutoffs1 = data.frame(cut=ROCRperf1@alpha.values[[1]], fpr=ROCRperf1@x.values[[1]], 
                       tpr=ROCRperf1@y.values[[1]])

# Use a new score, tpr*tnr/(tpr=tnr), to decide which threshold is best 
cutoffs1$NewScore = cutoffs1$tpr * (1-cutoffs1$fpr) / (cutoffs1$tpr + (1-cutoffs1$fpr))

# then just get the threshold with the max of new score
thresh1 = cutoffs1$cut[which.max(cutoffs1$NewScore)]

thresh1 # 0.4942487

# try the classification confusion table again:

table(pred3 >= thresh1, newspop1$lbscut)
#              0     1
#    FALSE 13151  6914
#    TRUE   6613 12338
# the accuracy is (13151+12338)/(13151+12338+6613+6914) == 0.6532961
# compared to 0.6531167 when simply using 0.5 as the threshold.
# so the best threshold is not much better than simplely using 0.5

