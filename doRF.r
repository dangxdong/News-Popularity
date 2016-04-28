
doRF = function(cvlists, test=test) {

    Nfold = length(cvlists$lbCVs)
    all_idx = cvlists$all_idx
    
    library(randomForest) # in case the package is not loaded yet
    if ("predcv_temp" %in% ls()) { rm(predcv_temp) }    #ensure the temp objects are blank
    if ("predtest_temp" %in% ls()) { rm(predtest_temp) }
    for (i in 1:Nfold) {
        # extract the sub-train and cv sets
        train1 = cvlists$trainDFs[[i]]
        cv1 = cvlists$cvDFs[[i]]
        train1_lbs = cvlists$lbTrains[[i]]
        cv1_lbs = cvlists$lbCVs[[i]]
        
        # do the model
        rf_model_1 = randomForest(train1[,1:57], train1_lbs, mtry=18, ntree=200)
        
        # predict on the cv set
        rf_pred_cv1 = predict(rf_model_1, cv1, type="prob")[,2]
        # store the prediction
        if ("predcv_temp" %in% ls()) {
            predcv_temp = c(predcv_temp, rf_pred_cv1)
        } else {
            predcv_temp = rf_pred_cv1
        }
        
        # predict on the test set with the same model:
        rf_pred_test1 = predict(rf_model_1, test, type="prob")[,2]
        # store it
        if ("predtest_temp" %in% ls()) {
            predtest_temp = predtest_temp + rf_pred_test1
        } else {
            predtest_temp = rf_pred_test1
        }
    }
    # row-combine the predictions on the train set and sort into original order
    rf_pred_train = predcv_temp[order(all_idx)]
    # average the predictions on the test set
    rf_pred_test = predtest_temp / Nfold
    # return a list of them
    return(list(rf_pred_train=rf_pred_train, rf_pred_test=rf_pred_test))
    
}