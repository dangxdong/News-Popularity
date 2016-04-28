doANNh2o = function(cvlists, test=test) {
    
    Nfold = length(cvlists$lbCVs)
    all_idx = cvlists$all_idx
    # suppose that h2o is already initialized before this function is called
    test.h2o = as.h2o(test)
    if ("predcv_temp" %in% ls()) { rm(predcv_temp) }
    if ("predtest_temp" %in% ls()) { rm(predtest_temp) }
    for (i in 1:Nfold) {
        train1 = cvlists$trainDFs[[i]]
        cv1 = cvlists$cvDFs[[i]]
        train1_lbs = cvlists$lbTrains[[i]]
        cv1_lbs = cvlists$lbCVs[[i]]
        
        train1.h2o = as.h2o(train1)
        cv1.h2o = as.h2o(cv1)
        
        train1.h2o[,58] = as.factor(train1.h2o[,58])  # essential !!!
        cv1.h2o[,58] = as.factor(cv1.h2o[,58])  # essential !!!
        
        # below is the best setting with 5 and 2 units in a two-hidden-layer model
        model1 =
            h2o.deeplearning(x = 1:57,  # column numbers for predictors
                             y = 58,   # column number for label
                             training_frame = train1.h2o, # data in H2O format
                             validation_frame = cv1.h2o,
                             activation = "Maxout",
                             input_dropout_ratio = 0, # % of inputs dropout
                             hidden_dropout_ratios = c(0, 0), # % for nodes dropout
                             hidden = c(5, 2), # one layer of 60 nodes
                             momentum_stable = 0.99,
                             nesterov_accelerated_gradient = T, # use it for speed
                             epochs = 15) # after test, 15 epochs are enough
        pred_cv1.h2o = h2o.predict(model1, cv1.h2o)
        pred_cv1.df = as.data.frame(pred_cv1.h2o[,3]) # only get the third column to save time
        pred_cv1 = pred_cv1.df$p1

        if ("predcv_temp" %in% ls()) {
            predcv_temp = c(predcv_temp, pred_cv1)
        } else {
            predcv_temp = pred_cv1
        }
        
        pred_test1.h2o = h2o.predict(model1, test.h2o)
        
        # only get the third column as the predicted probabilities.
        pred_test1.df = as.data.frame(pred_test1.h2o[,3]) 
        pred_test1 = pred_test1.df$p1
        
        if ("predtest_temp" %in% ls()) {
            predtest_temp = predtest_temp + pred_test1
        } else {
            predtest_temp = pred_test1
        }
    }
    ann_pred_train = predcv_temp[order(all_idx)]
    ann_pred_test = predtest_temp / Nfold
    
    return(list(ann_pred_train=ann_pred_train, ann_pred_test=ann_pred_test))
    
}