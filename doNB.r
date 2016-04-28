doNB = function(cvlists, test=test) {
if (require('caret', quietly = TRUE)) {
    Nfold = length(cvlists$lbCVs)
    all_idx = cvlists$all_idx
    
    if ("predcv_temp" %in% ls()) { rm(predcv_temp) }
    if ("predtest_temp" %in% ls()) { rm(predtest_temp) }
    for (i in 1:Nfold) {
        train1 = cvlists$trainDFs[[i]]
        cv1 = cvlists$cvDFs[[i]]
        
        train1_lbs = cvlists$lbTrains[[i]]
        cv1_lbs = cvlists$lbCVs[[i]]
        
        train1_lbsX = train1_lbs
        levels(train1_lbsX) = c("x0", "x1")
        
        # to save time, do not tune every time. in this case,
        # fL=0 and useKernel=T have been proved to be the best,
        # so just use the fixed parameters and don't do any further sub-cv
        train_control <- trainControl(method="none", classProbs=T)
        nbGrid = expand.grid(.fL=c(0), .usekernel=c(T))
        nb_model_1 = train(x=train1[,1:57], y=train1_lbsX, method="nb",
                      metric="ROC", tuneGrid=nbGrid, trControl=train_control)
        
        nb_model_train = nb_model_1$finalModel
        
        nb_pred_v = predict(nb_model_train, cv1[,1:57])
        nb_pred_v = nb_pred_v$posterior[,2]
        
        if ("predcv_temp" %in% ls()) {
            predcv_temp = c(predcv_temp, nb_pred_v)
        } else {
                predcv_temp = nb_pred_v
        }
        
        nb_pred_t = predict(nb_model_train, test[,1:57])
        nb_pred_t = nb_pred_t$posterior[,2]
        
        if ("predtest_temp" %in% ls()) {
            predtest_temp = predtest_temp + nb_pred_t
        } else {
            predtest_temp = nb_pred_t
        }
    }
    nb_pred_train = predcv_temp[order(all_idx)]
    nb_pred_test = predtest_temp / Nfold
    
    return(list(nb_pred_train=nb_pred_train, nb_pred_test=nb_pred_test))
    
} else {
        warning('Please install the packages caret and klaR.')
}

}
