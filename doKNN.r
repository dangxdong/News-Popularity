doKNN = function(cvlists, test=test) {
    library(class)
    library(e1071)
    Nfold = length(cvlists$lbCVs)
    all_idx = cvlists$all_idx
    
    Xtest = test[,1:57]
    if ("predcv_temp" %in% ls()) { rm(predcv_temp) }
    if ("predtest_temp" %in% ls()) { rm(predtest_temp) }
    for (i in 1:Nfold) {
        train1 = cvlists$trainDFs[[i]]
        cv1 = cvlists$cvDFs[[i]]
        
        train1_lbs = cvlists$lbTrains[[i]]
        cv1_lbs = cvlists$lbCVs[[i]]
        
        Xtrain = train1[,1:57]
        Xcv = cv1[,1:57]
        
        # the range of k has been roughly explored before, so just use k=52:
        # don't do
        # knn_fit_1 = tune.knn(x=Xtrain, y=train1_lbs, k=c(45, 47, 50, 52, 55), 
        #                     validation.x=Xcv, validation.y=cv1_lbs,
        #                     tunecontrol=tune.control(sampling = "fix", fix=1))
        # bestk = knn_fit_1$best.parameters[[1]]
        
        knn_pred_1 = knn(train=Xtrain, test=Xcv, 
                          cl=train1_lbs, prob=T, k=52)
        knn_pred_v = attr(knn_pred_1,"prob")
        knn_pred_11 = as.integer(as.character(knn_pred_1))
        knn_pred_v = (1-knn_pred_11) * (1-knn_pred_v) + knn_pred_11 * knn_pred_v
        
        if ("predcv_temp" %in% ls()) {
            predcv_temp = c(predcv_temp, knn_pred_v)
        } else {
                predcv_temp = knn_pred_v
        }
        
        knn_pred_2 = knn(train=Xtrain, test=Xtest, 
                          cl=train1_lbs, prob=T, k=52)
        knn_pred_t = attr(knn_pred_2,"prob")
        knn_pred_22 = as.integer(as.character(knn_pred_2))
        knn_pred_t = (1-knn_pred_22) * (1-knn_pred_t) + knn_pred_22 * knn_pred_t
        
        if ("predtest_temp" %in% ls()) {
            predtest_temp = predtest_temp + knn_pred_t
        } else {
            predtest_temp = knn_pred_t
        }
    }
    knn_pred_train = predcv_temp[order(all_idx)]
    knn_pred_test = predtest_temp / Nfold
    
    return(list(knn_pred_train=knn_pred_train, knn_pred_test=knn_pred_test))
    
}