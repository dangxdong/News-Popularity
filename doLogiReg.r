doLogiReg = function(cvlists, test=test) {
    
    Nfold = length(cvlists$lbCVs)
    all_idx = cvlists$all_idx
    
    if ("predcv_temp" %in% ls()) { rm(predcv_temp) }
    if ("predtest_temp" %in% ls()) { rm(predtest_temp) }
    for (i in 1:Nfold) {
        train1 = cvlists$trainDFs[[i]]
        cv1 = cvlists$cvDFs[[i]]
        train1_lbs = cvlists$lbTrains[[i]]
        cv1_lbs = cvlists$lbCVs[[i]]
        
        glm1 = glm(lbscut ~ ., data=train1, family="binomial")
        glm_pred_1 = predict(glm1, cv1, type="response")
        
        if ("predcv_temp" %in% ls()) {
            predcv_temp = c(predcv_temp, glm_pred_1)
        } else {
            predcv_temp = glm_pred_1
        }
        
        glm_pred_test1 = predict(glm1, test, type="response")
        
        if ("predtest_temp" %in% ls()) {
            predtest_temp = predtest_temp + glm_pred_test1
        } else {
            predtest_temp = glm_pred_test1
        }
    }
    
    glm_pred_train = predcv_temp[order(all_idx)]
    glm_pred_test = predtest_temp / Nfold
    
    return(list(glm_pred_train=glm_pred_train, glm_pred_test=glm_pred_test))
    
}