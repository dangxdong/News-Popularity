getperformance = function(pred1, lable, getplot=TRUE, simple=FALSE) {
  
  if (require('ROCR', quietly = TRUE)) {
    ROCRpred1 = prediction(pred1, lable)
    ROCRperf1 = performance(ROCRpred1, "tpr", "fpr")
    
    auc1 = as.numeric(performance(ROCRpred1, "auc")@y.values)
    
    cutoffs1 <- data.frame(cut=ROCRperf1@alpha.values[[1]], fpr=ROCRperf1@x.values[[1]], 
                           tpr=ROCRperf1@y.values[[1]])
    
    # Use a new score, tpr*tnr/(tpr+tnr), to decide which threshold is best 
    cutoffs1$NewScore = cutoffs1$tpr * (1-cutoffs1$fpr) / (cutoffs1$tpr + (1-cutoffs1$fpr))
    
    # then just get the threshold with the max of new score
    thresh1=cutoffs1$cut[which.max(cutoffs1$NewScore)]
    
    m = length(pred1)
    
    tb05 = table(lable, prediction=(pred1 >= 0.5))
    
    tbct = table(lable, prediction=(pred1 >= thresh1))
    
    accuracy05 = (tb05[1,1] + tb05[2,2]) / m
    accuracy.best = (tbct[1,1] + tbct[2,2]) / m
    
    recall05 = tb05[2,2] / (tb05[2,1] + tb05[2,2])
    precision05 = tb05[2,2] / (tb05[1,2] + tb05[2,2])
    specificity05 = tb05[1,1] / (tb05[1,1] + tb05[1,2])
    
    recall.best = tbct[2,2] / (tbct[2,1] + tbct[2,2])
    precision.best = tbct[2,2] / (tbct[1,2] + tbct[2,2])
    specificity.best = tbct[1,1] / (tbct[1,1] + tbct[1,2])
    
    if (getplot) { plot(ROCRperf1, colorize=TRUE) }
    
    if (simple) {
      return(list(             AUC = auc1, 
                               bestcutoff = thresh1,
                               accuracy.best = accuracy.best,
                               ConfusionMatrixTable_best = tbct) )
    } else {
      return(list(             AUC = auc1, 
                               bestcutoff = thresh1, 
                               accuracy05 = accuracy05, 
                               accuracy.best = accuracy.best,
                               recall05 = recall05,
                               precision05 = precision05,
                               specificity05 = specificity05,
                               recall.best = recall.best,
                               precision.best = precision.best,
                               specificity.best = specificity.best,
                               ConfusionMatrixTable_0.5 = tb05,
                               ConfusionMatrixTable_best = tbct) )
    }
  } else {
    warning('Please install the package ROCR. ')
  }
  
}
