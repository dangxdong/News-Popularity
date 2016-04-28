getCVsets = function(train, labelvec, n=5, seed=3030) {
    source('doNfoldsplit.r')
    all_idxlist = doNfoldsplit(labelvec, n, seed)
    labelvec = as.factor(as.character(labelvec))
    mm = length(all_idxlist)
    
    trainDFs = list()
    cvDFs = list()
    lbTrains = list()
    lbCVs = list()
    for (i in 1:mm) {
        trainDFs[[i]] = train[-all_idxlist[[i]],]
        cvDFs[[i]] = train[all_idxlist[[i]],]
        lbTrains[[i]] = labelvec[-all_idxlist[[i]]]
        lbCVs[[i]] = labelvec[all_idxlist[[i]]]
        names(trainDFs)[i] = paste0("trainset", i)
        names(cvDFs)[i] = paste0("cvset", i)
        names(lbTrains)[i] = paste0("lbs_train_", i)
        names(lbCVs)[i] = paste0("lbs_cv_", i)
    }
    all_idx = unlist(all_idxlist)
    list(trainDFs=trainDFs, cvDFs=cvDFs, lbTrains=lbTrains, lbCVs=lbCVs, all_idx=all_idx)
}


