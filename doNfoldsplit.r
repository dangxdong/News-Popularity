doNfoldsplit = function(train_lbs, n=5, seed=3030) {

if (require('caTools', quietly = TRUE)) {
    idxlist=list()
    idxtrain=1:length(train_lbs)
    if ("idxtemp" %in% ls()) { rm("idxtemp") }
    for (i in 1:(n-1)) {
      if ("idxtemp" %in% ls()) {
      set.seed(seed)
      split5 = sample.split(cvtemp_lbs, SplitRatio = 1/(n-i+1) )
      
      cvtemp_lbs = subset(cvtemp_lbs, split5 == F)
      idxlist[[i]] = subset(idxtemp, split5 == T)
      idxtemp = subset(idxtemp, split5 == F)
      } else {
      split5 = sample.split(train_lbs, SplitRatio = 1/(n-i+1) )

      cvtemp_lbs = subset(train_lbs, split5 == F)
      idxlist[[i]] = subset(idxtrain, split5 == T)
      idxtemp = subset(idxtrain, split5 == F)
      }
    }
    idxlist[[n]] = idxtemp
    return(idxlist)
} else {
    warning('Please install the package caTools. ')
}

}