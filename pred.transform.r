pred.transform = function(pred, bestcut) {
    predtemp = pred
    pred1 = pred
    pred1[predtemp < bestcut] = pred1[predtemp < bestcut] * 0.5 / bestcut
    pred1[predtemp > bestcut] = 1 - (1- pred1[predtemp > bestcut]) * 0.5 / (1 - bestcut)
    pred1[predtemp == bestcut] = 0.5
    rm(predtemp)
    return(pred1)
}