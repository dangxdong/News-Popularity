calclogloss = function (pred, y) {
    y = y[!is.na(pred)]
    pred = pred[!is.na(pred)]
    m = length(pred)
    y = as.numeric(as.character(y))
    pred = pmax(pmin(pred, 1-10^(-15)), 10^(-15))
    lossf = -y*log(pred) - (1-y)*log(1-pred)
    cost = 1 / m * sum(lossf)
    sum(cost)
}