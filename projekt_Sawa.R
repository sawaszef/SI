#funkcje
#jakość klasyfikatora
acc = function(x)
{sum(diag(x))/sum(x)}

myidx = function(arg){
  return (which(arg == max(arg)))
}

nor = function(x)
{
  (x-min(x))/(max(x)-min(x))
}


#biblioteki
library(class)
library(naivebayes)
library(neuralnet)
library(dplyr)

#wczytanie danych
data = read.csv("balance-scale.data", header=F)
datatemp = read.csv("balance-scale.data", header=F)
data["V1"][data["V1"] == "L"] <- "0"
data["V1"][data["V1"] == "B"] <- "1"
data["V1"][data["V1"] == "R"] <- "2"

dataNonNorm = data
dataNonNorm$V1 = as.factor(data$V1)

dataattributes = nor(select(data, V2:V5))
data$V1 = as.factor(data$V1)

data = select(data, V1)
data = data.frame(data, dataattributes)



#podział zbioru na testowy i treningowy, w różnych proporcjach
#normalizowane
idx80 = sample(1:nrow(data), 0.8*nrow(data))
idx60 = sample(1:nrow(data), 0.6*nrow(data))
idx40 = sample(1:nrow(data), 0.4*nrow(data))

train80 = data[idx80,]
test80 = data[-idx80,]
train60 = data[idx60,]
test60 = data[-idx60,]
train40 = data[idx40,]
test40 = data[-idx40,]

#nienormalizowane
idx80n = sample(1:nrow(dataNonNorm), 0.8*nrow(dataNonNorm))
idx60n = sample(1:nrow(dataNonNorm), 0.6*nrow(dataNonNorm))
idx40n = sample(1:nrow(dataNonNorm), 0.4*nrow(dataNonNorm))

train80n = dataNonNorm[idx80n,]
test80n = dataNonNorm[-idx80n,]
train60n = dataNonNorm[idx60n,]
test60n = dataNonNorm[-idx60n,]
train40n = dataNonNorm[idx40n,]
test40n = dataNonNorm[-idx40n,]

#KNN
cl = data$V1

#80
clTrain80 = cl[idx80n]
clTest80 = cl[-idx80n]
modelKNN80 = knn(train80n, test80n, cl=clTrain80, k=5)
tabKNN80 = table(modelKNN80, clTest80)

tabKNN80
acc(tabKNN80)

#60
clTrain60 = cl[idx60n]
clTest60 = cl[-idx60n]
modelKNN60 = knn(train60n, test60n, cl=clTrain60, k=5)
tabKNN60 = table(modelKNN60, clTest60)

tabKNN60
acc(tabKNN60)

#40
clTrain40 = cl[idx40n]
clTest40 = cl[-idx40n]
modelKNN40 = knn(train40n, test40n, cl=clTrain40, k=5)
tabKNN40 = table(modelKNN40, clTest40)

tabKNN40
acc(tabKNN40)

#NaiveBayes


#80
modelNB80 = naive_bayes(V1 ~ ., data=train80n)
pNB80 = predict(modelNB80, test80n)
tabNB80 = table(pNB80, test80n$V1)

tabNB80
acc(tabNB80)

#60
modelNB60 = naive_bayes(V1 ~ ., data=train60n)
pNB60 = predict(modelNB60, test60n)
tabNB60 = table(pNB60, test60n$V1)

tabNB60
acc(tabNB60)

#40
modelNB40 = naive_bayes(V1 ~ ., data=train40n)
pNB40 = predict(modelNB40, test40n)
tabNB40 = table(pNB40, test40n$V1)

tabNB40
acc(tabNB40)

#NN

#80
modelNN80 = neuralnet(data$V1 ~ .,
                  data=train80,
                  hidden=c(5,3),
                  threshold = 0.02,
                  stepmax = 1e+07)
plot(modelNN80)
predictNN80 = neuralnet::compute(modelNN80, test80[-1])$net.result
idx_outNN80 = apply(predictNN80, c(1), myidx)
predictionNN80 = c("0", "1", "2")[idx_outNN80]
tabNN80 = table(predictionNN80, test80$V1)
tabNN80
acc(tabNN80)

#60
modelNN60 = neuralnet(data$V1 ~ .,
                      data=train60,
                      hidden=c(5,3),
                      threshold = 0.02,
                      stepmax = 1e+07)
plot(modelNN60)
predictNN60 = neuralnet::compute(modelNN60, test60[-1])$net.result
idx_outNN60 = apply(predictNN60, c(1), myidx)
predictionNN60 = c("0", "1", "2")[idx_outNN60]
tabNN60 = table(predictionNN60, test60$V1)
tabNN60
acc(tabNN60)

#40
modelNN40 = neuralnet(data$V1 ~ .,
                      data=train40,
                      hidden=c(5,3),
                      threshold = 0.02,
                      stepmax = 1e+07)
plot(modelNN40)
predictNN40 = neuralnet::compute(modelNN40, test40[-1])$net.result
idx_outNN40 = apply(predictNN40, c(1), myidx)
predictionNN40 = c("0", "1", "2")[idx_outNN40]
tabNN40 = table(predictionNN40, test40$V1)
tabNN40
acc(tabNN40)