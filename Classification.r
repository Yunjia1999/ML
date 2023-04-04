---
title: "Classification"
author: "Yunjia Hu"
output:
  pdf_document: default
  html_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
#setwd()
set.seed(123)
```

# Simulating data

Assume a true decision boundary in a unit square with functional form: $y = 0.2 + x + 0.5 x^2 + 0.1 x^3 - 0.5 x^4$
```{r}
f<-function(x){
  return(0.2 + x - 0.5*x^2 + 0.1*x^3 - 0.5*x^4)
}
xv = seq(0,1,0.001)
yv = f(xv)
#plot(yv~xv,xlim=c(0,1), ylim=c(0,1),t = "l",col="red",xlab="x", ylab="y")
```

Simulate 500 data points uniformly distributed in the unit square
```{r}
dx = runif(500)
dy = runif(500)
#plot(yv~xv,xlim=c(0,1), ylim=c(0,1),t = "l",col="red",xlab="x", ylab="y")
#points(dy~dx,cex=0.5)
```


Classify the points above the boundary as "orange", and "blue" otherwise
```{r}
boundry = f(dx)
label = (dy>boundry)+0
# plot(yv~xv,xlim=c(0,1), ylim=c(0,1),t = "l",col="red",xlab="x", ylab="y")
# points(dy[label==1]~dx[label==1],cex=0.5,col="orange")
# points(dy[label==0]~dx[label==0],cex=0.5,col="navyblue")


```

Finally, add some white noise to y and obtain the final observed training data.
```{r}
x_value = dx
y_value = dy + rnorm(length(dy),sd=0.1)
training_data = cbind(y_value, x_value, label)
# plot(yv~xv,xlim=c(0,1), ylim=c(0,1),t = "l",col="red",xlab="x", ylab="y")
# points(y_value[label==1]~x_value[label==1],cex=0.5,col="orange")
# points(y_value[label==0]~x_value[label==0],cex=0.5,col="navyblue")

```

# Problem 1
```{r}
# Implement a bootstrap method to estimate the prediction error (EPE) of the linear classifier
# that we used in the class and compare it to the K-fold cross-validation results for K = 2, 5 and 10.
```

## Bootstrap
```{r}
epe1 = 0
for(i in 1:1000){
   for(b in 1:500){
      xb = sample(x_value, replace=TRUE)
   }
   yb = c(rep(0,500))
   labelb = c(rep(0,500))
   for(b in 1:500){
      yb[b] = y_value[which(xb[b]==x_value)]
      labelb[b] = label[which(xb[b]==x_value)]
   }
   #train
   fitb = glm(labelb~yb+xb,family=binomial(link="probit"))
   fitb
   interceptb = -fitb$coef[1]/fitb$coef[2]
   slopeb = -fitb$coef[3]/fitb$coef[2]
   #epe
   label1 = (y_value>slopeb*x_value+interceptb) + 0
   epe1sum = 0
   for(b in 1:500){
      epe1sum = epe1sum+(label[b]-label1[b])^2
   }
   epe1 = epe1+epe1sum/500
}
epe1 = epe1/1000
# epe1=0.111252

```

## K-fold
K-fold for K=2
```{r}
set.seed(1234)
library(caret)
folds = createFolds(y=x_value,k=2,list=T)
epe2 = 0
for(t in 1:2){
   xk = x_value[folds[[t]]] #xk
   yk = c(rep(0,length(folds[[t]]))) #yk
   labelk = c(rep(0,length(folds[[t]]))) #labelk
   for(k in 1:length(folds[[t]])){
      yk[k] = y_value[which(xk[k]==x_value)]
      labelk[k] = label[which(xk[k]==x_value)]
   }
   folds_train = c() #xk_train
   for(i in 1:2){
      if(i!=t){
         folds_train = c(folds_train,folds[[i]])
      }
   }
   xk_train = x_value[folds_train]
   yk_train = c(rep(0,500-length(folds[[t]]))) #yk_train
   labelk_train = c(rep(0,500-length(folds[[t]]))) #labelk_train
   for(k in 1:(500-length(folds[[t]]))){
      yk_train[k] = y_value[which(xk_train[k]==x_value)]
      labelk_train[k] = label[which(xk_train[k]==x_value)]
   }
   #train
   fitk = glm(labelk_train~yk_train+xk_train,family=binomial(link="probit"))
   fitk
   interceptk = -fitk$coef[1]/fitk$coef[2]
   slopek = -fitk$coef[3]/fitk$coef[2]
   #epe
   label2 = (yk>slopek*xk+interceptk) + 0
   epe2sum = 0
   for(k in 1:length(folds[[t]])){
      epe2sum = epe2sum+(labelk[k]-label2[k])^2
   }
   epe2 = epe2+epe2sum/(length(folds[[t]]))
}
epe2 = epe2/2
# If K=2, epe2=0.110.

```

K-fold for K=5
```{r}
set.seed(1234)
library(caret)
folds = createFolds(y=x_value,k=5,list=T)
epe2 = 0
for(t in 1:5){
   xk = x_value[folds[[t]]] #xk
   yk = c(rep(0,length(folds[[t]]))) #yk
   labelk = c(rep(0,length(folds[[t]]))) #labelk
   for(k in 1:length(folds[[t]])){
      yk[k] = y_value[which(xk[k]==x_value)]
      labelk[k] = label[which(xk[k]==x_value)]
   }
   folds_train = c() #xk_train
   for(i in 1:5){
      if(i!=t){
         folds_train = c(folds_train,folds[[i]])
      }
   }
   xk_train = x_value[folds_train]
   yk_train = c(rep(0,500-length(folds[[t]]))) #yk_train
   labelk_train = c(rep(0,500-length(folds[[t]]))) #labelk_train
   for(k in 1:(500-length(folds[[t]]))){
      yk_train[k] = y_value[which(xk_train[k]==x_value)]
      labelk_train[k] = label[which(xk_train[k]==x_value)]
   }
   #train
   fitk = glm(labelk_train~yk_train+xk_train,family=binomial(link="probit"))
   fitk
   interceptk = -fitk$coef[1]/fitk$coef[2]
   slopek = -fitk$coef[3]/fitk$coef[2]
   #epe
   label2 = (yk>slopek*xk+interceptk) + 0
   epe2sum = 0
   for(k in 1:length(folds[[t]])){
      epe2sum = epe2sum+(labelk[k]-label2[k])^2
   }
   epe2 = epe2+epe2sum/(length(folds[[t]]))
}
epe2 = epe2/5
# If K=5, epe2=0.112.

```


K-fold for K=10
```{r}
set.seed(1234)
library(caret)
folds = createFolds(y=x_value,k=10,list=T)
epe2 = 0
for(t in 1:10){
   xk = x_value[folds[[t]]] #xk
   yk = c(rep(0,length(folds[[t]]))) #yk
   labelk = c(rep(0,length(folds[[t]]))) #labelk
   for(k in 1:length(folds[[t]])){
      yk[k] = y_value[which(xk[k]==x_value)]
      labelk[k] = label[which(xk[k]==x_value)]
   }
   folds_train = c() #xk_train
   for(i in 1:10){
      if(i!=t){
         folds_train = c(folds_train,folds[[i]])
      }
   }
   xk_train = x_value[folds_train]
   yk_train = c(rep(0,500-length(folds[[t]]))) #yk_train
   labelk_train = c(rep(0,500-length(folds[[t]]))) #labelk_train
   for(k in 1:(500-length(folds[[t]]))){
      yk_train[k] = y_value[which(xk_train[k]==x_value)]
      labelk_train[k] = label[which(xk_train[k]==x_value)]
   }
   #train
   fitk = glm(labelk_train~yk_train+xk_train,family=binomial(link="probit"))
   fitk
   interceptk = -fitk$coef[1]/fitk$coef[2]
   slopek = -fitk$coef[3]/fitk$coef[2]
   #epe
   label2 = (yk>slopek*xk+interceptk) + 0
   epe2sum = 0
   for(k in 1:length(folds[[t]])){
      epe2sum = epe2sum+(labelk[k]-label2[k])^2
   }
   epe2 = epe2+epe2sum/(length(folds[[t]]))
}
epe2 = epe2/10
# If K=10, epe2=0.10990.

```
Comparing bootstrap method and K-fold cross-validation results, EPE from bootstrap is a little bit smaller than K-fold when K=5, and a little larger than K-fold when K=2 and 10. But all the methods perform well, and K-fold CV result for K=10 seems the best.



# Problem 2
Since K-fold CV for K=10 shows the best result in question 1, apply K-fold=10 in this question.

## (a)
```{r}
# Implement a cross-validation scheme select the optimal tuning parameter k,
# i.e., find the “optimal” number of nearest neighbors.
set.seed(1234)
library(caret)
j=3
l=1
q=3
minepe=1
epe3=c(rep(0,15))
while( j<33 ){
   folds = createFolds(y=x_value,k=10,list=T)
   epe3[l] = 0
   for(t in 1:10){
      xk = x_value[folds[[t]]] #xk
      yk = c(rep(0,length(folds[[t]]))) #yk
      labelk = c(rep(0,length(folds[[t]]))) #labelk
      for(k in 1:length(folds[[t]])){
         yk[k] = y_value[which(xk[k]==x_value)]
         labelk[k] = label[which(xk[k]==x_value)]
      }
      folds_train = c() #xk_train
      for(i in 1:10){
         if(i!=t){
            folds_train = c(folds_train,folds[[i]])
         }
      }
      xk_train = x_value[folds_train]
      yk_train = c(rep(0,500-length(folds[[t]]))) #yk_train
      labelk_train = c(rep(0,500-length(folds[[t]]))) #labelk_train
      for(k in 1:(500-length(folds[[t]]))){
         yk_train[k] = y_value[which(xk_train[k]==x_value)]
         labelk_train[k] = label[which(xk_train[k]==x_value)]
      }
      #train
      Grid=cbind(yk,xk)
      training_data1 = cbind(yk_train,xk_train,labelk_train)
      vote<-function(target, K, td = training_data1  ){
      dist = apply(td,1, function(x) (x[1]-target[1])^2+(x[2]-target[2])^2)
      # find the first k-ranked points
      index = which(rank(dist)<=K)
      rst = 1
      if(sum(td[index,3])<K/2){
      rst = 0
      }
      return(rst)
      }
      est_rst = apply(Grid,1,function(x) vote(x,K=j))
      #epe
      label3 = est_rst
      epe3sum = 0
      for(k in 1:length(folds[[t]])){
         epe3sum = epe3sum+(labelk[k]-label3[k])^2
      }
      epe3[l] = epe3[l]+epe3sum/(length(folds[[t]]))
   }
   epe3[l] = epe3[l]/10
   if(epe3[l]<minepe){
      q=j
      minepe=epe3[l]
   }
   j = j+2
   if(j<33){
      l = l+1
   }
}
epe3
q
# q=23
```
The optimal number of knn is k=23.


Draw a picture when k=15,training fold=7.
```{r}
set.seed(1234)
library(caret)
   folds = createFolds(y=x_value,k=10,list=T)
      xk = x_value[folds[[7]]] #xk
      yk = c(rep(0,length(folds[[7]]))) #yk
      labelk = c(rep(0,length(folds[[7]]))) #labelk
      for(k in 1:length(folds[[7]])){
         yk[k] = y_value[which(xk[k]==x_value)]
         labelk[k] = label[which(xk[k]==x_value)]
      }
      folds_train = c() #xk_train
      for(i in 1:10){
         if(i!=7){
            folds_train = c(folds_train,folds[[i]])
         }
      }
      xk_train = x_value[folds_train]
      yk_train = c(rep(0,500-length(folds[[7]]))) #yk_train
      labelk_train = c(rep(0,500-length(folds[[7]]))) #labelk_train
      for(k in 1:(500-length(folds[[7]]))){
         yk_train[k] = y_value[which(xk_train[k]==x_value)]
         labelk_train[k] = label[which(xk_train[k]==x_value)]
      }
      #train
      Grid=cbind(yk,xk)
      training_data1 = cbind(yk_train,xk_train,labelk_train)
      vote<-function(target, K, td = training_data1  ){
      dist = apply(td,1, function(x) (x[1]-target[1])^2+(x[2]-target[2])^2)
      # find the first k-ranked points
      index = which(rank(dist)<=K)
      rst = 1
      if(sum(td[index,3])<K/2){
      rst = 0
      }
      return(rst)
      }
      est_rst = apply(Grid,1,function(x) vote(x,K=15))
      plot(Grid[,1]~Grid[,2], pch=16,cex=0.45,xlab="x",ylab="y",col="navyblue")
      index = which(est_rst==1)
      points(Grid[index,1]~Grid[index,2], col="orange",pch=16,cex=0.45)
      lines(yv~xv,col="red")
      label3 = est_rst
      epe3sum = 0
      for(k in 1:length(folds[[7]])){
         epe3sum = epe3sum+(labelk[k]-label3[k])^2
      }
      epe3=epe3sum/(length(folds[[7]]))

```

## (b) 
```{r}
#Estimate the EPE for the optimal k=23 using the training data.
epe4 = 0
#train
Grid=cbind(y_value,x_value)
vote<-function(target, K, td = training_data  ){
   dist = apply(td,1, function(x) (x[1]-target[1])^2+(x[2]-target[2])^2)
   # find the first k-ranked points
   index = which(rank(dist)<=K)
   rst = 1
   if(sum(td[index,3])<K/2){
      rst = 0
   }
   return(rst)
}
est_rst = apply(Grid,1,function(x) vote(x,K=23))
#epe
label4 = est_rst
epe4sum = 0
for(k in 1:500){
   epe4sum = epe4sum+(label[k]-label4[k])^2
}
epe4 = epe4sum/500
epe4
#epe4=0.088
```

Estimated EPE for k=23 using the training data is 0.088.


## (c) 
```{r}
# Simulate new data according to the true generative model and re-estimate the EPE for the estimated optimal k.
# Simulate 500 new data points.
dx0 = runif(500)
dy0 = runif(500)
boundry0 = f(dx0)
label0 = (dy0>boundry0)+0
x_value0 = dx0
y_value0 = dy0 + rnorm(length(dy0),sd=0.1)
training_data0 = cbind(y_value0, x_value0, label0)
# Re-estimate EPE.
epe5 = 0
#train
Grid=cbind(y_value0,x_value0)
vote<-function(target, K, td = training_data0  ){
   dist = apply(td,1, function(x) (x[1]-target[1])^2+(x[2]-target[2])^2)
   # find the first k-ranked points
   index = which(rank(dist)<=K)
   rst = 1
   if(sum(td[index,3])<K/2){
      rst = 0
   }
   return(rst)
}
est_rst = apply(Grid,1,function(x) vote(x,K=23))
#epe
label5 = est_rst
epe5sum = 0
for(k in 1:500){
   epe5sum = epe5sum+(label0[k]-label5[k])^2
}
epe5 = epe5sum/500
epe5
#epe5 is about 0.066.

```

The re-estimated EPE for 500 new data is 0.06.


# Problem 3
## (a) 
```{r}
# Use caret package to determine the optimal k value for the simple classification example.
trdf = data.frame(training_data)
trdf$label = as.factor(trdf$label)
class(trdf$label)
levels(trdf$label)=make.names(levels(trdf$label))
tr = trainControl(method = "cv", number = 10, classProbs = T)
model1 = train(label ~ y_value+x_value, data=trdf,
               method="knn", trControl=tr, tuneLength=15)
model1
# The final value used for the model was k = 23.
```
The optimal k value for the simple classification example is 23.


## (b)
```{r}
# Compare the knn classifier to the naive Bayes classifier implemented in the caret package.
# Given a brief summary on your conclusions.
trdf$label = as.factor(trdf$label)
class(trdf$label)
levels(trdf$label)=make.names(levels(trdf$label))
model2 = train(label ~ y_value+x_value, data=trdf,
               method='nb', trControl=tr)
model2

```
The knn classifier and the naive Bayes classifier both perform well in simple classification, with accuracy higher than 90%.














