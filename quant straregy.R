######################################################
# Machine Trading Analysis with R                    #
# (c) Diego Fernandez Garcia 2015-2017               #
# www.exfinsis.com                                   #
######################################################

# 1. Machine Trading Analysis Data

# 1.1. Load R packages 
library("caret")
library("corrplot")
library("forecast")
library("kernlab")
library("neuralnet")
library("PerformanceAnalytics")
library("quantmod")
library("tseries")
library("xgboost")

# 1.2. Data Downloading or Reading

# 1.2. Data Downloading or Reading

# 1.2.1. Yahoo Finance 

getSymbols("AAPL") 

data <- AAPL$AAPL.Adjusted 



# getSymbols("APPL")

# data <- MSFT$MSFT.Adjusted

# 2. Feature Creation

# 2.1. Target Feature
returndata <- dailyReturn(data,type="log")




# 2.2. Predictor Features
returndata1 <- Lag(returndata,k=1)
returndata2 <- Lag(returndata,k=2)
returndata3 <- Lag(returndata,k=3)
returndata4 <- Lag(returndata,k=4)
returndata5 <- Lag(returndata,k=5)
returndata6 <- Lag(returndata,k=6)
returndata7 <- Lag(returndata,k=7)
returndata8 <- Lag(returndata,k=8)
returndata9 <- Lag(returndata,k=9)

# 2.3. All Features
returnall <- cbind(returndata,returndata1,returndata2,returndata3,returndata4,returndata5,returndata6,returndata7,returndata8,returndata9)
colnames(returnall) <- c("returndata","returndata1","returndata2","returndata3","returndata4","returndata5","returndata6","returndata7","returndata8","returndata9")
returnall <- returnall[complete.cases(returnall),]

# 3. Range Delimiting

length(returndata)

returnallpart <- length(returndata)/10-1

returntrainingend <-  returnallpart*7

returntestingend  <-  returnallpart*9

returntradingend <- returnallpart*10


resturninttestingstart <- returnallpart*2

returninttradingstart <- returnallpart*3


# 3.1. Training Range



returntrainingdata <- returnall[(1:returntrainingend),]


# 3.2. Testing Range
returntestingdata <- returnall[((returntrainingend+1):returntestingend),]

# 3.3. Intermediate Testing Range 
# Same length as Training Range
returnintertestingdata <- returnall[((resturninttestingstart+1):returntestingend),]
length(returntrainingdata)
length(returnintertestingdata)

# 3.4. Trading Range
returntradingdata <- returnall[((returntestingend):(returntradingend)),]

# 3.5. Intermediate Trading Range
# Same length as Training Range
returnintertradingdata<- returnall[((returninttradingstart+1):(returntradingend)),]


length(returntrainingdata)
length(returnintertradingdata)

# 4. Predictor Features Selection

# 4.2. Predictor Features Linear Regression
lmta <- lm(returndata~returndata1+returndata2+returndata3+returndata4+returndata5+returndata6+returndata7+returndata8+returndata9,data=returntrainingdata)
summary(lmta)
lmtb <- lm(returndata~returndata1+returndata2+returndata6,data=returntrainingdata)
summary(lmtb)
# Clear Plots area before running code
par(mfrow=c(1,2))
plot(coredata(returntrainingdata$returndata1),coredata(returntrainingdata$returndata),xlab="returndata1t",ylab="returntrainingdata")
abline(lm(returntrainingdata$returndata~returntrainingdata$returndata1),col="red")
plot(coredata(returntrainingdata$returndata2),coredata(returntrainingdata$returndata),xlab="returndata2t",ylab="returntrainingdata")
abline(lm(returntrainingdata$returndata~returntrainingdata$returndata2),col="red")
plot(coredata(returntrainingdata$returndata5),coredata(returntrainingdata$returndata),xlab="returndata5t",ylab="returntrainingdata")
abline(lm(returntrainingdata$returndata~returntrainingdata$returndata5),col="red")
par(mfrow=c(1,1))

# 4.2. Predictor Features Correlation
creturntrainingdata <- round(cor(returntrainingdata[,2:10]),2)
creturntrainingdata
corrplot(creturntrainingdata,type="lower")

# 4.3. Predictor Features Selection Filter Methods

# 4.3.1. Univariate Filters
sbfctrlt <- sbfControl(functions=lmSBF)
sbft <- sbf(returndata~returndata1+returndata2+returndata3+returndata4+returndata5+returndata6+returndata7+returndata8+returndata9,data=returntrainingdata,sbfControl=sbfctrlt)
sbft

# 4.4. Predictor Features Selection Wrapper Methods

# 4.4.1. Recursive Feature Elimination
rfectrlt <- rfeControl(functions=lmFuncs)
rfet <- rfe(returndata~returndata1+returndata2+returndata3+returndata4+returndata5+returndata6+returndata7+returndata8+returndata9,data=returntrainingdata,rfeControl=rfectrlt)
rfet

# 4.5. Predictor Features Selection Embedded Methods
lassot <- train(returndata~returndata1+returndata2+returndata3+returndata4+returndata5+returndata6+returndata7+returndata8+returndata9,data=returntrainingdata,method="lasso")
predictors(lassot)

# 4.6. Predictor Features Extraction

# 4.6.1. Principal Component Analysis
pcat <- princomp(returntrainingdata[,2:10])
summary(pcat)
plot(pcat)

# 5. Algorithm Training and Testing

# 5.1. Ensemble Methods

# eXtreme Gradient Boosting Machine Regression

# 5.1.1. eXtreme Gradient Boosting Regression training
xgbmta <- train(returndata~returndata1+returndata2+returndata6,data=returntrainingdata,method="xgbTree")
xgbmtb <- train(returndata~returndata1+returndata2+returndata3+returndata4+returndata5+returndata6+returndata7+returndata8+returndata9,data=returntrainingdata,method="xgbTree",preProcess="pca")

# eXtreme Gradient Boosting Regression optimal training parameters
xgbmta$bestTune
plot(xgbmta)
xgbmtb$bestTune
plot(xgbmtb)

# eXtreme Gradient Boosting Regression training results
xgbmta$results
xgbmtb$results

# 5.1.2. eXtreme Gradient Boosting Regression testing
# Intermediate testing step as newdata needs to be same length as training range 
xgbmpa <- predict.train(xgbmta,newdata=returnintertestingdata)
xgbmpb <- predict.train(xgbmtb,newdata=returnintertestingdata)

# Limited to testing range
xgbmdfa <- cbind(index(returnintertestingdata),as.data.frame(xgbmpa))
xgbmla <- xts(xgbmdfa[,2],order.by=as.Date(xgbmdfa[,1]))
xgbmfa <- window(xgbmla,start=index(returntestingdata[1,]))
xgbmdfb <- cbind(index(returnintertestingdata),as.data.frame(xgbmpb))
xgbmlb <- xts(xgbmdfb[,2],order.by=as.Date(xgbmdfb[,1]))
xgbmfb <- window(xgbmlb,start=index(returntestingdata[1,]))

# 5.1.3. eXtreme Gradient Boosting Regression testing chart
plot(returntestingdata[,1],type="l",main="eXtreme Gradient Boosting Regression A Testing Chart")
lines(xgbmfa,col="blue")
plot(returntestingdata[,1],type="l",main="eXtreme Gradient Boosting Regression B Testing Chart")
lines(xgbmfb,col="green")

# 5.1.4. eXtreme Gradient Boosting Regression forecasting accuracy
# Convert xts to ts for accuracy function
xgbmftsa <- ts(coredata(xgbmfa),frequency=252,start=index(returntestingdata[1,]))
xgbmftsb <- ts(coredata(xgbmfb),frequency=252,start=index(returntestingdata[1,]))
returntestingdatats <- ts(coredata(returntestingdata[,1]),frequency=252,start=index(returntestingdata[1,]))
returndata1fts <- ts(coredata(returntestingdata[,2]),frequency=252,start=index(returntestingdata[1,]))
accuracy(xgbmftsa,returntestingdatats)
rndmape <- accuracy(returntestingdatats,returndata1fts)[5]
xgbmmasea <- accuracy(xgbmftsa,returntestingdatats)[5]/rndmape
xgbmmasea
accuracy(xgbmftsb,returntestingdatats)
xgbmmaseb <- accuracy(xgbmftsb,returntestingdatats)[5]/rndmape
xgbmmaseb

# 5.2. Algorithm Training Optimal Parameters Selection Control

# 5.2.1. Time Series Cross-Validation
tsctrlt <- trainControl(method="timeslice",initialWindow=16,horizon=8,fixedWindow=TRUE)

# 5.3. Maximum Margin Methods

# Support Vector Machine Regression with Radial Basis Function Kernel

# 5.3.1. RBF Support Vector Machine Regression training
rsvmta <- train(returndata~returndata1+returndata2+returndata6,data=returntrainingdata,method="svmRadial",trControl=tsctrlt)
rsvmtb <- train(returndata~returndata1+returndata2+returndata3+returndata4+returndata5+returndata6+returndata7+returndata8+returndata9,data=returntrainingdata,method="svmRadial",preProcess="pca",trControl=tsctrlt)

# RBF Support Vector Machine Regression optimal training parameters
rsvmta$bestTune
plot(rsvmta)
rsvmtb$bestTune
plot(rsvmtb)

# RBF Support Vector Machine Regression training results
rsvmta$results
rsvmtb$results

# 5.3.2. RBF Support Vector Machine Regression testing
# Intermediate testing step as newdata needs to be same length as training range 
rsvmpa <- predict.train(rsvmta,newdata=returnintertestingdata)
rsvmpb <- predict.train(rsvmtb,newdata=returnintertestingdata)

# Limited to testing range
rsvmdfa <- cbind(index(returnintertestingdata),as.data.frame(rsvmpa))
rsvmla <- xts(rsvmdfa[,2],order.by=as.Date(rsvmdfa[,1]))
rsvmfa <- window(rsvmla,start=index(returntestingdata[1,]))
rsvmdfb <- cbind(index(returnintertestingdata),as.data.frame(rsvmpb))
rsvmlb <- xts(rsvmdfb[,2],order.by=as.Date(rsvmdfb[,1]))
rsvmfb <- window(rsvmlb,start=index(returntestingdata[1,]))

# 5.3.3. RBF Support Vector Machine Regression testing chart
plot(returntestingdata[,1],type="l",main="RBF Support Vector Machine Regression A Testing Chart")
lines(rsvmfa,col="blue")
plot(returntestingdata[,1],type="l",main="RBF Support Vector Machine Regression B Testing Chart")
lines(rsvmfb,col="green")

# 5.3.4. RBF Support Vector Machine Regression forecasting accuracy
# Convert xts to ts for accuracy function
rsvmftsa <- ts(coredata(rsvmfa),frequency=252,start=index(returntestingdata[1,]))
accuracy(rsvmftsa,returntestingdatats)
rsvmmasea <- accuracy(rsvmftsa,returntestingdatats)[5]/rndmape
rsvmmasea
rsvmftsb <- ts(coredata(rsvmfb),frequency=252,start=index(returntestingdata[1,]))
accuracy(rsvmftsb,returntestingdatats)
rsvmmaseb <- accuracy(rsvmftsb,returntestingdatats)[5]/rndmape
rsvmmaseb

# 5.4. Multi-Layer Perceptron Methods

# Artificial Neural Network Regression 

# 5.4.1. Artificial Neural Network Regression training
annta <- train(returndata~returndata1+returndata2+returndata6,data=returntrainingdata,method="neuralnet",trControl=tsctrlt)
anntb <- train(returndata~returndata1+returndata2+returndata3+returndata4+returndata5+returndata6+returndata7+returndata8+returndata9,data=returntrainingdata,method="neuralnet",preProcess="pca",trControl=tsctrlt)

# Artificial Neural Network Regression optimal training parameters
annta$bestTune
plot(annta)
anntb$bestTune
plot(anntb)

# Artificial Neural Network Regression training results
annta$results
anntb$results

# 5.4.2. Artificial Neural Network Regression testing
# Intermediate testing step as newdata needs to be same length as training range 
annpa <- predict.train(annta,newdata=returnintertestingdata)
annpb <- predict.train(anntb,newdata=returnintertestingdata)

# Limited to testing range
anndfa <- cbind(index(returnintertestingdata),as.data.frame(annpa))
annla <- xts(anndfa[,2],order.by=as.Date(anndfa[,1]))
annfa <- window(annla,start=index(returntestingdata[1,]))
anndfb <- cbind(index(returnintertestingdata),as.data.frame(annpb))
annlb <- xts(anndfb[,2],order.by=as.Date(anndfb[,1]))
annfb <- window(annlb,start=index(returntestingdata[1,]))

# 5.4.3. Artificial Neural Network Regression testing chart
plot(returntestingdata[,1],type="l",main="Artificial Neural Network Regression A Testing Chart")
lines(annfa,col="blue")
plot(returntestingdata[,1],type="l",main="Artificial Neural Network Regression B Testing Chart")
lines(annfb,col="green")

# 5.4.4. Artificial Neural Network Regression forecasting accuracy
# Convert xts to ts for accuracy function
annftsa <- ts(coredata(annfa),frequency=252,start=index(returntestingdata[1,]))
accuracy(annftsa,returntestingdatats)
annmasea <- accuracy(annftsa,returntestingdatats)[5]/rndmape
annmasea
annftsb <- ts(coredata(annfb),frequency=252,start=index(returntestingdata[1,]))
accuracy(annftsb,returntestingdatats)
annmaseb <- accuracy(annftsb,returntestingdatats)[5]/rndmape
annmaseb

# 5.6. Algorithm Testing Accuracy Comparison
accuracy(xgbmftsb,returntestingdatats)
accuracy(rsvmftsa,returntestingdatats)
accuracy(annftsa,returntestingdatats)
xgbmmaseb
rsvmmasea
annmasea

# 6. Machine Trading Strategies

# 6.1. Ensemble Method Trading Strategy

# 6.1.1. eXtreme Gradient Boosting Regression forecasting
# Intermediate forecasting step as newdata needs to be same length as training range 
xgbmi <- predict.train(xgbmtb,newdata=returnintertradingdata)

# Limited to trading range
xgbmdfi <- cbind(index(returnintertradingdata),as.data.frame(xgbmi)) 
xgbmli <- xts(xgbmdfi[,2],order.by=as.Date(xgbmdfi[,1]))
xgbms <- window(xgbmli,start="2018-08-26")

# 6.1.2. eXtreme Gradient Boosting Regression trading signals
xgbmsig <- Lag(ifelse(Lag(xgbms)<0&xgbms>0,1,ifelse(Lag(xgbms)>0&xgbms<0,-1,0)))
xgbmsig[is.na(xgbmsig)] <- 0

# 6.1.3. eXtreme Gradient Boosting Regression trading positions
xgbmpos <- ifelse(xgbmsig>1,1,0)

for(i in 1:length(xgbmpos)){xgbmpos[i] <- ifelse(xgbmsig[i]==1,1,ifelse(xgbmsig[i]==-1,0,xgbmpos[i-1]))}
xgbmpos[is.na(xgbmpos)] <- 0
xgbmtr <- cbind(xgbms,xgbmsig,xgbmpos)
colnames(xgbmtr) <- c("xgbms","xgbmsig","xgbmpos")
View(xgbmtr)

# 6.2. Maximum Margin Method Trading Strategy

# 6.2.1. RBF Support Vector Machine Regression forecasting
# Intermediate forecasting step as newdata needs to be same length as training range 
rsvmi <- predict.train(rsvmta,newdata=returnintertradingdata)

# Limited to trading range
rsvmdfi <- cbind(index(returnintertradingdata),as.data.frame(rsvmi))
rsvmli <- xts(rsvmdfi[,2],order.by=as.Date(rsvmdfi[,1]))
rsvms <- window(rsvmli,start="2018-08-26")

# 6.2.2. RBF Support Vector Machine Regression trading signals
rsvmsig <- Lag(ifelse(Lag(rsvms)<0&rsvms>0,1,ifelse(Lag(rsvms)>0&rsvms<0,-1,0)))
rsvmsig[is.na(rsvmsig)] <- 0

# 6.2.3. RBF Support Vector Machine Regression trading positions
rsvmpos <- ifelse(rsvmsig>1,1,0)
for(i in 1:length(rsvmpos)){rsvmpos[i] <- ifelse(rsvmsig[i]==1,1,ifelse(rsvmsig[i]==-1,0,rsvmpos[i-1]))}
rsvmpos[is.na(rsvmpos)] <- 0
rsvmtr <- cbind(rsvms,rsvmsig,rsvmpos)
colnames(rsvmtr) <- c("rsvms","rsvmsig","rsvmpos")
View(rsvmtr)

# 6.3. Multi-Layer Perceptron Method Trading Strategy

# 6.3.1. Artificial Neural Network Regression forecasting
# Intermediate forecasting step as newdata needs to be same length as training range 
anni <- predict.train(annta,newdata=returnintertradingdata)

# Limited to trading range
anndfi <- cbind(index(returnintertradingdata),as.data.frame(anni))
annli <- xts(anndfi[,2],order.by=as.Date(anndfi[,1]))
anns <- window(annli,start="2018-08-26")

# 6.3.2. Artificial Neural Network Regression trading signals
annsig <- Lag(ifelse(Lag(anns)<0&anns>0,1,ifelse(Lag(anns)>0&anns<0,-1,0)))
annsig[is.na(annsig)] <- 0

# 6.3.3. Artificial Neural Network Regression trading positions
annpos <- ifelse(annsig>1,1,0)
for(i in 1:length(annpos)){annpos[i] <- ifelse(annsig[i]==1,1,ifelse(annsig[i]==-1,0,annpos[i-1]))}
annpos[is.na(annpos)] <- 0
anntr <- cbind(anns,annsig,annpos)
colnames(anntr) <- c("anns","annsig","annpos")
View(anntr)

# 7. Machine Trading Strategies Performance Comparison

# 7.1. Ensemble Method Trading Strategy Performance Comparison
xgbmret <- xgbmpos*returntradingdata[,1]
xgbmretc <- ifelse((xgbmsig==1|xgbmsig==-1)&xgbmpos!=Lag(xgbmpos),(xgbmpos*returntradingdata[,1])-0.001,xgbmpos*returntradingdata[,1])
xgbmcomp <- cbind(xgbmret,xgbmretc,returntradingdata[,1])
colnames(xgbmcomp) <- c("xgbmret","xgbmretc","returndata")
table.AnnualizedReturns(xgbmcomp)

charts.PerformanceSummary(xgbmcomp)

# 7.2. Maximum Margin Method Trading Strategy Performance Comparison
rsvmret <- rsvmpos*returntradingdata[,1]
rsvmretc <- ifelse((rsvmsig==1|rsvmsig==-1)&rsvmpos!=Lag(rsvmpos),(rsvmpos*returntradingdata[,1])-0.001,rsvmpos*returntradingdata[,1])
rsvmcomp <- cbind(rsvmret,rsvmretc,returntradingdata[,1])
colnames(rsvmcomp) <- c("rsvmret","rsvmretc","returndata")
table.AnnualizedReturns(rsvmcomp)
charts.PerformanceSummary(rsvmcomp)

# 7.3. Multi-Layer Perceptron Method Trading Strategy Performance Comparison
annret <- annpos*returntradingdata[,1]
annretc <- ifelse((annsig==1|annsig==-1)&annpos!=Lag(annpos),(annpos*returntradingdata[,1])-0.001,annpos*returntradingdata[,1])
anncomp <- cbind(annret,annretc,returntradingdata[,1])
colnames(anncomp) <- c("annret","annretc","returndata")
table.AnnualizedReturns(anncomp)
charts.PerformanceSummary(anncomp)



# 7.4. Machine Trading Strategies Performance Comparison

# 7.4.1. Without Trading Commissions
comp <- cbind(xgbmret,rsvmret,annret,returntradingdata[,1])
colnames(comp) <- c("xgbmret","rsvmret","annret","returndata")
table.AnnualizedReturns(comp)
charts.PerformanceSummary(comp)

# 7.4.2. With Trading Commissions
compc <- cbind(xgbmretc,rsvmretc,annretc,returntradingdata[,1])
colnames(compc) <- c("xgbmretc","rsvmretc","annretc","returndata")
table.AnnualizedReturns(compc)
charts.PerformanceSummary(compc)

