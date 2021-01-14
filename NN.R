######################################################
# NN  Analysis with R                                #             
# Rýdvan Sözen                                       #
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

# 1.2.1. Yahoo Finance 

# getFX("USD/AUD") 

# data <- USDAUD$USD.AUD 



getSymbols("XOM",to="2018-01-01")



data <- XOM$XOM.Adjusted

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
lmtb <- lm(returndata~returndata1+returndata6+returndata8,data=returntrainingdata)
summary(lmtb)
# Clear Plots area before running code
par(mfrow=c(1,2))
plot(coredata(returntrainingdata$returndata1),coredata(returntrainingdata$returndata),xlab="returndata1training",ylab="returntrainingdata")
abline(lm(returntrainingdata$returndata~returntrainingdata$returndata1),col="red")
plot(coredata(returntrainingdata$returndata6),coredata(returntrainingdata$returndata),xlab="returndata2t",ylab="returntrainingdata")
abline(lm(returntrainingdata$returndata~returntrainingdata$returndata6),col="red")
plot(coredata(returntrainingdata$returndata8),coredata(returntrainingdata$returndata),xlab="returndata5t",ylab="returntrainingdata")
abline(lm(returntrainingdata$returndata~returntrainingdata$returndata8),col="red")
par(mfrow=c(1,1))


# 4.2. Predictor Features Correlation
predictcor  <- round(cor(returntrainingdata[,2:10]),2)
predictcor
corrplot(predictcor,type="lower")

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

returndata1fts <- ts(coredata(returntestingdata[,1]),frequency=256)


returndata1fts <- ts(coredata(returntestingdata[,2]),frequency=256)



# 5.2.1. Time Series Cross-Validation
tsctrlt <- trainControl(method="timeslice",initialWindow=16,horizon=8,fixedWindow=TRUE)

# 5.4. Multi-Layer Perceptron Methods

# Artificial Neural Network Regression 

# 5.4.1. Artificial Neural Network Regression training
annta <- train(returndata~returndata1+returndata6+returndata8,data=returntrainingdata,method="neuralnet",trControl=tsctrlt)
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
annfa <- window(annla,start=index(returntestingdata[1,0]))
anndfb <- cbind(index(returnintertestingdata),as.data.frame(annpb))
annlb <- xts(anndfb[,2],order.by=as.Date(anndfb[,1]))
annfb <- window(annlb,start=index(returntestingdata[1,0]))



# 5.4.3. Artificial Neural Network Regression testing chart
plot(returntestingdata[,1],type="l",main="Artificial Neural Network Regression A Testing Chart")
lines(annfa,col="blue")
plot(returntestingdata[,1],type="l",main="Artificial Neural Network Regression B Testing Chart")
lines(annfb,col="green")

# 5.4.4. Artificial Neural Network Regression forecasting accuracy
# Convert xts to ts for accuracy function
annftsa <- ts(coredata(annfa),frequency=256)
accuracy(annftsa,returndata1fts)

rndmape <- accuracy(returndata1fts,returndata1fts)[5]


annmasea <- accuracy(annftsa,returndata1fts)[5]/rndmape
annmasea

annftsb <- ts(coredata(annfb),frequency=256)
accuracy(annftsb,returndata1fts)
annmaseb <- accuracy(annftsb,returndata1fts)[5]/rndmape
annmaseb

# 5.6. Algorithm Testing Accuracy Comparison

accuracy(annftsa,returndata1fts)


annmasea
annmaseb

# 6. Machine Trading Strategies



# 6.3. Multi-Layer Perceptron Method Trading Strategy

# 6.3.1. Artificial Neural Network Regression forecasting
# Intermediate forecasting step as newdata needs to be same length as training range 
anni <- predict.train(annta,newdata=returnintertradingdata)

# Limited to trading range
anndfi <- cbind(index(returnintertradingdata),as.data.frame(anni))
annli <- xts(anndfi[,2],order.by=as.Date(anndfi[,1]))
anns <- window(annli,start=index(returntradingdata[1,]))

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
# 7.3. Multi-Layer Perceptron Method Trading Strategy Performance Comparison
annret <- annpos*returntradingdata[,1]
annretc <- ifelse((annsig==1|annsig==-1)&annpos!=Lag(annpos),(annpos*returntradingdata[,1])-0.0001,annpos*returntradingdata[,1])
anncomp <- cbind(annret,annretc,returntradingdata[,1])
colnames(anncomp) <- c("annret","annretc","rspy")
table.AnnualizedReturns(anncomp)
charts.PerformanceSummary(anncomp)


