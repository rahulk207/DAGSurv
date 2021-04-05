library(bnlearn)
library(tidyverse)
library(orientDAG)

train = read.csv( "kkbox_sample.csv", na.strings=c("NA",""), stringsAsFactors=FALSE,header=TRUE)
train <- subset(train, select = -c(event, index_survival, msno, censor_duration))
Factind = sapply(train, is.character)
train[Factind] = lapply(train[Factind], factor)
#omit missing data and generate graph
newdata<-na.omit(train)
newdata[]<-lapply(newdata,as.numeric)
dag <- hc(newdata)
#fitted = bn.fit(dag, newdata)
plot(dag)
#impute missing data and generate graph
cancerdata<-train
cancerdata[]<-lapply(cancerdata,as.numeric)
sem =structural.em(cancerdata, maximize = "hc", maximize.args = list(), fit = "mle",
                   fit.args = list(), impute="bayes-lw", impute.args = list(), return.all = FALSE,
                   start = NULL, max.iter = 1, debug = FALSE)
plot(sem)
adjmatrix <- bn_to_adjmatrix(sem)
# newdata<-na.omit(cancerdata)
# newdata <- newdata %>% mutate(CELLULARITY = replace(CELLULARITY, CELLULARITY ==  "", NA))
# train$CELLULARITY<-as.numeric(as.factor(train$CELLULARITY))
# train$CHEMOTHERAPY<-as.numeric(as.factor(train$CHEMOTHERAPY))
# train$HORMONE_THERAPY<-as.numeric(as.factor(train$HORMONE_THERAPY))
# train$INFERRED_MENOPAUSAL_STATE<-as.numeric(as.factor(train$INFERRED_MENOPAUSAL_STATE))
# train$ER_IHC<-as.numeric(as.factor(train$ER_IHC))
# train$OS_STATUS<-as.numeric(as.factor(train$OS_STATUS))