rm(list=ls(all=T))
setwd("G:/Edwisor")

#Current working directory
getwd()

train = read.csv("Train_data.csv", header = T, na.strings = c(" ", "", "NA"))
test = read.csv("Test_data.csv", header = T, na.strings = c(" ", "", "NA"))

data = rbind(train, test)
View(data)

class(data)
summary(data)
length(unique(data))
colnames(data)


data$area.code = as.factor(data$area.code)
data$phone.number = NULL

for(i in 1:ncol(data)){
  
  if(class(data[,i]) == 'factor'){
    
    data[,i] = factor(data[,i], labels=(1:length(levels(factor(data[,i])))))
    
  }
}


#missingvalues

missing_val = data.frame(apply(data,2,function(x){sum(is.na(x))}))
View(missing_val)

new = data

data = new


#Boxplot

numeric_index = sapply(data,is.numeric)

numeric_data = data[,numeric_index]

cnames = colnames(numeric_data)

for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Churn"), data = subset(data))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="Churn")+
           ggtitle(paste("Box plot of Churn for",cnames[i])))
}

gridExtra::grid.arrange(gn1,gn2,gn3,gn4,gn5,ncol=5)

gridExtra::grid.arrange(gn6,gn7,gn8,gn9,gn10,ncol=5)

gridExtra::grid.arrange(gn11,gn12,gn13,gn14,gn15,ncol=5)



#no.of.outliers
for(i in cnames){
     val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
     print(length(val))
   }

#replacing outliers
qplot(y=data$account.length, x= 1, geom = "boxplot")
quantile(data$account.length, c(.01,.99))
data[which(data$account.length>194),("account.length")] = 194
qplot(y=data$account.length, x= 1, geom = "boxplot")

qplot(y=data$number.vmail.messages, x= 1, geom = "boxplot")
quantile(data$number.vmail.messages, c(.98))
data[which(data$number.vmail.messages>41),("number.vmail.messages")] = 42
qplot(y=data$number.vmail.messages, x= 1, geom = "boxplot")

qplot(y=data$total.day.minutes, x= 1, geom = "boxplot")
quantile(data$total.day.minutes, c(.01,.99))
data[which(data$total.day.minutes>304.61),("total.day.minutes")] = 304.61
data[which(data$total.day.minutes<54.2),("total.day.minutes")] = 54.2
qplot(y=data$total.day.minutes, x= 1, geom = "boxplot")


qplot(y=data$total.day.calls, x= 1, geom = "boxplot")
quantile(data$total.day.calls, c(.01,.99))
data[which(data$total.day.calls>146),("total.day.calls")] = 146
data[which(data$total.day.calls<54),("total.day.calls")] = 54
qplot(y=data$total.day.calls, x= 1, geom = "boxplot")


qplot(y=data$total.day.charge, x= 1, geom = "boxplot")
quantile(data$total.day.charge, c(.01,.99))
data[which(data$total.day.charge>51.78),("total.day.charge")] = 51.78
data[which(data$total.day.charge<9.21),("total.day.charge")] = 9.21
qplot(y=data$total.day.charge, x= 1, geom = "boxplot")


qplot(y=data$total.night.minutes, x= 1, geom = "boxplot")
quantile(data$total.night.minutes, c(.01,.99))
data[which(data$total.night.minutes>318),("total.night.minutes")] = 318
data[which(data$total.night.minutes<81.6),("total.night.minutes")] = 81.6
qplot(y=data$total.night.minutes, x= 1, geom = "boxplot")

qplot(y=data$total.night.calls, x= 1, geom = "boxplot")
quantile(data$total.night.calls, c(.01,.99))
data[which(data$total.night.calls>148),("total.night.calls")] = 148
data[which(data$total.night.calls<54.99),("total.night.calls")] = 54.99
qplot(y=data$total.night.calls, x= 1, geom = "boxplot")

qplot(y=data$total.night.charge, x= 1, geom = "boxplot")
quantile(data$total.night.charge, c(.01,.99))
data[which(data$total.night.charge>14.31),("total.night.charge")] = 14.31
data[which(data$total.night.charge<3.67),("total.night.charge")] = 3.67
qplot(y=data$total.night.charge, x= 1, geom = "boxplot")

qplot(y=data$total.eve.minutes, x= 1, geom = "boxplot")
quantile(data$total.eve.minutes, c(.01,.99))
data[which(data$total.eve.minutes>318.8),("total.eve.minutes")] = 318.8
data[which(data$total.eve.minutes<80.6),("total.eve.minutes")] = 80.6
qplot(y=data$total.eve.minutes, x= 1, geom = "boxplot")

qplot(y=data$total.eve.calls, x= 1, geom = "boxplot")
quantile(data$total.eve.calls, c(.01,.99))
data[which(data$total.eve.calls>147),("total.eve.calls")] = 147
data[which(data$total.eve.calls<54),("total.eve.calls")] = 54
qplot(y=data$total.eve.calls, x= 1, geom = "boxplot")

qplot(y=data$total.eve.charge, x= 1, geom = "boxplot")
quantile(data$total.eve.charge, c(.01,.99))
data[which(data$total.eve.charge>27.1),("total.eve.charge")] = 27.1
data[which(data$total.eve.charge<6.85),("total.eve.charge")] = 6.85
qplot(y=data$total.eve.charge, x= 1, geom = "boxplot")

qplot(y=data$total.intl.minutes, x= 1, geom = "boxplot")
quantile(data$total.intl.minutes, c(.01,.99))
data[which(data$total.intl.minutes>16.6),("total.intl.minutes")] = 16.6
data[which(data$total.intl.minutes<3.5),("total.intl.minutes")] = 3.5
qplot(y=data$total.intl.minutes, x= 1, geom = "boxplot")

qplot(y=data$total.intl.calls, x= 1, geom = "boxplot")
quantile(data$total.intl.calls, c(.97))
data[which(data$total.intl.calls>10),("total.intl.calls")] = 12.01
qplot(y=data$total.intl.calls, x= 1, geom = "boxplot")

qplot(y=data$total.intl.charge, x= 1, geom = "boxplot")
quantile(data$total.intl.charge, c(.01,.99))
data[which(data$total.intl.charge>4.48),("total.intl.charge")] = 4.48
data[which(data$total.intl.charge<0.95),("total.intl.charge")] = 0.95
qplot(y=data$total.intl.charge, x= 1, geom = "boxplot")


qplot(y=data$number.customer.service.calls, x= 1, geom = "boxplot")
quantile(data$number.customer.service.calls, c(.92))
data[which(data$number.customer.service.calls>),("number.customer.service.calls")] = 6
qplot(y=data$number.customer.service.calls, x= 1, geom = "boxplot")


for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Churn"), data = subset(data))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="Churn")+
           ggtitle(paste("Box plot of Churn for",cnames[i])))
}

gridExtra::grid.arrange(gn1,gn2,gn3,gn4,gn5,ncol=5)

gridExtra::grid.arrange(gn6,gn7,gn8,gn9,gn10,ncol=5)

gridExtra::grid.arrange(gn11,gn12,gn13,gn14,gn15,ncol=5)

df = data

summary(data)

#correlation
corrgram(data[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


#chisquare
factor_index = sapply(data,is.factor)
factor_data = data[,factor_index]

for (i in 1:4)
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}

new = data

data = subset(data, 
                select = -c(area.code,total.day.minutes,total.eve.minutes,total.night.minutes,total.intl.minutes))

#Normalitycheck

qqnorm(data$account.length)
hist(data$account.length) #normal
hist(data$number.vmail.messages) #normal
hist(data$total.day.calls) #normal
hist(data$total.day.charge) #Normal
hist(data$total.eve.calls) #normal
hist(data$total.eve.charge)
hist(data$total.night.calls) #normal
hist(data$total.night.charge) #normal
hist(data$total.intl.calls) #left
hist(data$total.intl.charge) #right
hist(data$number.customer.service.calls) #left


df=data


library(DataCombine)
set.seed(1234)
train.index = createDataPartition(data$Churn, p = .70, list = FALSE)
train = data[ train.index,]
test  = data[-train.index,]

#Develop Model on training data
C50_model = C5.0(Churn ~., train, trials = 100, rules = TRUE)

summary(C50_model)


#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules.txt")

#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-15], type = "class")


##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$Churn, C50_Predictions)
confusionMatrix(ConfMatrix_C50)

#Randomforest
RF_model = randomForest(Churn ~ ., train, importance = TRUE, ntree = 100)

#transform rf object to an inTrees' format
treeList = RF2List(RF_model)

#Extract rules
exec = extractRules(treeList, train[,-17])  # R-executable conditions

exec[1:2,]

# #Make rules more readable:
readableRules = presentRules(exec, colnames(train))
readableRules[1:2,]

ruleMetric = getRuleMetric(exec, train[,-17], train$Churn)  # get rule metrics
# 
# #evaulate few rules
ruleMetric[1:2,]

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, test[,-15])

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$Churn, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

#Logistic Regression
logit_model = glm(Churn ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


##Evaluate the performance of classification model
ConfMatrix_RF = table(test$Churn, logit_Predictions)

##KNN Implementation
library(class)

#Predict test data
KNN_Predictions = knn(train[, 1:14], test[, 1:14], train$Churn, k = 7)

#Confusion matrix
Conf_matrix = table(KNN_Predictions, test$Churn)

#Accuracy
sum(diag(Conf_matrix))/nrow(test)

#naive Bayes
library(e1071)

#Develop model
NB_model = naiveBayes(Churn ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:14], type = 'class')

#Look at confusion matrix
Conf_matrix = table(observed = test[,15], predicted = NB_Predictions)
confusionMatrix(Conf_matrix)


