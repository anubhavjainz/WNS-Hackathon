############ Importing the required libraries
library(dplyr)
library(ggplot2) #for the exploratory data analysis

library(ROCR) # ROCR model validation
#Decision Tree Libraries
library(irr)
library(rpart)
library(caret)
#Tree plotting
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(dummies)

##Random Forest Libraries
library(randomForest)


setwd("D:\\WNS HACKATHON")

getwd()

TTrain<-read.csv("train_LZdllcl.csv",header = T,sep = ",",na.strings = "")
TTest<-read.csv("test_2umaH9m.csv",header = T,sep = ",",na.strings = "")

TTrain$Set<-"Train"
TTest$Set<-"Test"

names(TTrain)

TTest$is_promoted<-NA

Full<-rbind(TTrain,TTest)

summary(Full)
########### Exploratory data analysis
str(Full)
summary(Full)

#previous_year_rating  has 5936 missing values


hist(Full$age)
boxplot(Full$age)

ggplot(data=TTrain,mapping=aes(x=age,fill=as.factor(is_promoted),alpha=0.2))+geom_density()+scale_x_continuous(limit=c(0,100))


quantile(Full$age,na.rm = T)
median(Full$Age,na.rm = T)



############### Data Preparation


#Full$is_promoted<-as.factor(Full$is_promoted) for xgb


#Full$awards_won.<-as.factor(Full$awards_won.)


Full$gender<-ifelse(Full$gender=="m",1,0)
names(Full)

#Full$KPIs_met..80.<-as.factor(Full$KPIs_met..80.)

#Full$previous_year_rating<-ordered(Full$previous_year_rating)

summary(Full$previous_year_rating)

levels(Full$education)

Full$education_B<-ifelse(Full$education=="Bachelor's",1,0)
Full$education_BS<-ifelse(Full$education=="Below Secondary",1,0)
Full$education_M<-ifelse(Full$education=="Master's & above",1,0)


hist(Full$length_of_service,breaks = 30)

Full$SERVICE_ONE<-ifelse(Full$length_of_service==1,1,0)

Full$SERVICE_TWO<-ifelse(Full$length_of_service>=2 & Full$length_of_service<=7,1,0)

Full$SERVICE_EIGHT<-ifelse(Full$length_of_service>=8 & Full$length_of_service<=10,1,0)

Full$SERVICE_TEN<-ifelse(Full$length_of_service>11 & Full$length_of_service<=20,1,0)

Full$SERVICE_TWENTY<-ifelse(Full$length_of_service>20,1,0)

Full$Rating_1<-ifelse(Full$previous_year_rating==1,1,0)
Full$Rating_2<-ifelse(Full$previous_year_rating==2,1,0)
Full$Rating_3<-ifelse(Full$previous_year_rating==3,1,0)
Full$Rating_4<-ifelse(Full$previous_year_rating==4,1,0)
Full$Rating_5<-ifelse(Full$previous_year_rating==5,1,0)

names(Full)

Full[, 16:28][is.na(Full[, 16:28])] <- 0

unique(Full$department)

### Creating dummies
dep_dumm <- as.data.frame(Full[c("department")])
names(dep_dumm) <- c("DEP_")
dep_dumm=dummy.data.frame(dep_dumm)
Full <- cbind(Full,dep_dumm)


names(Full)[35]<-"DEP_RD"
names(Full)[36]<-"DEP_Sales"

Full$is
#################### Spliting Train, Validation and Test DataSet
set.seed(100)
index<-sample(x = 1:nrow(TTrain),size = 0.6*nrow(TTrain),replace = F)
Train<-Full[Full$Set=="Train",]

Temp<-Train[-index,]

Train<-Train[index,]

set.seed(100)
index1<-sample(x = 1:nrow(Temp),size = 0.5*nrow(Temp),replace = F)
TestA<-Temp[index1,]
Validate<-Temp[-index1,]

Test<-Full[Full$Set=="Test",]

names(Train)
#####################Applying the logistic regression directly on the imbalanced dataset to check the performance


mod1<-glm(formula = is_promoted~Rating_5+Rating_2+Rating_1+SERVICE_TWO+SERVICE_ONE+education_M+education_B+age+KPIs_met..80.+awards_won.+avg_training_score+DEP_Analytics+DEP_Finance+DEP_HR+DEP_Legal+DEP_Operations+DEP_Procurement+DEP_RD+DEP_Sales+DEP_Technology, family = "binomial", data = Train)
mod1
summary(mod1)
#### Model Accuracy on the Train Data Itself
actual<-Train$is_promoted
pred<-predict(mod1,type="response")

table(Train$is_promoted)/nrow(Train)

predicted<-ifelse(pred>0.915,1,0)
actual<-as.factor(actual)
predicted<-as.factor(predicted)

#### Kappa Metric
kappa2(data.frame(actual,predicted))

#Confusion Matric
confusionMatrix(predicted,actual,positive="1")

#ROCR curve and AUC value
head(predicted)
head(as.numeric(predicted))
predicted<-as.numeric(predicted)
predicted<-ifelse(predicted==2,1,0)

head(actual)
head(as.numeric(actual))
actual<-as.numeric(actual)
actual<-ifelse(actual==2,1,0)

pred<-prediction(actual,predicted)
perf<-performance(pred,"tpr","fpr")
plot(perf,col="red")
abline(0,1, lty = 8, col = "grey")

auc<-performance(pred,"auc")
unlist(auc@y.values)

################ Testing on the Validation DataSet

actual<-Validate$is_promoted
predicted<-predict(mod1,type = "response",newdata = Validate)

predicted<-ifelse(predicted>0.915,1,0)
actual<-as.factor(actual)
predicted<-as.factor(predicted)

#### Kappa Metric
kappa2(data.frame(actual,predicted))

#Confusion Matric
confusionMatrix(predicted,actual,positive="1")

#ROCR curve and AUC value
head(predicted)
head(as.numeric(predicted))
predicted<-as.numeric(predicted)
predicted<-ifelse(predicted==2,1,0)

head(actual)
head(as.numeric(actual))
actual<-as.numeric(actual)
actual<-ifelse(actual==2,1,0)

pred<-prediction(actual,predicted)
perf<-performance(pred,"tpr","fpr")
plot(perf,col="red")
abline(0,1, lty = 8, col = "grey")

auc<-performance(pred,"auc")
unlist(auc@y.values)










#################################Apply Sample methods to Imbalanced Data#######################

# As the data has less Fraud transactions(less than 1%), we have to apply sample methods to balance the data
# We applied Over, Upper, Mixed(both) and ROSE sampling methods using ROSE package and SMOTE sampling method using DMwR package
#install.packages('ROSE')
#install.packages('DMwR')
library(DMwR)
library(ROSE)

print(table(Train$is_promoted))

30103 +2781 
# Oversampling, as Fraud transactions(1) are having less occurrence, so this Over sampling method will increase the Fraud records untill matches good records 227452
# Here N= 40113*2
30103*2
nrow(Train)
over_sample_train_data <- ovun.sample( is_promoted~Rating_5+Rating_2+Rating_1+SERVICE_TWO+SERVICE_ONE+education_M+education_B+age+KPIs_met..80.+awards_won.+avg_training_score+DEP_Analytics+DEP_Finance+DEP_HR+DEP_Legal+DEP_Operations+DEP_Procurement+DEP_RD+DEP_Sales+DEP_Technology, data = Train, method="over", N=60206)$data

print('Number of transactions in train dataset after applying Over sampling method')
print(table(over_sample_train_data$is_promoted))
2781*2
# Undersampling,as Fraud transactions(1) are having less occurrence, so this Under sampling method will descrease the Good records untill matches Fraud records, But, you see that we’ve lost significant information from the sample. 
over_sample_train_data <- ovun.sample(is_promoted ~ Rating_5+Rating_2+Rating_1+SERVICE_TWO+SERVICE_ONE+education_M+education_B+age+KPIs_met..80.+awards_won.+avg_training_score, data = Train,method="under", N=5562)$data
print('Number of transactions in train dataset after applying Under sampling method')
print(table(under_sample_train_data$is_promoted))

# Mixed Sampling, apply both under sampling and over sampling on this imbalanced data
over_sample_train_data <- ovun.sample(is_promoted ~ Rating_5+Rating_2+Rating_1+SERVICE_TWO+SERVICE_ONE+education_M+education_B+age+KPIs_met..80.+awards_won.+avg_training_score, data = Train, method="both", p=0.5, seed=222, N=30103)$data
print('Number of transactions in train dataset after applying Mixed sampling method')
print(table(both_sample_train_data$is_promoted))

# ROSE Sampling, this helps us to generate data synthetically. It generates artificial datas instead of dulicate data.
over_sample_train_data <- ROSE(is_promoted~Rating_5+Rating_2+Rating_1+SERVICE_TWO+SERVICE_ONE+education_M+education_B+age+KPIs_met..80.+awards_won.+avg_training_score+DEP_Analytics+DEP_Finance+DEP_HR+DEP_Legal+DEP_Operations+DEP_Procurement+DEP_RD+DEP_Sales+DEP_Technology, data = Train,  seed=111)$data
print('Number of transactions in train dataset after applying ROSE sampling method')
print(table(rose_sample_train_data$Class))

over_train_data$is_promoted
# SMOTE(Synthetic Minority Over-sampling Technique) Sampling
# formula - relates how our dependent variable acts based on other independent variable.
# data - input data
# perc.over - controls the size of Minority class
# perc.under - controls the size of Majority class
# since my data has less Majority class, increasing it with 200 and keeping the minority class to 100.
smote_sample_train_data <- SMOTE(Class ~ ., data = training_set, perc.over = 100, perc.under=200)
print('Number of transactions in train dataset after applying SMOTE sampling method')
print(table(smote_sample_train_data$Class))




mod1<-glm(formula = is_promoted~Rating_5+Rating_2+Rating_1+SERVICE_TWO+SERVICE_ONE+education_M+education_B+age+KPIs_met..80.+awards_won.+avg_training_score+DEP_Analytics+DEP_Finance+DEP_HR+DEP_Legal+DEP_Operations+DEP_Procurement+DEP_RD+DEP_Sales+DEP_Technology, family = "binomial", data = over_sample_train_data)
mod1
summary(mod1)
#### Model Accuracy on the Train Data Itself
actual<-over_sample_train_data$is_promoted
pred<-predict(mod1,type="response")

table(over_sample_train_data$is_promoted)/nrow(over_sample_train_data)

predicted<-ifelse(pred>0.5,1,0)
actual<-as.factor(actual)
predicted<-as.factor(predicted)

#### Kappa Metric
kappa2(data.frame(actual,predicted))

#Confusion Matric
confusionMatrix(predicted,actual,positive="1")

#ROCR curve and AUC value
head(predicted)
head(as.numeric(predicted))
predicted<-as.numeric(predicted)
predicted<-ifelse(predicted==2,1,0)

head(actual)
head(as.numeric(actual))
actual<-as.numeric(actual)
actual<-ifelse(actual==2,1,0)

pred<-prediction(actual,predicted)
perf<-performance(pred,"tpr","fpr")
plot(perf,col="red")
abline(0,1, lty = 8, col = "grey")

auc<-performance(pred,"auc")
unlist(auc@y.values)

################ Testing on the Validation DataSet

actual<-Validate$is_promoted
predicted<-predict(mod1,type = "response",newdata = Validate)

predicted<-ifelse(predicted>0.5,1,0)
actual<-as.factor(actual)
predicted<-as.factor(predicted)

#### Kappa Metric
kappa2(data.frame(actual,predicted))

#Confusion Matric
confusionMatrix(predicted,actual,positive="1")

#ROCR curve and AUC value
head(predicted)
head(as.numeric(predicted))
predicted<-as.numeric(predicted)
predicted<-ifelse(predicted==2,1,0)

head(actual)
head(as.numeric(actual))
actual<-as.numeric(actual)
actual<-ifelse(actual==2,1,0)

pred<-prediction(actual,predicted)
perf<-performance(pred,"tpr","fpr")
plot(perf,col="red")
abline(0,1, lty = 8, col = "grey")

auc<-performance(pred,"auc")
unlist(auc@y.values)


##################Random forest Algoithm

model1 <- randomForest(formula = is_promoted~Rating_5+Rating_2+Rating_1+SERVICE_TWO+SERVICE_ONE+education_M+education_B+age+KPIs_met..80.+awards_won.+avg_training_score+DEP_Analytics+DEP_Finance+DEP_HR+DEP_Legal+DEP_Operations+DEP_Procurement+DEP_RD+DEP_Sales+DEP_Technology,ntree = 500, data = over_sample_train_data, importance = TRUE)
model1
a=c()
i=5
for (i in 3:6) {
  model3 <- randomForest(is_promoted~Rating_5+Rating_2+Rating_1+SERVICE_TWO+SERVICE_ONE+education_M+education_B+age+KPIs_met..80.+awards_won.+avg_training_score,ntree = 500,mtry=i, data = over_sample_train_data, importance = TRUE)
  predValid <- predict(model3, Validate, type = "class")
  a[i-2] = mean(predValid == Validate$is_promoted)
}

a

plot(3:6,a)


######max accuracy at mtry=3
model1 <- randomForest(formula = Survived~Sex_Male+Sex_Female+Pclass_1+Pclass_2+Pclass_3+Age+SibSp+Ticket_count,ntree = 500,mtry=3, data = Train, importance = TRUE)
model1


#### Model Accuracy on the Train Data Itself
actual<-over_sample_train_data$is_promoted
predicted<-predict(model1,type = "class")
actual<-as.factor(actual)

#### Kappa Metric
kappa2(data.frame(actual,predicted))

#Confusion Matric
confusionMatrix(predicted,actual,positive="1")

#ROCR curve and AUC value
head(predicted)
head(as.numeric(predicted))
predicted<-as.numeric(predicted)
predicted<-ifelse(predicted==2,1,0)

head(actual)
head(as.numeric(actual))
actual<-as.numeric(actual)
actual<-ifelse(actual==2,1,0)

pred<-prediction(actual,predicted)
perf<-performance(pred,"tpr","fpr")
plot(perf,col="red")
abline(0,1, lty = 8, col = "grey")

auc<-performance(pred,"auc")
unlist(auc@y.values)

################ Testing on the Validation DataSet

actual<-Validate$is_promoted
predicted<-predict(model1,type = "class",newdata = Validate)
actual<-as.factor(actual)

#### Kappa Metric
kappa2(data.frame(actual,predicted))

#Confusion Matric
confusionMatrix(predicted,actual,positive="1")

#ROCR curve and AUC value
head(predicted)
head(as.numeric(predicted))
predicted<-as.numeric(predicted)
predicted<-ifelse(predicted==2,1,0)

head(actual)
head(as.numeric(actual))
actual<-as.numeric(actual)
actual<-ifelse(actual==2,1,0)

pred<-prediction(actual,predicted)
perf<-performance(pred,"tpr","fpr")
plot(perf,col="red")
abline(0,1, lty = 8, col = "grey")

auc<-performance(pred,"auc")
unlist(auc@y.values)



actual<-TestA$is_promoted
predicted<-predict(model1,type = "class",newdata = TestA)
actual<-as.factor(actual)

#### Kappa Metric
kappa2(data.frame(actual,predicted))

#Confusion Matric
confusionMatrix(predicted,actual,positive="1")

#ROCR curve and AUC value
head(predicted)
head(as.numeric(predicted))
predicted<-as.numeric(predicted)
predicted<-ifelse(predicted==2,1,0)

head(actual)
head(as.numeric(actual))
actual<-as.numeric(actual)
actual<-ifelse(actual==2,1,0)

pred<-prediction(actual,predicted)
perf<-performance(pred,"tpr","fpr")
plot(perf,col="red")
abline(0,1, lty = 8, col = "grey")

auc<-performance(pred,"auc")
unlist(auc@y.values)






library(xgboost)

names(over_sample_train_data)

Train_x<-over_sample_train_data[,c("Rating_5","Rating_2","Rating_1","SERVICE_TWO","SERVICE_ONE","education_M","education_B","age","KPIs_met..80.","awards_won.","avg_training_score","DEP_Analytics","DEP_Finance","DEP_HR","DEP_Legal","DEP_Operations", "DEP_Procurement","DEP_RD","DEP_Sales","DEP_Technology" )]

Train_y<-over_sample_train_data[,c("is_promoted")]

over_sample_train_data$is_promoted
View(Train_y)


names(TestA)
names(over_sample_train_data)

Validate_x<-TestA[,c("Rating_5","Rating_2","Rating_1","SERVICE_TWO","SERVICE_ONE","education_M","education_B","age","KPIs_met..80.","awards_won.","avg_training_score","DEP_Analytics","DEP_Finance","DEP_HR","DEP_Legal","DEP_Operations", "DEP_Procurement","DEP_RD","DEP_Sales","DEP_Technology" )]
Validate_y<-TestA[,c("is_promoted")]


View(Train)
#put into the xgb matrix format

dtrain = xgb.DMatrix(data =  as.matrix(Train_x), label = Train_y )
dtest = xgb.DMatrix(data =  as.matrix(Validate_x), label = Validate_y)

View(Train_y)

# these are the datasets the rmse is evaluated for at each iteration
watchlist = list(train=dtrain, test=dtest)


# try 1 - off a set of paramaters I know work pretty well for most stuff

xgb.tr

bst = xgb.train(data = dtrain, 
                max.depth = 8, 
                eta = 0.3, 
                nthread = 2, 
                nround = 1000, 
                watchlist = watchlist, 
                objective = "binary:logistic", 
                early_stopping_rounds = 50,
                print_every_n = 500)
xgb.tr

bst_slow = xgb.train(data = dtrain, 
                     max.depth=5, 
                     eta = 0.01, 
                     nthread = 2, 
                     nround = 10000, 
                     watchlist = watchlist, 
                     objective = "binary:logistic", 
                     early_stopping_rounds = 50,
                     print_every_n = 500)




Test_x<-Validate[,c("Rating_5","Rating_2","Rating_1","SERVICE_TWO","SERVICE_ONE","education_M","education_B","age","KPIs_met..80.","awards_won.","avg_training_score","DEP_Analytics","DEP_Finance","DEP_HR","DEP_Legal","DEP_Operations", "DEP_Procurement","DEP_RD","DEP_Sales","DEP_Technology" )]



dtest = xgb.DMatrix(data =  as.matrix(Test_x))

#test the model on truly external data

y_hat_valid = predict(bst, dtest)

table(over_sample_train_data$is_promoted)/nrow(over_sample_train_data)

predicted<-ifelse(y_hat_valid>0.50,1,0)

actual<-Validate$is_promoted
actual<-as.factor(actual)
predicted<-as.factor(predicted)

#### Kappa Metric
kappa2(data.frame(actual,predicted))

#Confusion Matric
confusionMatrix(predicted,actual,positive="1")

#ROCR curve and AUC value
head(predicted)
head(as.numeric(predicted))
predicted<-as.numeric(predicted)
predicted<-ifelse(predicted==2,1,0)

head(actual)
head(as.numeric(actual))
actual<-as.numeric(actual)
actual<-ifelse(actual==2,1,0)

pred<-prediction(actual,predicted)
perf<-performance(pred,"tpr","fpr")
plot(perf,col="red")
abline(0,1, lty = 8, col = "grey")

auc<-performance(pred,"auc")
unlist(auc@y.values)












Test$Survived<-predicted


submission<-Test%>%select(PassengerId,Survived)


write.csv(x = submission,file = "Gender_submission.csv",row.names = F)



###
# Grid search first principles 
###

max.depths = c(7, 9)
etas = c(0.01, 0.001)

best_params = 0
best_score = 0

count = 1
for( depth in max.depths ){
  for( num in etas){
    
    bst_grid = xgb.train(data = dtrain, 
                         max.depth = depth, 
                         eta=num, 
                         nthread = 2, 
                         nround = 10000, 
                         watchlist = watchlist, 
                         objective = "binary:logistic", 
                         early_stopping_rounds = 50, 
                         verbose=0)
    
    if(count == 1){
      best_params = bst_grid$params
      best_score = bst_grid$best_score
      count = count + 1
    }
    else if( bst_grid$best_score < best_score){
      best_params = bst_grid$params
      best_score = bst_grid$best_score
    }
  }
}

best_params
best_score


bst_tuned = xgb.train( data = dtrain, 
                       max.depth = 9, 
                       eta = 0.01, 
                       nthread = 2, 
                       nround = 10000, 
                       watchlist = watchlist, 
                       objective = "binary:logistic", 
                       early_stopping_rounds = 50,
                       print_every_n = 500)

y_hat_xgb_grid = predict(bst_tuned, dtest)





predicted<-ifelse(y_hat_xgb_grid>0.60,1,0)


Test$Survived<-predicted


submission<-Test%>%select(PassengerId,Survived)


write.csv(x = submission,file = "Gender_submission.csv",row.names = F)


