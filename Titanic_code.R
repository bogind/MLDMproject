# My Titanic competition submission code
library(randomForest); library(caret); library(doSNOW); library(rpart); library(rpart.plot);
library(infotheo); library(corrplot)

# feature engineering -------------------------
# combining the train and test sets
test$Survived <- NA
combined <- rbind(train, test)

combined$Survived <- as.factor(combined$Survived)
combined$Pclass <- as.factor(combined$Pclass)
combined$Sex <- as.factor(combined$Sex)

# a function that takes the full name and extracts the title from it
title_create <- function(f_name) {
  title<-gsub(" ","",unlist(strsplit(f_name,"[,.]")))[2]
  return(title)
}

#create a new title column and populate it with titles
combined$title<-NA

for(i in 1:dim(combined)[1])
{
  combined$title[i]<-title_create(combined$Name[i])
}

combined$title <- as.factor(combined$title)
combined$title[which(combined$title%in%c('Capt','Col','Don','Major','Jonkheer','Rev','Sir'))]<- 'Mr'
combined$title[which(combined$title=='Dona')]<- 'Mrs'
combined$title[which(combined$title=='Lady')]<- 'Mrs'
combined$title[which(combined$title=='Mme')]<- 'Miss'
combined$title[which(combined$title=='Mlle')]<- 'Miss'
combined$title[which(combined$title%in%c('Ms','theCountess'))]<- 'Miss'
combined$title[which(combined$PassengerId==797)]<- 'Miss'
combined$title[which(combined$title=='Dr')]<- 'Mr'
combined<-droplevels(combined)

#create family member size by summing sibsp+parch+1
combined$fsize<-NA

for(i in 1:dim(combined)[1])
{
  combined$fsize[i]<-combined$SibSp[i]+combined$Parch[i]+1
}


#some people are not a family but travel together
t_list <- unique(combined$Ticket)
combined$group <- NA
for(i in t_list)
{
  num <- combined$PassengerId[which(combined$Ticket==i)]
  counter <- length(num)
  combined$group[which(combined$Ticket==i)] <- counter
}

#fare is not per-ticket but all the sum together
combined$Ticket_price <- combined$Fare/combined$group


#one fare is na in the test, ill impute it with the median of combined id=1044
combined[which(combined$PassengerId==1044),]
median(combined$Fare[which(combined$title=='Mr' & combined$Pclass == 3 & combined$group == 1)])
combined$Fare[which(combined$PassengerId==1044)] <- 7.85


# cross validation function --------------------
crossFun <- function(seed, X, Y, K, Times, algo) {
  set.seed(seed)
  multiFolds <- createMultiFolds(Y, k = K, times = Times)
  cv_ctrl <- trainControl(method ='repeatedcv', number = K, repeats = Times, index = multiFolds)
  
  cl <- makeCluster(3, type = 'SOCK')
  registerDoSNOW(cl)

   result <- train(x = X, y = Y, method = algo, trControl = cv_ctrl)
  
  stopCluster(cl)
  
  return(result)
}


# after 1st submission feature engineering ---------------
combined$Age <- age_col$Age       #resets age for different imputations

combined$child <- 0
combined$child[which(combined$Age<14)] <- 1

library(e1071)
skewness(combined$Ticket_price) 


# feature selection -------------------------------

cart_cv <- crossFun(118,X,Y,3,10,'rpart')
prp(cart_cv$finalModel,type = 0,extra = 1,under = TRUE)


mutinformation(Y, combined$Pclass[1:891])*100
mutinformation(Y, discretize(combined$child[1:891]))*100

corMat <- cor(combined[,unlist(lapply(combined, is.numeric))])
corrplot(corMat, method='number', type='upper', order = "hclust")
findCorrelation(corMat, cutoff = .75)
# rf+cv --------------------------

features <- c('Pclass','title','group','Ticket_price')
X <- combined[1:891,features]
Y <- combined$Survived[1:891]

rf_cv <- crossFun(118,X,Y,10,10,'rf')
rf_cv
rf_cv$finalModel

rf_cv$resample


# logistic + cv -------------------------------

log_cv <- crossFun(118,X,Y,3,10,'glm',1)
log_cv
log_cv$finalModel
summary(log_cv$finalModel)

# submission to Kaggle ----------------------

Xtest <- combined[892:1309,features]
subm <- predict(rf_cv$finalModel,Xtest) 
tabb <- data.frame(PassengerId=test$PassengerId,Survived=subm)
write.csv(tabb, file = 'RF_CV_#11.csv',row.names = FALSE)


# missForest imputation for Age ------------------------
library(missForest)
excludes <- names(combined) %in% c('Name','Ticket','Cabin','Embarked','deck')
mf_imp <- missForest(combined[!excludes])
combined$Age<-mf_imp$ximp$Age
combined$Age <- round(combined$Age,0)
mf_imp$OOBerror

# data balancing -------------------------
s <- sample(549,342)
survived <- combined[which(combined$Survived == 1),]
died <- combined[which(combined$Survived == 0),]
selected_died <- died[s,]  
combined_balanced <- rbind(survived , selected_died)


features_b <- c('Pclass','title','group','Ticket_price')
X_b <- combined_balanced[,features]
Y_b <- combined_balanced$Survived

rf_cv_b <- crossFun(118,X_b,Y_b,3,10,'rf',5)
rf_cv_b
rf_cv_b$finalModel

cart_cv_b <- crossFun(118,X_b,Y_b,3,10,'rpart',30)
prp(cart_cv_b$finalModel,type = 0,extra = 1,under = TRUE)

subm <- predict(rf_cv_b$finalModel,Xtest) 
tabb <- data.frame(PassengerId=test$PassengerId,Survived=subm)
write.csv(tabb, file = 'RF_CV_#3.csv',row.names = FALSE)


# missForest imputation for deck ----------------
combined$deck <- NA
combined$deck <- substr(combined$Cabin,1,1) #gets the deck number out of the cabin
combined$deck <- as.factor(combined$deck)

library(missForest)
excludes <- names(combined) %in% c('Name','Ticket','Cabin','Embarked')
deck_mf_imp <- missForest(combined[!excludes])
deck_mf_imp$OOBerror
combined$deck<-deck_mf_imp$ximp$deck

knnFit <- knn3(x=X, y=Y, k=5)
knnFit


