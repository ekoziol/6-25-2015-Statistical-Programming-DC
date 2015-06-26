###### Created by
######  Eric Koziol
######  For Statistical Programming DC Meetup
######  On 6/25/2015
######  Machine Learning in R

#declare required packages
require(caret)
require(h2o)
require(GGally) #should load ggplot with it
require(pROC)
#change this directory to wherever you put the data
setwd("~/Talks/6-25-2015 Statistical Programming DC")


myseed = 1337 
set.seed(myseed)

#read in data and name columns
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)
pima <- read.csv("pima-indians-diabetes.data", header=FALSE)
pnames <- c("NumPreg", "Glucose", "BloodPressure", "Triceps", 
            "Insulin", "BMI", "Pedigree", "Age", "Class")
names(pima) <- pnames

#make prediction column a factor otherwise caret will think it is a regression problem
pima$Class[pima$Class == 0 ] <- "N" # Negative Class
pima$Class[pima$Class == 1 ] <- "P" # Positive Class
pima$Class <- as.factor(pima$Class)
#separate data into train, test and cv sets = 60% train, 20% test, 20% cv
pima$set <- sample(c(1,2,3), nrow(pima), replace = TRUE, prob = c(0.6,0.2,0.2))

pimaTrain <- pima[pima$set == 1, pnames]
pimaTest <- pima[pima$set == 2, pnames]
pimaCV <- pima[pima$set == 3, pnames]



#jump right in!

firstTimeCaret <- train(Class ~., data = pimaTrain, method = "rf")
firstPrediction <- predict(firstTimeCaret, pimaTest)
confusionMatrix(firstPrediction, pimaTest$Class)

#look at a pairs plot
png("PimaPairs.png", height = 3000, width = 3000)
print(ggpairs(pimaTrain, color="Class", alpha = 0.3))
dev.off()


#select new features and retrain
#all three training methods are the same
lessFeaturesFit <- train(Class ~ Triceps + BMI + Age + Glucose, data = pimaTrain, method = "rf")
lessFeaturesFit <- train(Class ~ ., 
                         data = pimaTrain[, c("Triceps", "BMI", "Age", "Glucose", "Class")], 
                         method = "rf")

selectedFeatures <- c("Triceps", "BMI", "Age", "Glucose", "Class")

lessFeaturesFit <- train(Class ~ ., data = pimaTrain[,selectedFeatures], method = "rf")

lessPrediction <- predict(lessFeaturesFit, pimaTest)
confusionMatrix(lessPrediction, pimaTest$Class)


#inspect data set to see if balanced 
qplot(pimaTrain$Class)

#since the data is unbalanced we need to find 
#a different metric for assessing the different algorithms
#define train control to be used for each algorithm
trcontrol <- trainControl(classProbs = TRUE, 
                          verboseIter = FALSE,
                          number = 10, 
                          method = "repeatedcv", 
                          summaryFunction=twoClassSummary)

set.seed(myseed)
#caret defaults to 500 trees
rf <- train(Class ~. , data = pimaTrain, method="rf", metric="ROC", 
            trControl = trcontrol)

#typical prediction, specification of newdata= is unncessary
#type = "prob" allows for ROC curve to be produced
rfPred <- predict(rf, newdata=pimaTest, type = "prob")
rfROC <- roc(pimaTest$Class, rfPred$P, plot=TRUE)

#caret does not auto tune the number of trees so it must be specified
set.seed(myseed)
rf1000 <- train(Class ~. , data = pimaTrain, ntree = 1000, method="rf", metric="ROC", 
            trControl = trcontrol)

rf1000Pred <- predict(rf1000, pimaTest, type = "prob")

rf1000ROC <- roc(pimaTest$Class, rf1000Pred$P, plot=TRUE)

#train knn with ROC
set.seed(myseed)
knn <- train(Class ~ . , data = pimaTrain, method="knn", metric="ROC", 
             tuneLength = 20, 
             trControl = trcontrol)

knnPred <- predict(knn, newdata=pimaTest, type = "prob")
knnROC <- roc(pimaTest$Class, knnPred$P, plot=TRUE)

#look at the effect of scaling with knn
set.seed(myseed)
knnScaled <- train(Class ~ . , data = pimaTrain, method="knn", metric="ROC", 
             tuneLength = 20, preProcess = c("center", "scale"),
             trControl = trcontrol)



knnScaledPred <- predict(knnScaled, newdata=pimaTest, type = "prob")
knnScaledROC <- roc(pimaTest$Class, knnScaledPred$P, plot=TRUE)

#train SVM generally
set.seed(myseed)
svm <- train(Class ~ . , data = pimaTrain, method="svmRadial", metric="ROC", 
             preProcess = c("center", "scale"),
             trControl = trcontrol)

svmPred <- predict(svm, pimaTest, type = "prob")
svmROC <- roc(pimaTest$Class, svmPred$P, plot=TRUE)

svmLinear <- train(Class ~ . , data = pimaTrain, method="svmLinear", metric="ROC", 
             preProcess = c("center", "scale"),
             trControl = trcontrol)

svmLinearPred <- predict(svmLinear, pimaTest, type = "prob")
svmLinearROC <- roc(pimaTest$Class, svmLinearPred$P, plot=TRUE)

#train SVM with tune length
set.seed(myseed)
svmTuned10 <- train(Class ~ . , data = pimaTrain, method="svmRadial", metric="ROC", 
             tuneLength = 10,
             preProcess = c("center", "scale"),
             trControl = trcontrol)

plot(svmTuned10)
svmTuned10Pred <- predict(svmTuned10, pimaTest, type = "prob")

svmTuned10ROC <- roc(pimaTest$Class, svmTuned10Pred$P, plot=TRUE)


#train SVM with tune grid
set.seed(myseed)
svmGrid <- expand.grid(C=c(0.001,0.01,0.1,0.5,1,2), sigma = c(0.001,0.01,0.1,0.5,1,2))
svmTunedGrid<- train(Class ~ . , data = pimaTrain, method="svmRadial", metric="ROC", 
                    tuneGrid = svmGrid,
                    preProcess = c("center", "scale"),
                    trControl = trcontrol)

svmTunedGridPred <- predict(svmTunedGrid, pimaTest, type = "prob")
svmTunedGridROC <- roc(pimaTest$Class, svmTunedGridPred$P, plot=TRUE)

#brief intro into h2o Neural Network
localH2O <- h2o.init(nthread=16, max_mem_size="4g")

train.hex <- as.h2o(localH2O,pimaTrain)
test.hex <- as.h2o(localH2O,pimaTest)


predictors <- c(1:(ncol(train.hex)-1))
response <- ncol(train.hex)

#note that nfolds is currently not supported.  Use dropout methods instead
#to help avoid overfitting
#documentation unclear if h2o automatically standarizes inputs or not
#current h2o implementation will automatically mean impute missing values
h2omodel <- h2o.deeplearning(x=predictors,
                          y=response,
                          training_frame=train.hex,
                          classification_stop = -1,
                          activation="TanhWithDropout",
                          hidden=c(20,20,20),
                          hidden_dropout_ratio=c(0.5,0.5,0.5),
                          input_dropout_ratio=0.05,
                          epochs=50,
                          l1=1e-5,
                          l2=1e-5,
                          rho=0.99,
                          epsilon=1e-8,
                          train_samples_per_iteration=100,
                          max_w2=10,
                          seed=myseed,
                          balance_classes = T)

h2oPred <- as.data.frame(h2o.predict(h2omodel,test.hex))
h2oROC <- roc(pimaTest$Class, h2oPred$P, plot=TRUE)





#brief ensemble with simple averaging
ensemblePred <- (rfPred$P + rf1000Pred$P + knnPred$P + knnScaledPred$P + 
                   svmPred$P + h2oPred$P + svmTunedGridPred$P)/7
ensembleROC <- roc(pimaTest$Class, ensemblePred, plot=TRUE)

#compareROC charts
plot(ensembleROC, col = "black")
plot(h2oROC, col = "red", add = T)
plot(svmROC, col = "green", add = T)
plot(knnScaledROC, col = "orange", add = T)
plot(rfROC, col = "blue", add = T)

cv.hex <- as.h2o(localH2O, pimaCV)
CVh2oPred <- as.data.frame(h2o.predict(h2omodel,cv.hex))
CVh2oROC <- roc(pimaCV$Class, CVh2oPred$P, plot=FALSE)

#type each prediction function explicitely for ease of learning
#normally perform sapply on a dataframe
ens1 <- predict(rf, pimaCV, type = "prob")
ens2 <- predict(rf1000, pimaCV, type = "prob")
ens3 <- predict(knn, pimaCV, type = "prob")
ens4 <- predict(knnScaled, pimaCV, type = "prob")
ens5 <- predict(svm, pimaCV, type = "prob")
ens6 <- predict(svmTunedGrid, pimaCV, type = "prob")
                   
CVensemblePred <- (ens1$P + ens2$P + ens3$P + ens4$P + 
                     ens5$P + ens6$P + CVh2oPred$P)/7
CVensembleROC <- roc(pimaCV$Class, CVensemblePred, plot=FALSE)

plot(CVensembleROC, col= "red")
plot(CVh2oROC, col = "blue", add = TRUE)

