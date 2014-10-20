# Weight Lifting Classification
Chris Breaux  

### Introduction

This exercise is the course project for the Practical Machine Learning class for Johns Hopkins University Data Science specialization. In this project, we investigate a dataset from the Human Activity Recognition lab containing data on how people perform Weight Lifting Exercises. Six different male participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). The participants wore sensors to track their movements-- on their arm, forearm, and waist-- and finally one was placed on the dumbbell itself.

We will use machine learning techniques to attempt to classify these exercises and predict further movements in the test dataset.

### Data Processing

In this section, we will explain how we have processed the data from the Human Activity Recognition lab. This includes downloading the data, reading it into R, and transforming it into a suitable form for further analysis.

First, we download and load the data:


```r
if(!file.exists("pml-training.csv")){
    
    download = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(download,"pml-training.csv")
}
if(!file.exists("pml-testing.csv")){
    
    download = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(download,"pml-testing.csv")
}

train.data = read.csv("pml-training.csv")
test.data = read.csv("pml-testing.csv")
```

Once loaded, a quick examination of the data reveals that we have 19622 observed events in the train.data set, and 160 variables describing each. The test.data dataset has 20 observations to be correctly classified and the same 160 variables, except that the classe variable identifying the class in the train.data data has been replaced by a problem_id variable.

The first thing we would like to do is to reduce the dimensionality of the data. In particular, the test.data dataset contains a number of variables that are identically NA for all rows, so we discard this data from both the test.data and train.data sets:


```r
keep.cols = colSums(is.na(test.data))==0
test.data = subset(test.data, select = keep.cols)
train.data = subset(train.data, select = keep.cols)
```

We will also throw out the time, window, and X variables.


```r
keep.cols = grep("time|window|X", colnames(test.data),invert = TRUE)
test.data = subset(test.data, select = keep.cols)
train.data = subset(train.data, select = keep.cols)
```


This allows us to reduce the number of variables to just 54.

The strategy we will take for developing a model with the training data is that we will create a partition of 25% of the training data that will be used for out-of-sample error estimation and validation. With the remaining 75%, we will develop the best possible model for the data. Note that in the following code "training" and "testing" refer to partitions of the train.data dataset, and in particular that "testing" is not test.data.


```r
library(caret)
set.seed(1)
inTrain = createDataPartition(train.data$classe, p = .75)[[1]]
training = train.data[ inTrain,]
testing = train.data[-inTrain,]
```

### Random Forest Fit

We will use the random forest algorithm for our model fit. We increased the number of trees to achieve a slightly greater accuracy. I experimented with linear discriminant analysis and gradient boosting methods, but the accuracy achieved by the default parameterization of the randomForest() method was far superior.


```r
library(randomForest)
set.seed(2)
trees = 2000
RFfit = randomForest(classe ~ ., data = training, ntree= trees)
RFfit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, ntree = trees) 
##                Type of random forest: classification
##                      Number of trees: 2000
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.43%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4181    4    0    0    0 0.0009557945
## B    9 2836    3    0    0 0.0042134831
## C    0    8 2556    3    0 0.0042851578
## D    0    0   28 2382    2 0.0124378109
## E    0    0    1    5 2700 0.0022172949
```

Here, we can see that the out-of-bag estimate of the error rate is 0.0042805. From the RandomForest documentation:

>In random forests, there is no need for cross-validation or a separate test set to get an unbiased estimate of the test set error. It is estimated internally, during the run, as follows:

>Each tree is constructed using a different bootstrap sample from the original data. About one-third of the cases are left out of the bootstrap sample and not used in the construction of the kth tree.

>Put each case left out in the construction of the kth tree down the kth tree to get a classification. In this way, a test set classification is obtained for each case in about one-third of the trees. At the end of the run, take j to be the class that got most of the votes every time case n was oob. The proportion of times that j is not equal to the true class of n averaged over all cases is the oob error estimate. This has proven to be unbiased in many tests.

Hence, this out-of-bag estimate serves as sufficient cross validation. However, we would also like to validate the model fit on an out-of-sample estimate, so we leverage the reserved testing data as follows:



```r
predRF = predict(RFfit,testing)
confusionMatrix(testing$classe,predRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    1    0    0    1
##          B    3  945    1    0    0
##          C    0    3  850    2    0
##          D    0    0    5  798    1
##          E    0    0    0    2  899
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9961         
##                  95% CI : (0.994, 0.9977)
##     No Information Rate : 0.2847         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9951         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9979   0.9958   0.9930   0.9950   0.9978
## Specificity            0.9994   0.9990   0.9988   0.9985   0.9995
## Pos Pred Value         0.9986   0.9958   0.9942   0.9925   0.9978
## Neg Pred Value         0.9991   0.9990   0.9985   0.9990   0.9995
## Prevalence             0.2847   0.1935   0.1746   0.1635   0.1837
## Detection Rate         0.2841   0.1927   0.1733   0.1627   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9986   0.9974   0.9959   0.9968   0.9986
```

Here, the estimate of the out-of-sample accuracy is **0.9961**, with a 95% confidence interval of (0.994, 0.9977). With the high level of accuracy, we can feel fairly confident about the results of applying this model to the test.data. Indeed, the predictions from the fit score 20/20 on the test.data.



```r
predAnswers = predict(RFfit,test.data)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predAnswers)
```
