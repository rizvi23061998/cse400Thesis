require("e1071");
require("randomForest");
require("ROCR");
require("pracma");
library(DMwR)
library(dplyr)
library(caret)
library(caTools)
library(here)
library(mlr)
library(e1071)
library(kernlab)
library(ada)
# library(bimba)

configureMlr(on.par.without.desc = "quiet")
source("raw/learn.R");

learnWithCV <-
  function(formula, data, cross, learner,bType, ...) {
    trainType = "cs";
    # data <- data[sample(nrow(data)),]
    # N = length(data[, 1])
    # folds = seq(from=1,to=N, by=round(N/cross))
    # folds[cross+1] = N+1
    # data = data[folds[2]:N,]
    print(paste("Starting training with ",learner));
    N = length(data[, 1])
    folds = seq(from=1,to=N, by=round(N/cross))
    folds[cross+1] = N+1
    predVector = c()
    newpredvec = c()
    for (i in 1:cross) {
      trainFolds = data
      testFold = data[(folds[i]:(folds[i+1]-1)),]
      trainFolds = data[-(folds[i]:(folds[i+1]-1)),]
      
      if(bType == "balanced"){
        trainFolds <- SMOTE( protection~., trainFolds, perc.over = 200, k = 5, perc.under = 180);
        # trainFolds <- ADASYN(trainFolds,perc_over = 90,k=10);
        print(as.data.frame(table(trainFolds$protection)));
        
      }
      if(mod(i,3) == 0){
        print(paste("model ",i,"is starting training..."))
      }
      
      if(trainType == "cs"){
        trainFolds.task = makeClassifTask(data = trainFolds,target = "protection");
        costs = matrix(c(0, 1, 5, 0), 2);
        colnames(costs) = rownames(costs) = c("1","0");
        # print(costs);
        
        #lrn = makeLearner("classif.cforest", trace = FALSE,predict.type = "prob")
        #lrn = makeWeightedClassesWrapper(lrn, wcw.weight = .5 )
        
        lrn = makeLearner("classif.randomForest", class.weights = c("0" = 55, "1" = 89),predict.type = "prob")
        model = train(lrn,trainFolds.task);
        testFold.task = makeClassifTask(data = testFold,target = "protection");
        mlPred = predict(model, testFold.task);
        mlPred.th = setThreshold(mlPred,0.50);
        print(mlPred.th$data$response)
        prf = performance(mlPred.th)
        print(prf)
        
      }
      else{
        model = learn(formula, trainFolds, learner, ...);
      }
      if(mod(i,3) == 0){
        print(paste("model ",i,"is finished training."))     
      }
      
      # mlPred = predict(model, testFold);
      # print(confusionMatrix(data=mlPred, reference=testFold$protection))
      
      # newpredvec = c(newpredvec,mlPred)
      # print(newpredvec)
      # print("mlprd")
      
      # print(mlPred)
      predVector = c(predVector, as.numeric(mlPred.th$data$response))
      # print(predVector)
      i = i + 1
    }
    
    # Now generate the model on full dataset to find the no. of support vectors
    # This can be used for model selection in. Lesser number of SVs will
    # result in better generalization
    model = learn(formula, data, learner, ...);
    
    dependentVar = all.vars(formula)[1];
    
    # perform classification based perf. measures
    if (is.factor(data[,dependentVar])) {
      print("classification based predictions");
      mlPrediction = ROCR::prediction(as.numeric(predVector), as.numeric(data[,dependentVar]))
      
      # newmlpred <- prediction(newpredvec,data$protection)
      # ag$taac
      # Find the ROC curve and AUCROC
      AUCROC  = ROCR::performance(mlPrediction,"auc")@y.values[[1]];
      rocCurve = ROCR::performance(mlPrediction,"tpr", "fpr");
      plot(rocCurve)
      # Find the PR curve and AUCPR
      prCurve  = ROCR::performance(mlPrediction,"prec", "rec");
      plot(prCurve)
      x = unlist(prCurve@x.values);
      y = unlist(prCurve@y.values);
      df = data.frame(x = x[2:length(x)], y = y[2:length(y)]);
      AUCPR  = trapz(df$x, df$y)
      
      
      
      acc = unlist(ROCR::performance(mlPrediction,"acc")@y.values)[2]
      sensitivity = unlist(ROCR::performance(mlPrediction,"sens")@y.values)[2];
      specificity = unlist(ROCR::performance(mlPrediction,"spec")@y.values)[2];
      precision   = unlist(ROCR::performance(mlPrediction,"prec")@y.values)[2];
      mcc = unlist(ROCR::performance(mlPrediction,"mat")@y.values)[2];
      f1  = unlist(ROCR::performance(mlPrediction,"f")@y.values)[2];
      
      return(list(
        "model"= model,
        "rocCurve"= rocCurve,
        "prCurve"= prCurve,
        "AUCROC"= AUCROC,
        "AUCPR" = AUCPR,
        "acc"  = acc,
        "sens" = sensitivity,
        "spec" = specificity,
        "mcc"  = mcc,
        "prec" = precision,
        "f1"   = f1
      ))
    }
    else {
      # perform regression based perf. measurements
      mlPrediction = prediction(predVector, data[, dependentVar]);
      
      print(confusionMatrix())
      # Find the ROC curve and AUCROC
      AUCROC  = ROCR::performance(mlPrediction,"auc")@y.values[[1]];
      rocCurve = ROCR::performance(mlPrediction,"tpr", "fpr");
      
      # Find the PR curve and AUCPR
      prCurve  = ROCR::performance(mlPrediction,"prec", "rec");
      x = unlist(prCurve@x.values);
      y = unlist(prCurve@y.values);
      df = data.frame(x = x[2:length(x)], y = y[2:length(y)]);
      AUCPR  = trapz(df$x, df$y)
      
      # Find optimal threshold based on accuracy
      # accSeries = ROCR::performance(mlPrediction,"acc");
      # threshold = unlist(accSeries@x.values)[[which.max(unlist(accSeries@y.values))]];
      
      # Use a fixed threshold of 0.5
      threshold = .5;
      
      mlPrediction = prediction(as.numeric(predVector >= threshold), data[, dependentVar]);
      
      acc = unlist(ROCR::performance(mlPrediction,"acc")@y.values)[2]
      sensitivity = unlist(ROCR::performance(mlPrediction,"sens")@y.values)[2];
      specificity = unlist(ROCR::performance(mlPrediction,"spec")@y.values)[2];
      precision   = unlist(ROCR::performance(mlPrediction,"prec")@y.values)[2];
      mcc = unlist(ROCR::performance(mlPrediction,"mat")@y.values)[2];
      f1  = unlist(ROCR::performance(mlPrediction,"f")@y.values)[2];
      
      return(list(
        "model"     = model,
        "threshold" = threshold,
        "rocCurve"  = rocCurve,
        "prCurve"   = prCurve,
        "AUCROC"    = AUCROC,
        "AUCPR"     = AUCPR,
        "acc"       = acc,
        "sens"      = sensitivity,
        "spec"      = specificity,
        "prec"      = precision,
        "f1"        = f1,
        "mcc"       = mcc
      ))
    }
  }