library(ROCR)
library(e1071)
library(randomForest)
library(DMwR)
library(dplyr)
library(caret)
library(caTools)
library(PRROC)
library(here)
library("SuperLearner")



source('raw/learnWithCV.R')

classifier = "svm"
type = "unbalanced"
output="out/comb/VH/"
output1="out/comb/VL/"
output_comb="out/comb/"

nFolds = 137;
#output="out/VL/"
fpsf = readRDS(paste(output,"featurized_PSF.rds",sep=""));
ngpsf = readRDS(paste(output,"featurized_nGrams.rds",sep=""));
fngdip = readRDS(paste(output,"featurized_nGDip.rds",sep=""));
fpsf1 = readRDS(paste(output1,"featurized_PSF.rds",sep=""));
ngpsf1 = readRDS(paste(output1,"featurized_nGrams.rds",sep=""));
fngdip1 = readRDS(paste(output1,"featurized_nGDip.rds",sep=""));

ngpsf$protection = NULL;
fngdip$protection = NULL;
fpsf1$protection = NULL;
ngpsf1$protection = NULL;
fngdip1$protection = NULL;

x <- merge(fpsf,ngpsf,by="Name");
x1<- merge(x,fngdip,by="Name");
x2<- merge(x1,fpsf1,by="Name");
x3<- merge(x2,ngpsf1,by="Name");
comb_data<- merge(x3,fngdip1,by="Name")
#comb_data <- merge(x,fngdip,by="Name"); 
x$Name = NULL;
print("Combined_Version Running....")
# print(comb_data$protection)
# print(comb_data$Name)
ngram_VH= readRDS(paste(output,"ff_nGrams.rds",sep=""));
ngdip_VH= readRDS(paste(output,"ff_nGDip.rds",sep=""));
psf_VH= readRDS(paste(output,"ff_PSF.rds",sep=""));
ngram_VL= readRDS(paste(output1,"ff_nGrams.rds",sep=""));
ngdip_VL= readRDS(paste(output1,"ff_nGDip.rds",sep=""));
psf_VL= readRDS(paste(output1,"ff_PSF.rds",sep=""));


features_important <- c(ngram_VH,ngdip_VH,psf_VH,ngram_VL,ngdip_VL,psf_VL);
features_important <- c(features_important,"Name","protection");
comb_data <- subset(comb_data, select = c(features_important));

saveRDS(comb_data,paste(output_comb,"comb_raw.rds",sep=""));
comb_data$Name = NULL;
print("Applying smote");
set.seed(112);
# comb_data <- SMOTE( protection~., comb_data, perc.over = 280, k = 5, perc.under = 150)

comb_data_1 <- comb_data[which(comb_data$protection == 1),]
comb_data_0 <- comb_data[which(comb_data$protection == 0),]
#creating balanced(not smote) train test split
split <- sample.split(comb_data_1$protection, SplitRatio = .8)

train1 <- subset(comb_data_1, split == TRUE)
test1 <- subset(comb_data_1, split == FALSE)

split <- sample.split(comb_data_0$protection, SplitRatio = .8)

train2 <- subset(comb_data_0, split == TRUE)
test2 <- subset(comb_data_0, split == FALSE)

print("splitted")
dresstrain <- rbind(train1,train2)
dresstest <- rbind(test1,test2)

split <- sample.split(dresstrain$protection, SplitRatio = .8)
validation_data <-subset(dresstrain,split==FALSE)
print("test train completed")

# dresstrain <- subset(comb_data, split == TRUE)
# dresstest <- subset(comb_data, split == FALSE)

if(type == "balanced"){
  print("Applying smote");
  # dresstrain <- SMOTE( protection~., dresstrain, perc.over = 280, k = 5, perc.under = 150);
}else{
  print("Not balancing")
}

print(as.data.frame(table(dresstrain$protection)));
if(classifier == "rf"){
  # nFolds =10;
  maxFeatureCount = 14000;
  perf = learnWithCV(protection ~ ., comb_data, cross = nFolds, "rf",type);
  
  rocCurvePoints = NULL;
  bestPerf = NULL;
  bestParams = NULL;
  accData = NULL;
  
  df = data.frame(
    x = unlist(perf$rocCurve@x.values), 
    y = unlist(perf$rocCurve@y.values), 
    Features = as.character(maxFeatureCount)
  );
  rocCurvePoints = rbind(rocCurvePoints, df);
  
  df = data.frame(
    x = unlist(perf$prCurve@x.values), 
    y = unlist(perf$prCurve@y.values), 
    Features = as.character(maxFeatureCount)
  );
  prCurvePoints = NULL;
  prCurvePoints = rbind(prCurvePoints, df);
  
  cat(
    maxFeatureCount,
    ",", round(perf$AUCROC, 2),
    ",", round(perf$AUCPR, 2),
    ",", round(perf$acc, 2),
    ",", round(perf$sens, 2),
    ",", round(perf$spec, 2),
    ",", round(perf$prec, 2),
    ",", round(perf$f1, 2),
    ",", round(perf$mcc, 2)
  );
  accData = rbind(
    accData, 
    c(
      maxFeatureCount 
      , perf$AUCROC
      , perf$AUCPR
      , perf$acc
      , perf$sens
      , perf$spec
      , perf$prec
      , perf$f1
      , perf$mcc
    )
  );
  colnames(accData) = c(
    "nF"
    , "AUCROC"
    , "AUCPR"
    , "Accuracy"
    , "Sensitivity"
    , "Specificity"
    , "Precision"
    , "F1"
    , "MCC"
  );
  write.csv(accData,paste(output_comb,classifier,"_acc.csv",sep=""));
  
  if (is.null(bestPerf) || bestPerf$mcc < perf$mcc) {
    bestPerf = perf;
    bestParams = list(
      "maxFeatureCount" = maxFeatureCount
    )
    cat(",<-- BEST");
  }
  
  cat("\n");
  
  saveRDS(rocCurvePoints,paste(output_comb,"rocData.rds",sep=""));
  saveRDS(prCurvePoints , paste(output_comb,"prData.rds",sep=""));
  
  cat("Best Result for nF = ", bestParams$maxFeatureCount, "\n");
  cat("AUCROC      : ", bestPerf$auc, "\n");
  cat("Threshold   : ", bestPerf$threshold, "\n");
  cat("Accuracy    : ", bestPerf$acc, "\n");
  cat("Sensitivity : ", bestPerf$sens, "\n");
  cat("Specificity : ", bestPerf$spec, "\n");
  cat("MCC         : ", bestPerf$mcc, "\n");
  cat("Precision   : ", bestPerf$prec,"\n");
  cat("F1          : ", bestPerf$f1,"\n")
  # if(!file.exists("out/rf_model_comb.rds")){
  #   model.forest = randomForest(protection ~., data=dresstrain )
  #   saveRDS(model.forest,"out/rf_model_comb.rds");
  # }else{
  #   model.forest = readRDS("out/rf_model_comb.rds");
  # }
  # predicted <- predict(model.forest, dresstest)
  # print(as.numeric(predicted))
  # print(confusionMatrix(data=predicted, reference=dresstest$protection))
  # predicted <- predict(model.forest, dresstest,type = 'prob')
  # print(as.numeric(predicted))
  # 
  # ROCRpred2 <- prediction(predicted[,2], dresstest$protection)
  # ROCRperf2 <- performance(ROCRpred2, 'tpr','fpr')
  # plot(ROCRperf2, colorize = TRUE, text.adj = c(-0.2,1.7))
  # 
  # fg <- predicted[dresstest$protection == 1,2]
  # bg <- predicted[dresstest$protection == 0,2]
  # # bg <- probs[df$label == 0]
  # 
  # # ROC Curve
  # roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  # plot(roc)
  # 
  # pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  # plot(pr)
}else if(classifier == "svm"){
  # nFolds =10;
  maxFeatureCount = 14000;
  perf = learnWithCV(protection ~ ., comb_data, cross = nFolds, "svm",type);
  
  rocCurvePoints = NULL;
  bestPerf = NULL;
  bestParams = NULL;
  accData = NULL;
  
  df = data.frame(
    x = unlist(perf$rocCurve@x.values), 
    y = unlist(perf$rocCurve@y.values), 
    Features = as.character(maxFeatureCount)
  );
  rocCurvePoints = rbind(rocCurvePoints, df);
  
  df = data.frame(
    x = unlist(perf$prCurve@x.values), 
    y = unlist(perf$prCurve@y.values), 
    Features = as.character(maxFeatureCount)
  );
  prCurvePoints = NULL;
  prCurvePoints = rbind(prCurvePoints, df);
  
  cat(
    maxFeatureCount,
    ",", round(perf$AUCROC, 2),
    ",", round(perf$AUCPR, 2),
    ",", round(perf$acc, 2),
    ",", round(perf$sens, 2),
    ",", round(perf$spec, 2),
    ",", round(perf$prec, 2),
    ",", round(perf$f1, 2),
    ",", round(perf$mcc, 2)
  );
  accData = rbind(
    accData, 
    c(
      maxFeatureCount 
      , perf$AUCROC
      , perf$AUCPR
      , perf$acc
      , perf$sens
      , perf$spec
      , perf$prec
      , perf$f1
      , perf$mcc
    )
  );
  colnames(accData) = c(
    "nF"
    , "AUCROC"
    , "AUCPR"
    , "Accuracy"
    , "Sensitivity"
    , "Specificity"
    , "Precision"
    , "F1"
    , "MCC"
  );
  write.csv(accData,paste(output_comb,classifier,"_acc.csv",sep=""));
  
  if (is.null(bestPerf) || bestPerf$mcc < perf$mcc) {
    bestPerf = perf;
    bestParams = list(
      "maxFeatureCount" = maxFeatureCount
    )
    cat(",<-- BEST");
  }
  
  cat("\n");
  
  saveRDS(rocCurvePoints,paste(output_comb,"rocData.rds",sep=""));
  saveRDS(prCurvePoints ,paste(output_comb,"prData.rds",sep=""));
  
  cat("Best Result for nF = ", bestParams$maxFeatureCount, "\n");
  cat("AUCROC      : ", bestPerf$auc, "\n");
  cat("Threshold   : ", bestPerf$threshold, "\n");
  cat("Accuracy    : ", bestPerf$acc, "\n");
  cat("Sensitivity : ", bestPerf$sens, "\n");
  cat("Specificity : ", bestPerf$spec, "\n");
  cat("MCC         : ", bestPerf$mcc, "\n")
  cat("Precision   : ", bestPerf$prec,"\n");
  cat("F1          : ", bestPerf$f1,"\n")
  
  # if(!file.exists("out/rf_model_comb.rds")){
  #   model.forest = randomForest(protection ~., data=dresstrain )
  #   saveRDS(model.forest,"out/rf_model_comb.rds");
  # }else{
  #   model.forest = readRDS("out/rf_model_comb.rds");
  # }
  # predicted <- predict(model.forest, dresstest)
  # print(as.numeric(predicted))
  # print(confusionMatrix(data=predicted, reference=dresstest$protection))
  # predicted <- predict(model.forest, dresstest,type = 'prob')
  # print(as.numeric(predicted))
  # 
  # ROCRpred2 <- prediction(predicted[,2], dresstest$protection)
  # ROCRperf2 <- performance(ROCRpred2, 'tpr','fpr')
  # plot(ROCRperf2, colorize = TRUE, text.adj = c(-0.2,1.7))
  # 
  # fg <- predicted[dresstest$protection == 1,2]
  # bg <- predicted[dresstest$protection == 0,2]
  # # bg <- probs[df$label == 0]
  # 
  # # ROC Curve
  # roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  # plot(roc)
  # 
  # pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  # plot(pr)
}else{
  # nFolds =136;
  maxFeatureCount = 14000;
  perf = learnWithCV(protection ~ ., comb_data, cross = nFolds, "nb",type);
  
  rocCurvePoints = NULL;
  bestPerf = NULL;
  bestParams = NULL;
  accData = NULL;
  
  df = data.frame(
    x = unlist(perf$rocCurve@x.values), 
    y = unlist(perf$rocCurve@y.values), 
    Features = as.character(maxFeatureCount)
  );
  rocCurvePoints = rbind(rocCurvePoints, df);
  
  df = data.frame(
    x = unlist(perf$prCurve@x.values), 
    y = unlist(perf$prCurve@y.values), 
    Features = as.character(maxFeatureCount)
  );
  prCurvePoints = NULL;
  prCurvePoints = rbind(prCurvePoints, df);
  
  cat(
    maxFeatureCount,
    ",", round(perf$AUCROC, 2),
    ",", round(perf$AUCPR, 2),
    ",", round(perf$acc, 2),
    ",", round(perf$sens, 2),
    ",", round(perf$spec, 2),
    ",", round(perf$prec, 2),
    ",", round(perf$f1, 2),
    ",", round(perf$mcc, 2)
  );
  accData = rbind(
    accData, 
    c(
      maxFeatureCount 
      , perf$AUCROC
      , perf$AUCPR
      , perf$acc
      , perf$sens
      , perf$spec
      , perf$prec
      , perf$f1
      , perf$mcc
    )
  );
  colnames(accData) = c(
    "nF"
    , "AUCROC"
    , "AUCPR"
    , "Accuracy"
    , "Sensitivity"
    , "Specificity"
    , "Precision"
    , "F1"
    , "MCC"
  );
  write.csv(accData,paste(output_comb,classifier,"_acc.csv",sep=""));
  
  if (is.null(bestPerf) || bestPerf$mcc < perf$mcc) {
    bestPerf = perf;
    bestParams = list(
      "maxFeatureCount" = maxFeatureCount
    )
    cat(",<-- BEST");
  }
  
  cat("\n");
  
  saveRDS(rocCurvePoints, paste(output_comb,"rocData.rds"));
  saveRDS(prCurvePoints , paste(output_comb,"prData.rds"));
  
  cat("Best Result for nF = ", bestParams$maxFeatureCount, "\n");
  cat("AUCROC      : ", bestPerf$auc, "\n");
  cat("Threshold   : ", bestPerf$threshold, "\n");
  cat("Accuracy    : ", bestPerf$acc, "\n");
  cat("Sensitivity : ", bestPerf$sens, "\n");
  cat("Specificity : ", bestPerf$spec, "\n");
  cat("MCC         : ", bestPerf$mcc, "\n")
  cat("Precision   : ", bestPerf$prec,"\n");
  cat("F1          : ", bestPerf$f1,"\n")
  
  
}
  

