library(ROCR)
library(e1071)
library(randomForest)
library(DMwR)
library(dplyr)
library(caret)
library(caTools)
library(PRROC)

classifier = "rf"
fpsf = readRDS("out/featurized_PSF.rds");
ngpsf = readRDS("out/featurized_nGrams.rds");
fngdip = readRDS("out/featurized_nGDip.rds");
ngpsf$protection = NULL;
fngdip$protection = NULL;
x <- merge(fpsf,ngpsf,by="Name");
comb_data <- merge(x,fngdip,by="Name"); 
x$Name = NULL;

# print(comb_data$protection)
# print(comb_data$Name)

features_important <- c(readRDS("out/ff_nGrams.rds"),readRDS("out/ff_nGDip.rds"),readRDS("out/ff_PSF.rds"));
features_important <- c(features_important,"Name","protection");
comb_data <- subset(comb_data, select = c(features_important))

saveRDS(comb_data,"out/comb_raw.rds");
comb_data$Name = NULL;

nFolds =10;
maxFeatureCount = 14000;
perf = learnWithCV(protection ~ ., trainingSet, cross = nFolds, "rf");

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
write.csv(accData, outFile);

if (is.null(bestPerf) || bestPerf$mcc < perf$mcc) {
  bestPerf = perf;
  bestParams = list(
    "maxFeatureCount" = maxFeatureCount
  )
  cat(",<-- BEST");
}

cat("\n");

saveRDS(rocCurvePoints, "rocData.rds");
saveRDS(prCurvePoints , "prData.rds");

cat("Best Result for nF = ", bestParams$maxFeatureCount, "\n");
cat("AUCROC      : ", bestPerf$auc, "\n");
cat("Threshold   : ", bestPerf$threshold, "\n");
cat("Accuracy    : ", bestPerf$acc, "\n");
cat("Sensitivity : ", bestPerf$sens, "\n");
cat("Specificity : ", bestPerf$spec, "\n");
cat("MCC         : ", bestPerf$mcc, "\n")
