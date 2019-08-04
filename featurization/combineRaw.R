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
print("Applying smote");
set.seed(112);
# comb_data <- SMOTE( protection~., comb_data, perc.over = 280, k = 5, perc.under = 150)


split <- sample.split(comb_data$protection, SplitRatio = 0.75)

dresstrain <- subset(comb_data, split == TRUE)
dresstest <- subset(comb_data, split == FALSE)

dresstrain <- SMOTE( protection~., dresstrain, perc.over = 280, k = 5, perc.under = 150);
print(as.data.frame(table(dresstrain$protection)));
if(classifier == "rf"){
  if(!file.exists("out/rf_model_comb.rds")){
    model.forest = randomForest(protection ~., data=dresstrain )
    saveRDS(model.forest,"out/rf_model_comb.rds");
  }else{
    model.forest = readRDS("out/rf_model_comb.rds");
  }
  predicted <- predict(model.forest, dresstest)
  print(as.numeric(predicted)>=.5)
  print(confusionMatrix(data=predicted, reference=dresstest$protection))
  predicted <- predict(model.forest, dresstest,type = 'prob')
  print(as.numeric(predicted))
  
  ROCRpred2 <- prediction(predicted[,2], dresstest$protection)
  ROCRperf2 <- performance(ROCRpred2, 'tpr','fpr')
  plot(ROCRperf2, colorize = TRUE, text.adj = c(-0.2,1.7))
  
  fg <- predicted[dresstest$protection == 1,2]
  bg <- predicted[dresstest$protection == 0,2]
  # bg <- probs[df$label == 0]

  # ROC Curve
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  plot(roc)
  
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  plot(pr)
}else{
  model <- glm (protection~., data=dresstrain, family = binomial)
  summary(model)
  
}