library(ROCR)
library(e1071)
library(randomForest)
library(DMwR)
library(dplyr)
library(caret)
library(caTools)
library(PRROC)



rerun = 1;
seed = 120; 
#seed=120,(4,15,25),seed_rankFeatures=20,accuracy=.7714(random forest,unbalanced),concat(VH,VL)
#seed=120,Accuracy : 0.8(nb,unbalanced),.6286(balanced)
#seed=120,Accuracy : 0.6857 (svm,unblanced)         

output="out/VH/"
output1="out/VL/"
output_comb="out/comb/"

classifier = "rf"
type = "unbalanced"
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

# print(comb_data$protection)
# print(comb_data$Name)
ngram_VH= readRDS(paste(output,"ff_nGrams.rds",sep=""));
ngdip_VH= readRDS(paste(output,"ff_nGDip.rds",sep=""));
psf_VH= readRDS(paste(output,"ff_PSF.rds",sep=""));
ngram_VL= readRDS(paste(output1,"ff_nGrams.rds",sep=""));
ngdip_VL= readRDS(paste(output1,"ff_nGDip.rds",sep=""));
psf_VL= readRDS(paste(output1,"ff_PSF.rds",sep=""));

features_important <- c(ngram_VH,ngdip_VH,psf_VH,ngram_VL,ngdip_VL,psf_VL);
# features_important <- c(readRDS("out/ff_SvmRFE_nGrams.rds"),readRDS("out/ff_SvmRFE_nGDip.rds"),readRDS("out/ff_SvmRFE_PSF.rds"));
# features_important <- c(features_important,"Name","protection");
features_important <- c(features_important,"Name","protection");
# features_important <- c(readRDS("out/ff_SvmRFE2_PSF.rds"),"Name","protection");
comb_data <- subset(comb_data, select = c(features_important))

saveRDS(comb_data,paste(output_comb,"comb_raw.rds",sep=""));
# fngdip$protection <- comb_data$protection;
# comb_data <- fpsf;
comb_data$Name = NULL;

set.seed(seed);



comb_data_1 <- comb_data[which(comb_data$protection == 1),]
comb_data_0 <- comb_data[which(comb_data$protection == 0),]
#creating balanced(not smote) train test split
split <- sample.split(comb_data_1$protection, SplitRatio = 0.75)

train1 <- subset(comb_data_1, split == TRUE)
test1 <- subset(comb_data_1, split == FALSE)

split <- sample.split(comb_data_0$protection, SplitRatio = 0.75)

train2 <- subset(comb_data_0, split == TRUE)
test2 <- subset(comb_data_0, split == FALSE)

print("splitted")
dresstrain <- rbind(train1,train2)
dresstest <- rbind(test1,test2)

split <- sample.split(dresstrain$protection, SplitRatio = 0.75)
validation_data <-subset(dresstrain,split==FALSE)
print("test train completed")

# dresstrain <- subset(comb_data, split == TRUE)
# dresstest <- subset(comb_data, split == FALSE)

if(type == "balanced"){
  print("Applying smote");
  dresstrain <- SMOTE( protection~., dresstrain, perc.over = 280, k = 5, perc.under = 150);
}else{
  print("Not balancing")
}

print(as.data.frame(table(dresstrain$protection)));
if(classifier == "rf"){
  print("rf model is training")
  # if(!file.exists("out/rf_model_comb.rds")){
    model.forest = randomForest(protection ~., data=dresstrain )
    saveRDS(model.forest,paste(output_comb,"rf_model_comb.rds",sep=""));
  
    # model.forest = readRDS("out/rf_model_comb.rds");
  
  predicted <- predict(model.forest, dresstest)
  # print(as.numeric(predicted))
  print(confusionMatrix(data=predicted, reference=dresstest$protection,positive = "1"))
  print(confusionMatrix(data=predicted, reference=dresstest$protection,mode="prec_recall",positive = "1"))
  # cat("MCC:",mcc(predicted,dresstest$protection))
  predicted <- predict(model.forest, dresstest,type = 'prob')

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
  
  predicted <- predict(model.forest, validation_data)
  # print(as.numeric(predicted))
  print(confusionMatrix(data=predicted, reference=validation_data$protection,positive = "1"))
  
}else if(classifier == "svm"){

  model <- svm(protection~.,data=dresstrain,type = 'C-classification', kernel = 'radial',cost=10^2,Gamma=100,scale = FALSE)
  print(summary(model))
  predict <- predict(model,dresstest)
  print(confusionMatrix(data=(predict),reference = dresstest$protection,positive = "1"))
  print(confusionMatrix(data=(predict),reference = dresstest$protection,mode = "prec_recall",positive = "1"))

  predicted <- predict(model, dresstest,decision.values=TRUE)
  predicted_prob <- attr(predicted,"decision.values")

  ROCRpred2 <- prediction(predicted_prob, dresstest$protection)
  ROCRperf2 <- performance(ROCRpred2, 'tpr','fpr')
  plot(ROCRperf2, colorize = TRUE, text.adj = c(-0.2,1.7))

  fg <- predicted_prob[dresstest$protection == 1]
  bg <- predicted_prob[dresstest$protection == 0]
  # bg <- probs[df$label == 0]

  # ROC Curve
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  plot(roc)

  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  plot(pr)
  predict <- predict(model,validation_data)
  print(confusionMatrix(data=(predict),reference = validation_data$protection,positive = "1"))
  

}else if(classifier == "nb"){
  model <- naiveBayes(protection~.,data=dresstrain);
  print(summary(model))

  predicted <- predict(model, dresstest)

  # print(table(predicted,dresstest$protection))
  print(confusionMatrix(data=predicted, reference=dresstest$protection))
  predicted <- predict(model, dresstest,type = 'raw')
  print(predicted)

  ROCRpred2 <- prediction(predicted[,2], dresstest$protection)
  ROCRperf2 <- performance(ROCRpred2, 'tpr','fpr')
  plot(ROCRperf2, colorize = TRUE, text.adj = c(-0.2,1.7))
  predicted <- predict(model, validation_data)
  
  # print(table(predicted,dresstest$protection))
  print(confusionMatrix(data=predicted, reference=validation_data$protection))
  

}else{
  model <- glm (protection~., data=dresstrain, family = binomial,maxit=100)
  print(summary(model))
  predict <- predict(model,dresstest)
  print(table( predict > 0.5,dresstest$protection))
  # confusionMatrix(data=predict,reference = dresstest$protection)
  predict <- predict(model,dresstest,type="response")
  ROCRpred <- prediction(predict, dresstest$protection)
  ROCRperf <- performance(ROCRpred, 'tpr','fpr')
  plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.1))


}

