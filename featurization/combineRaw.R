library(e1071)
library(randomForest)
library(DMwR)
library(dplyr)
library(caTools)

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
# comb_data <- SMOTE( protection~., comb_data, perc.over = 280, k = 5, perc.under = 150)


split <- sample.split(comb_data$protection, SplitRatio = 0.75)

dresstrain <- subset(comb_data, split == TRUE)
dresstest <- subset(comb_data, split == FALSE)

dresstrain <- SMOTE( protection~., dresstrain, perc.over = 280, k = 5, perc.under = 150);
print(as.data.frame(table(dresstrain$protection)));

model.forest = randomForest(protection ~., data=dresstrain )
