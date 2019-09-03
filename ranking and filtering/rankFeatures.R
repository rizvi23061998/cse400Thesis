library(randomForest)
library(here)
timestamp();

set.seed(20);

fScheme = c("_nGrams","_nGDip","_PSF");
here();
for(i in (1:3)){

fileNameSuffix = paste(fScheme[i], ".rds", sep = "");

rfmodelFile        = paste("out/rfmodel"   , fileNameSuffix, sep = "");
rankedFeaturesFile = paste("out/ff"        , fileNameSuffix, sep = "");
featureFile        = paste("out/featurized", fileNameSuffix, sep = "");


if (!file.exists(rankedFeaturesFile)) {
  cat(as.character(Sys.time()),">> Loading feature file ...\n");
  features = readRDS(featureFile);
  cat(as.character(Sys.time()),">> Done ( from cached file:", featureFile, ")\n");
 
  # features$ID = NULL;
  # features$Type = NULL;
  features$Name = NULL;
  cat(as.character(Sys.time()),">> Total features: ", length(features[1,]) - 1, "\n");
  
  cat(as.character(Sys.time()),">> Computing random forest ...\n");
  if (!file.exists(rfmodelFile)) {
    rfmodel = randomForest(protection ~ ., features, importance=TRUE);
    saveRDS(rfmodel, rfmodelFile);
    cat(as.character(Sys.time()),">> Done.\n");
  } else {
    rfmodel = readRDS(rfmodelFile);
    cat(as.character(Sys.time()),">> Done ( from cached file:", rfmodelFile, ")\n");
  }
  
  cat(as.character(Sys.time()),">> Computing feature ranking ...\n");
  allRank = rfmodel$importance[order(-rfmodel$importance[,3]),];
  print(length(allRank))
  rankedFeatures = rownames(allRank[which(allRank[,3]>0),]);
  # rankedFeatures = rownames(allRank);
  saveRDS(rankedFeatures, rankedFeaturesFile);
  cat(as.character(Sys.time()),">> Done\n");
  
} else {
  cat(as.character(Sys.time()),">> Computing feature ranking ...\n");
  rankedFeatures = readRDS(rankedFeaturesFile);
  cat(as.character(Sys.time()),">> Done ( from cached file:", rankedFeaturesFile, ")\n");
}
}