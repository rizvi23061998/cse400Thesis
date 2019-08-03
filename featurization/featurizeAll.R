library(e1071)
library(ROCR)
library(randomForest)
library(readxl)

source(here('featurization','./featurization.R'));

timestamp();

dataFolder = "Sequences/"

seqFile = paste(dataFolder,"pnas_sd02.xlsx",sep="");
infoFile = paste(dataFolder,"pnas_sd01.xlsx",sep="");

info = read_excel(infoFile);
seq = read_excel(seqFile);

dataset = merge(info,seq,by="Name");
dataset = subset(dataset,select=c("Name","Clinical Status","VH","VL"));
names(dataset)[names(dataset) == "Clinical Status"]<-"protection";
dataset$Sequence = paste(dataset$VH,dataset$VL,sep="");
dataset$protection[dataset$protection == "Approved"]<- "1";
dataset$protection[dataset$protection != "1"]<- "0";

# nonAntigensFile = "nonAntigens.csv";
# featureFilePrefix = "featurized_1324";

# antigensFile = "Bartonella_Antigen.csv";
# nonAntigensFile = "Bartonella_NonAntigen.csv";
# featureFilePrefix = "testFeaturized";
for (i in (1:3)){
  print(i);
  fScheme = c("_nGrams","_nGDip","_PSF");
  
  outFolder = "out/"
  featureFile = paste(outFolder,"featurized",fScheme[i],".rds",sep = "")
  ngramVal = c(3,0,0);
  ngdipVal = c(0,10,0);
  psfVal = c(0,0,25);
  # featureFile = paste(featureFilePrefix, fScheme, ".rds", sep = "");
  
  amins = c("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y");
  
  cat(as.character(Sys.time()),">> Featurizing ...\n");
  if (!file.exists(featureFile)) {
    # cat(as.character(Sys.time()),">> Reading antigens file (", antigensFile, ") ...\n");
    # # antigens = read.csv(antigensFile);
    # # antigens$protection = 1;
    # cat(as.character(Sys.time()),">> Done\n");
    # 
    # cat(as.character(Sys.time()),">> Reading non-antigens file (", nonAntigensFile, ") ...\n");
    # # nonAntigens = read.csv(nonAntigensFile);
    # # nonAntigens$protection = 0;
    # cat(as.character(Sys.time()),">> Done\n");
    
    # data = rbind(antigens, nonAntigens);
    data = dataset;
    nData = length(data[,1]);
    
    features = featurization(data$Sequence, data$protection, amins, nGramOrder = ngramVal[i], nGDipOrder = ngdipVal[i], psfOrder = psfVal[i]);
    features$Name = data$Name
    # features$Type = data$Type;
    saveRDS(features, featureFile);
    cat(as.character(Sys.time()),">> Featurizing Done.\n");
  } else {
    features = readRDS(featureFile);
    cat(as.character(Sys.time()),">> Done ( from cached file:", featureFile, ")\n");
  }
  cat(as.character(Sys.time()),">> Total features: ", length(features[1,]), "\n");
}