require("e1071");
require("randomForest");

learn <-
  function(formula, data, learner, ...) {
    if (learner == "svm") {
      model = svm(formula, data, ...);
    } else if (learner == "rf") {
      model = randomForest(formula, data, ...);
    }
    else{
      model = naiveBayes(formula,data, ...);
    }
    
    return(model);
  }
