############################################################
# 07_model_training_and_evaluation.R
#
# Purpose:
# - Train multiple classification models to predict
#   Billboard Top 50 chart success
# - Evaluate model performance using confusion matrices
#   and ROC/AUC metrics
#
# Models included:
# - Ridge Logistic Regression (baseline linear model)
# - Support Vector Machine (RBF kernel)
# - Random Forest
# - XGBoost

############################################################


############################################################
# 8) Model training and evaluation
############################################################


# -------------------------------
# Ridge Logistic Regression
# -------------------------------
# Linear baseline model with L2 regularisation

set.seed(42)
cv_ridge <- cv.glmnet(
  X_train,
  y_train,
  family = "binomial",
  alpha = 0,
  nfolds = 5
)

ridge_prob <- as.numeric(
  predict(
    cv_ridge,
    newx = X_test,
    s = "lambda.min",
    type = "response"
  )
)

ridge_pred <- ifelse(ridge_prob > 0.5, 1, 0)

cat("\n--- Ridge Logistic ---\n")
print(
  confusionMatrix(
    factor(ridge_pred, levels = c(0, 1)),
    factor(y_test, levels = c(0, 1)),
    positive = "1"
  )
)

roc_ridge <- roc(y_test, ridge_prob)
cat("AUC:", auc(roc_ridge), "\n")


# -------------------------------
# Support Vector Machine (RBF)
# -------------------------------
# Non-linear classifier using a radial basis function kernel

X_train_dense <- as.matrix(X_train)
X_test_dense  <- as.matrix(X_test)

svm_model <- svm(
  x = X_train_dense,
  y = factor(y_train),
  kernel = "radial",
  probability = TRUE
)

svm_pred <- predict(svm_model, X_test_dense, probability = TRUE)
svm_prob <- attr(svm_pred, "probabilities")[, "1"]

cat("\n--- SVM (RBF) ---\n")
print(
  confusionMatrix(
    svm_pred,
    factor(y_test),
    positive = "1"
  )
)

roc_svm <- roc(y_test, as.numeric(svm_prob))
cat("AUC:", auc(roc_svm), "\n")


# -------------------------------
# Random Forest
# -------------------------------
# Ensemble tree-based model capturing non-linear interactions

set.seed(42)
rf_model <- randomForest(
  x = X_train_dense,
  y = factor(y_train),
  ntree = 500,
  importance = TRUE
)

rf_pred <- predict(rf_model, X_test_dense)
rf_prob <- predict(rf_model, X_test_dense, type = "prob")[, "1"]

cat("\n--- Random Forest ---\n")
print(
  confusionMatrix(
    rf_pred,
    factor(y_test),
    positive = "1"
  )
)

roc_rf <- roc(y_test, rf_prob)
cat("AUC:", auc(roc_rf), "\n")


# -------------------------------
# XGBoost
# -------------------------------
# Gradient boosting model optimising AUC

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 300,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 0
)

xgb_prob <- predict(xgb_model, dtest)
xgb_pred <- ifelse(xgb_prob > 0.5, 1, 0)

cat("\n--- XGBoost ---\n")
print(
  confusionMatrix(
    factor(xgb_pred, levels = c(0, 1)),
    factor(y_test, levels = c(0, 1)),
    positive = "1"
  )
)

roc_xgb <- roc(y_test, xgb_prob)
cat("AUC:", auc(roc_xgb), "\n")
