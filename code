############################################################
# IJC437 – Introduction to Data Science and IJC445 Data Visualisation

# Project: Predicting Billboard Top 50 Chart Success
# Dataset: Billboard Hot 100 (2000–2023)
#
# NOTE:
# This script implements the full data science pipeline:
# data loading → text processing → feature engineering →
# model training → evaluation → visualisation.
# No code logic is modified; comments are added for clarity.
############################################################


############################################################
# 1) Package installation and loading
############################################################

# Install required packages (run once if not installed)

install.packages(c(
  "tidyverse","caret","tidytext","SnowballC",
  "Matrix","glmnet","e1071","randomForest","xgboost","pROC"
))

# Load libraries used throughout the analysis
library(tidyverse)
library(caret)
library(tidytext)
library(SnowballC)
library(Matrix)
library(glmnet)
library(e1071)
library(randomForest)
library(xgboost)
library(pROC)

############################################################
# 2) Data loading
############################################################

# Download dataset from Kaggle using Kaggle CLI
# (Requires Kaggle API credentials configured locally)
system('C:\\Users\\ahmad\\AppData\\Local\\Python\\PythonCore-3.14-64\\Scripts\\kaggle.exe datasets download -d suparnabiswas/billboard-hot-1002000-2023-data-with-features')

# Unzip the downloaded dataset
unzip("billboard-hot-1002000-2023-data-with-features.zip", exdir = "billboard_data")

# Read dataset into R
df <- read_csv("billboard_data/billboard_24years_lyrics_spotify.csv")

# Inspect dataset structure and contents
glimpse(df)
view(df)




############################################################
# 3) Data cleaning and target variable creation
############################################################

# Select relevant columns and construct binary target variable
# top50 = 1 → song ranked in Top 50
# top50 = 0 → song ranked outside Top 50

df <- df %>%
  select(ranking, song, band_singer, lyrics, year) %>%
  drop_na(ranking, lyrics, band_singer, year) %>%
  mutate(
    song = replace_na(song, ""),
    top50 = if_else(ranking <= 50, 1L, 0L)
  )




############################################################
# 4) Train–test split (stratified sampling)
############################################################

# Stratified split preserves Top 50 / Not Top 50 ratio


set.seed(42)
train_idx <- createDataPartition(df$top50, p = 0.8, list = FALSE)
df_train <- df[train_idx, ]
df_test  <- df[-train_idx, ]


# Target variables
y_train <- df_train$top50
y_test  <- df_test$top50





############################################################
# 5) Text cleaning and preparation
############################################################

# Text cleaning function:
# - lowercase
# - remove non-alphabetic characters
# - remove extra whitespace

clean_text <- function(x) {
  x %>%
    stringr::str_to_lower() %>%
    stringr::str_replace_all("[^a-z\\s]", " ") %>%
    stringr::str_replace_all("\\s+", " ") %>%
    stringr::str_trim()
}



# Combine cleaned song title and lyrics into a single text field
df_train <- df_train %>%
  mutate(text = paste(clean_text(song), clean_text(lyrics)))



df_test <- df_test %>%
  mutate(text = paste(clean_text(song), clean_text(lyrics)))



############################################################
# 6) TF-IDF feature engineering (training data only)
############################################################

# Tokenise training text and remove stopwords
# Vocabulary is built ONLY on training data to avoid leakage
train_tokens <- df_train %>%
  mutate(doc_id = row_number()) %>%
  select(doc_id, text) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2) %>%
  mutate(word = wordStem(word)) %>%
  count(doc_id, word, sort = FALSE)




# Compute TF-IDF scores
train_tfidf <- train_tokens %>%
  bind_tf_idf(word, doc_id, n)


# Select top 2000 most informative words
top_words <- train_tfidf %>%
  group_by(word) %>%
  summarise(score = sum(tf_idf), .groups = "drop") %>%
  arrange(desc(score)) %>%
  slice_head(n = 2000) %>%
  pull(word)


############################################################
# Helper function to build TF-IDF sparse matrices
############################################################

# Ensures:
# - same vocabulary
# - same column order
# - missing terms filled with zeros
build_tfidf_sparse <- function(df_part, vocab) {
  tok <- df_part %>%
    mutate(doc_id = row_number()) %>%
    select(doc_id, text) %>%
    unnest_tokens(word, text) %>%
    anti_join(stop_words, by = "word") %>%
    filter(nchar(word) > 2) %>%
    mutate(word = wordStem(word)) %>%
    filter(word %in% vocab) %>%
    count(doc_id, word, sort = FALSE)
  
  tfidf <- tok %>%
    bind_tf_idf(word, doc_id, n)
  

  X <- tfidf %>%
    select(doc_id, word, tf_idf) %>%
    tidytext::cast_sparse(doc_id, word, tf_idf)
  
  # Add missing vocabulary columns if needed
  missing <- setdiff(vocab, colnames(X))
  if (length(missing) > 0) {
    zero_mat <- Matrix(0, nrow = nrow(X), ncol = length(missing), sparse = TRUE)
    colnames(zero_mat) <- missing
    X <- cbind(X, zero_mat)
  }
  
  # Reorder columns to match training vocabulary
  X <- X[, vocab, drop = FALSE]  # reorder exactly
  
  return(X)
}



# Build TF-IDF matrices for train and test sets
X_train_tfidf <- build_tfidf_sparse(df_train, top_words)
X_test_tfidf  <- build_tfidf_sparse(df_test,  top_words)

stopifnot(nrow(X_train_tfidf) == nrow(df_train))
stopifnot(nrow(X_test_tfidf)  == nrow(df_test))
stopifnot(ncol(X_train_tfidf) == ncol(X_test_tfidf))



############################################################
# 7) Metadata features (artist frequency and year)
############################################################

# Artist frequency is computed from training data only

artist_freq <- df_train %>%
  count(band_singer, name = "artist_freq")

# Prepare metadata features
meta_train <- df_train %>%
  left_join(artist_freq, by = "band_singer") %>%
  mutate(artist_freq = replace_na(artist_freq, 1)) %>%
  transmute(year = as.numeric(year), artist_freq = as.numeric(artist_freq))

meta_test <- df_test %>%
  left_join(artist_freq, by = "band_singer") %>%
  mutate(artist_freq = replace_na(artist_freq, 1)) %>%
  transmute(year = as.numeric(year), artist_freq = as.numeric(artist_freq))

# Standardise metadata using training statistics
meta_means <- colMeans(meta_train)
meta_sds   <- apply(meta_train, 2, sd)
meta_sds[meta_sds == 0] <- 1  # safety

meta_train_sc <- scale(meta_train, center = meta_means, scale = meta_sds)
meta_test_sc  <- scale(meta_test,  center = meta_means, scale = meta_sds)

# Combine TF-IDF and metadata into final feature matrices
X_train <- cbind(X_train_tfidf, Matrix(meta_train_sc, sparse = TRUE))
X_test  <- cbind(X_test_tfidf,  Matrix(meta_test_sc,  sparse = TRUE))

stopifnot(ncol(X_train) == ncol(X_test))




############################################################
# 8) Model training and evaluation
############################################################


# Ridge Logistic Regression (linear baseline)
set.seed(42)
cv_ridge <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 0, nfolds = 5)

ridge_prob <- as.numeric(predict(cv_ridge, newx = X_test, s = "lambda.min", type = "response"))
ridge_pred <- ifelse(ridge_prob > 0.5, 1, 0)

cat("\n--- Ridge Logistic ---\n")
print(confusionMatrix(factor(ridge_pred, levels = c(0,1)),
                      factor(y_test, levels = c(0,1)),
                      positive = "1"))
roc_ridge <- roc(y_test, ridge_prob)
cat("AUC:", auc(roc_ridge), "\n")


# Support Vector Machine (RBF kernel)
X_train_dense <- as.matrix(X_train)
X_test_dense  <- as.matrix(X_test)

svm_model <- svm(x = X_train_dense, y = factor(y_train),
                 kernel = "radial", probability = TRUE)

svm_pred <- predict(svm_model, X_test_dense, probability = TRUE)
svm_prob <- attr(svm_pred, "probabilities")[, "1"]

cat("\n--- SVM (RBF) ---\n")
print(confusionMatrix(svm_pred, factor(y_test), positive = "1"))
roc_svm <- roc(y_test, as.numeric(svm_prob))
cat("AUC:", auc(roc_svm), "\n")



# Random Forest
set.seed(42)
rf_model <- randomForest(x = X_train_dense, y = factor(y_train),
                         ntree = 500, importance = TRUE)

rf_pred <- predict(rf_model, X_test_dense)
rf_prob <- predict(rf_model, X_test_dense, type = "prob")[, "1"]

cat("\n--- Random Forest ---\n")
print(confusionMatrix(rf_pred, factor(y_test), positive = "1"))
roc_rf <- roc(y_test, rf_prob)
cat("AUC:", auc(roc_rf), "\n")




# XGBoost
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
print(confusionMatrix(factor(xgb_pred, levels = c(0,1)),
                      factor(y_test, levels = c(0,1)),
                      positive = "1"))

roc_xgb <- roc(y_test, xgb_prob)
cat("AUC:", auc(roc_xgb), "\n")





############################################################
# 9) VISUALISATIONS – IJC437 (Model evaluation)
############################################################

# DS1 – ROC Curve Comparison
# Compares discriminative ability of all models
plot(roc_ridge, col = "#DD8452", lwd = 2,
     main = "ROC Curves – Model Comparison",
     legacy.axes = TRUE)

plot(roc_svm, col = "#55A868", lwd = 2, add = TRUE)
plot(roc_rf, col = "#4C72B0", lwd = 2, add = TRUE)
plot(roc_xgb, col = "#C44E52", lwd = 2, add = TRUE)

legend("bottomright",
       legend = c(
         paste0("Ridge (AUC=", round(auc(roc_ridge),2),")"),
         paste0("SVM (AUC=", round(auc(roc_svm),2),")"),
         paste0("RF (AUC=", round(auc(roc_rf),2),")"),
         paste0("XGB (AUC=", round(auc(roc_xgb),2),")")
       ),
       col = c("#DD8452","#55A868","#4C72B0","#C44E52"),
       lwd = 2)






# DS2 – AUC Comparison Bar Plot
# Summarises model performance using a single metric

auc_df <- tibble(
  Model = c("Ridge", "SVM", "Random Forest", "XGBoost"),
  AUC = c(
    as.numeric(auc(roc_ridge)),
    as.numeric(auc(roc_svm)),
    as.numeric(auc(roc_rf)),
    as.numeric(auc(roc_xgb))
  )
)

ggplot(auc_df, aes(x = Model, y = AUC, fill = Model)) +
  geom_col(alpha = 0.8, width = 0.6) +
  coord_cartesian(ylim = c(0.5, 1)) +
  labs(
    title = "AUC Comparison Across Models",
    subtitle = "Performance comparison on test data",
    y = "Area Under ROC Curve"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")



# DS3 – Accuracy vs Balanced Accuracy
# Shows impact of class imbalance on evaluation metrics
metric_df <- tibble(
  Model = c("Ridge","SVM","RF","XGB"),
  Accuracy = c(0.599, 0.642, 0.664, 0.670),
  Balanced_Accuracy = c(0.597, 0.643, 0.664, 0.670)
) %>%
  pivot_longer(-Model, names_to = "Metric", values_to = "Value")

ggplot(metric_df, aes(Model, Value, fill = Metric)) +
  geom_col(position = "dodge") +
  labs(
    title = "Accuracy vs Balanced Accuracy",
    y = "Score"
  ) +
  theme_minimal()




# DS4 – Feature Importance (Random Forest)
# Provides model interpretability
varImpPlot(rf_model,
           main = "Random Forest Variable Importance",
           n.var = 10)




# DS5 – Model Complexity vs Performance
# Illustrates trade-off between complexity and AUC

complexity_df <- tibble(
  Model = c("Ridge","SVM","Random Forest","XGBoost"),
  Complexity = c(1,2,3,4),
  AUC = c(auc(roc_ridge), auc(roc_svm), auc(roc_rf), auc(roc_xgb))
)

ggplot(complexity_df, aes(Complexity, AUC, label = Model)) +
  geom_point(size = 4, color = "#4C72B0") +
  geom_text(vjust = -1) +
  scale_x_continuous(breaks = 1:4,
                     labels = c("Linear","Kernel","Ensemble","Boosted")) +
  labs(
    title = "Model Complexity vs Predictive Performance",
    x = "Model Complexity",
    y = "AUC"
  ) +
  theme_minimal()




############################################################
# 10) VISUALISATIONS – IJC445 (Interpretability & uncertainty)
############################################################

# DV1 – TF-IDF Signal Distribution
# Compares lexical signal strength between classes

ggplot(df_plot, aes(x = tfidf_sum, fill = chart_class)) +
  geom_density(alpha = 0.45, color = "grey20", linewidth = 0.6, adjust = 1.1) +
  
  geom_vline(
    data = df_plot %>% 
      group_by(chart_class) %>% 
      summarise(med = median(tfidf_sum)),
    aes(xintercept = med, color = chart_class),
    linetype = "dashed",
    linewidth = 0.8,
    show.legend = FALSE
  ) +
  
  scale_fill_manual(
    values = c(
      "Not Top 50" = "#E76F51",
      "Top 50"     = "#2A9D8F"
    )
  ) +
  
  scale_color_manual(
    values = c(
      "Not Top 50" = "#E76F51",
      "Top 50"     = "#2A9D8F"
    )
  ) +
  
  labs(
    title = "Overall TF-IDF Signal by Chart Class",
    x = "Sum of TF-IDF Weights",
    y = "Density",
    fill = "Chart Class"
  ) +
  
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    axis.title = element_text(face = "bold"),
    legend.title = element_text(face = "bold"),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )













# DV2 – Random Forest Feature Importance (Top 10)
# Highlights most influential features
ggplot(rf_imp_df,
       aes(x = reorder(Feature, MeanDecreaseGini),
           y = MeanDecreaseGini)) +
  
  geom_col(
    fill = "#2A6F97",
    width = 0.75
  ) +
  
  geom_text(
    aes(label = round(MeanDecreaseGini, 2)),
    hjust = -0.15,
    size = 3.8,
    color = "grey20"
  ) +
  
  coord_flip() +
  
  expand_limits(
    y = max(rf_imp_df$MeanDecreaseGini) * 1.15
  ) +
  
  labs(
    title = "Top 10 Most Important Features (Random Forest)",
    subtitle = "Lyrics TF-IDF + Metadata",
    x = NULL,
    y = "Mean Decrease in Gini"
  ) +
  
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 11, color = "grey30"),
    axis.text.y = element_text(face = "bold"),
    panel.grid.major.y = element_blank()
  )











# DV3 – Lyrics vs Metadata Contribution
# Shows relative contribution of feature types
rf_imp_df %>%
  mutate(
    Feature_Type = ifelse(Feature %in% c("year", "artist_freq"), "Metadata", "Lyrics")
  ) %>%
  count(Feature_Type) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = Feature_Type, y = n, fill = Feature_Type)) +
  geom_col(width = 0.65, alpha = 0.9, color = "grey20", linewidth = 0.4) +
  geom_text(aes(label = paste0(n, " (", scales::percent(pct, accuracy = 1), ")")),
            vjust = -0.5, fontface = "bold", size = 4) +
  scale_fill_manual(values = c("Lyrics" = "#E76F51", "Metadata" = "#2A9D8F")) +
  expand_limits(y = max((rf_imp_df %>% mutate(Feature_Type = ifelse(Feature %in% c("year","artist_freq"),"Metadata","Lyrics")) %>% count(Feature_Type))$n) * 1.2) +
  labs(
    title = "Relative Contribution of Lyrics vs Metadata",
    subtitle = "Share among the Top 10 Random Forest features",
    x = "Feature Type",
    y = "Count among Top Features"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(color = "grey30"),
    panel.grid.major.x = element_blank(),
    legend.position = "none"
  )





# DV4 – Prediction Confidence Distribution
# Visualises model uncertainty
ggplot(pred_df, aes(x = prob, fill = true)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 30,
    alpha = 0.4,
    position = "identity",
    color = NA
  ) +
  geom_density(
    alpha = 0.25,
    linewidth = 1
  ) +
  scale_fill_manual(
    values = c(
      "Not Top 50" = "#8E9AAF",
      "Top 50"     = "#6A4C93"
    )
  ) +
  labs(
    title = "Prediction Confidence Distribution",
    subtitle = "Overlap indicates regions of higher model uncertainty",
    x = "Predicted Probability (Top 50)",
    y = "Density",
    fill = "True Class"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 11, color = "grey30"),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )





