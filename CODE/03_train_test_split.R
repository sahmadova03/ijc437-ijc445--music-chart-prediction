############################################################
# 03_train_test_split.R
#
# Purpose:
# - Split the cleaned dataset into training and test sets
# - Use stratified sampling to preserve class proportions
# - Define target variables for model training and evaluation
############################################################


############################################################
# 4) Train–test split (stratified sampling)
############################################################

# Stratified split preserves the ratio of
# Top 50 vs Not Top 50 songs in both subsets

set.seed(42)
train_idx <- createDataPartition(df$top50, p = 0.8, list = FALSE)

df_train <- df[train_idx, ]
df_test  <- df[-train_idx, ]


# Define target variables
y_train <- df_train$top50
y_test  <- df_test$top50
