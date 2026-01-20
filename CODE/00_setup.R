############################################################
# IJC437 – Introduction to Data Science and IJC445 Data Visualisation

# Project: Predicting Billboard Top 50 Chart Success
# Dataset: Billboard Hot 100 (2000–2023)
#
# NOTE:
# This script implements the full data science pipeline:
# data loading → text processing → feature engineering →
# model training → evaluation → visualisation.
############################################################


############################################################
# 1) Package installation and loading
############################################################

# Install required packages (run once if not installed)
# NOTE: This line should be executed only once.

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
