############################################################
# 06_metadata_features.R
#
# Purpose:
# - Create metadata-based features (artist frequency and release year)
# - Ensure all metadata features are derived from training data only
# - Standardise metadata features using training statistics
# - Combine metadata features with TF-IDF text features
############################################################


############################################################
# 7) Metadata features (artist frequency and year)
############################################################

# Artist frequency is computed from training data only
artist_freq <- df_train %>%
  count(band_singer, name = "artist_freq")


# Prepare metadata features for the training set
meta_train <- df_train %>%
  left_join(artist_freq, by = "band_singer") %>%
  mutate(artist_freq = replace_na(artist_freq, 1)) %>%
  transmute(
    year = as.numeric(year),
    artist_freq = as.numeric(artist_freq)
  )


# Prepare metadata features for the test set
meta_test <- df_test %>%
  left_join(artist_freq, by = "band_singer") %>%
  mutate(artist_freq = replace_na(artist_freq, 1)) %>%
  transmute(
    year = as.numeric(year),
    artist_freq = as.numeric(artist_freq)
  )


# Standardise metadata using statistics from the training data
meta_means <- colMeans(meta_train)
meta_sds   <- apply(meta_train, 2, sd)
meta_sds[meta_sds == 0] <- 1  # safety check to avoid division by zero

meta_train_sc <- scale(meta_train, center = meta_means, scale = meta_sds)
meta_test_sc  <- scale(meta_test,  center = meta_means, scale = meta_sds)


# Combine TF-IDF text features and metadata features
# to form the final feature matrices
X_train <- cbind(X_train_tfidf, Matrix(meta_train_sc, sparse = TRUE))
X_test  <- cbind(X_test_tfidf,  Matrix(meta_test_sc,  sparse = TRUE))

# Sanity check: ensure matching feature dimensions
stopifnot(ncol(X_train) == ncol(X_test))
