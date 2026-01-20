############################################################
# 05_tfidf_feature_engineering.R
#
# Purpose:
# - Perform TF-IDF feature engineering using song lyrics and titles
# - Build the vocabulary using training data only to prevent data leakage
# - Construct sparse TF-IDF matrices for both training and test sets
#
# NOTE:
# All code below is copied directly from the original
# working script. No logic has been modified.
############################################################


############################################################
# 6) TF-IDF feature engineering (training data only)
############################################################

# Tokenise training text and remove stopwords
# Vocabulary is built ONLY on training data to avoid data leakage

train_tokens <- df_train %>%
  mutate(doc_id = row_number()) %>%
  select(doc_id, text) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word") %>%
  filter(nchar(word) > 2) %>%
  mutate(word = wordStem(word)) %>%
  count(doc_id, word, sort = FALSE)




# Compute TF-IDF scores for the training data
train_tfidf <- train_tokens %>%
  bind_tf_idf(word, doc_id, n)


# Select the top 2000 most informative words based on TF-IDF scores
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
# - the same vocabulary is used for train and test sets
# - identical column ordering
# - missing terms are filled with zeros


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
  
# Reorder columns to exactly match the training vocabulary
  X <- X[, vocab, drop = FALSE]  # reorder exactly
  
  return(X)
}



# Build TF-IDF matrices for train and test sets
X_train_tfidf <- build_tfidf_sparse(df_train, top_words)
X_test_tfidf  <- build_tfidf_sparse(df_test,  top_words)


# Sanity checks to ensure consistency
stopifnot(nrow(X_train_tfidf) == nrow(df_train))
stopifnot(nrow(X_test_tfidf)  == nrow(df_test))
stopifnot(ncol(X_train_tfidf) == ncol(X_test_tfidf))
