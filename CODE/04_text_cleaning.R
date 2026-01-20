############################################################
# 04_text_cleaning.R
#
# Purpose:
# - Clean raw text data from song titles and lyrics
# - Standardise text format for downstream NLP processing
# - Create a combined text field for each song
#
# Text cleaning steps:
# - Convert to lowercase
# - Remove non-alphabetic characters
# - Remove extra whitespace

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
