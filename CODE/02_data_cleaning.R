############################################################
# 02_data_cleaning.R
#
# Purpose:
# - Select relevant variables required for analysis
# - Handle missing values
# - Create the binary target variable indicating
#   Billboard Top 50 chart success
#
# Target variable:
# - top50 = 1 → song ranked within Top 50
# - top50 = 0 → song ranked outside Top 50
#
# NOTE:
# All code below is copied directly from the original
# working script. No logic has been modified.
############################################################


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
