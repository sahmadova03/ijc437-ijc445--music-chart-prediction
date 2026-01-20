############################################################
# 01_data_loading.R
#
# Purpose:
# - Download the Billboard Hot 100 dataset using Kaggle CLI
# - Unzip the dataset files
# - Load the main CSV file into R
# - Perform an initial inspection of the raw data

############################################################
# 2) Data loading
############################################################

# Download dataset from Kaggle using Kaggle CLI
# This step requires Kaggle API credentials to be configured locally.
# The Kaggle executable path is system-specific.
system('C:\\Users\\ahmad\\AppData\\Local\\Python\\PythonCore-3.14-64\\Scripts\\kaggle.exe datasets download -d suparnabiswas/billboard-hot-1002000-2023-data-with-features')

# Unzip the downloaded dataset into a local directory
unzip("billboard-hot-1002000-2023-data-with-features.zip",
      exdir = "billboard_data")

# Read the dataset into R
df <- read_csv("billboard_data/billboard_24years_lyrics_spotify.csv")

# Inspect dataset structure and contents
glimpse(df)
view(df)
