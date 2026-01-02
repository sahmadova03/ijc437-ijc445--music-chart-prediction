# ijc437 / ijc445 – Music Chart Success Prediction using Lyrics and Metadata

IJC437 – Introduction to Data Science  
IJC445 – Data Visualisation  

University of Sheffield

---

## Project Overview
This project was developed as part of the **IJC437 – Introduction to Data Science** and  
**IJC445 – Data Visualisation** modules at the University of Sheffield.

The aim of the project is to investigate whether song lyrics and basic artist-related metadata can be used to predict commercial success on the **Billboard Hot 100** chart.

Using a structured data science workflow, the project applies text mining and machine learning techniques in R to model the likelihood of a song reaching the **Top 50** of the Billboard Hot 100 between **2000 and 2023**.

---

## Research Aim
The primary aim of this study is to evaluate the predictive power of lyrical content, represented using TF-IDF features, combined with artist-related metadata, in identifying songs that achieve chart success.

---

## Research Questions
- **RQ1:** To what extent can lyrical content distinguish Top 50 songs from non-Top 50 songs?
- **RQ2:** Does combining lyrics with metadata improve predictive performance?
- **RQ3:** Which classification models perform best for this prediction task?

---

## Dataset
The dataset is downloaded directly from Kaggle using the Kaggle CLI.

To run the code, users must:
1. Create a Kaggle account.
2. Generate a Kaggle API token (`kaggle.json`).
3. Place the `kaggle.json` file in the appropriate local directory as required by Kaggle.
4. Ensure the Kaggle CLI is installed and accessible from the system path.

For security reasons, the `kaggle.json` file is not included in this repository.

The analysis uses the following publicly available dataset:

**Billboard Hot 100 (2000–2023)**  
Source: Kaggle  
https://www.kaggle.com/datasets/suparnabiswas/billboard-hot-1002000-2023-data-with-features  

The dataset combines:
- Weekly Billboard rankings
- Song titles and artist names
- Full song lyrics
- Release year metadata

A binary target variable is constructed where:
- `1` = Song ranked in the Top 50  
- `0` = Song ranked outside the Top 50  

---

## Methodology Summary
The project follows a typical data science process:
1. Data acquisition and cleaning  
2. Text pre-processing (normalisation, stop-word removal, stemming)  
3. Feature engineering using **TF-IDF**  
4. Metadata enrichment (artist frequency, release year)  
5. Stratified train–test split  
6. Model training and evaluation  
7. Visualisation and interpretation of results  

The TF-IDF vocabulary is constructed using **training data only** to avoid data leakage.

---

## Models Used
The following classification models were implemented and compared:
- Ridge Logistic Regression
- Support Vector Machine (RBF kernel)
- Random Forest
- XGBoost

Model performance is evaluated using:
- Confusion matrices
- ROC curves
- Area Under the Curve (AUC)
- Accuracy and balanced accuracy

---

## Key Findings
- Models combining **lyrics and metadata** outperform those relying on lyrics alone.
- **Ensemble-based methods** (Random Forest and XGBoost) achieve the strongest predictive performance.
- Artist frequency emerges as an important contextual feature alongside lyrical patterns.
- Prediction uncertainty highlights that chart success is influenced by additional external factors not captured in the dataset.

---

## Code Structure
ijc437-ijc445-music-chart-prediction/
│
├── code/
│ └── script.R
│
├── README.md


All analysis, modelling, and visualisation steps are contained in a single, well-documented R script located in the `code/` directory.

---

## How to Run the Code
1. Install R and RStudio  
2. Install the required R packages:

tidyverse, caret, tidytext, SnowballC,
Matrix, glmnet, e1071, randomForest,
xgboost, pROC


3. Ensure the Kaggle API is installed and configured locally  
4. Run the script:

   
code/script.R


The script automatically downloads the dataset, performs the analysis, and generates all results and visualisations used in the report.

---

## Author
Student: **Sara Ahmadova**  
Modules: **IJC437 – Introduction to Data Science** and **IJC445 – Data Visualisation**  
University of Sheffield

