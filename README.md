# IJC437 / IJC445 – Music Chart Success Prediction  
**Predicting Billboard Top 50 Success using Lyrics and Metadata**

**Modules:**  
- IJC437 – Introduction to Data Science  
- IJC445 – Data Visualisation  

**Institution:** University of Sheffield

---

## Project Aim
The aim of this project is to examine whether song lyrics and artist-related metadata can be used to predict commercial music success, operationalised as a song’s entry into the Top 50 of the Billboard Hot 100. Motivated by prior research in music information retrieval and data science, the project focuses on the underexplored role of lyrical content and evaluates how textual features interact with simple contextual metadata within predictive modelling frameworks. Beyond predictive accuracy, the project emphasises transparency, interpretability, and uncertainty-aware visualisation.

---

## Research Questions
This project addresses the following research questions:

- **RQ1:** To what extent can song lyrics, represented using TF-IDF features, distinguish between Top 50 and non-Top 50 songs?
- **RQ2:** Does combining lyrical features with artist-related metadata improve predictive performance?
- **RQ3:** Which modelling approaches (logistic regression, support vector machines, random forests, and gradient boosting) are most effective for predicting chart success in this dataset?

---

## Dataset
The analysis uses the **Billboard Hot 100 (2000–2023)** dataset, which combines chart rankings with song lyrics and artist-related metadata.

**Source:** Kaggle  
https://www.kaggle.com/datasets/suparnabiswas/billboard-hot-1002000-2023-data-with-features  

The dataset is downloaded programmatically using the **Kaggle API** via the Kaggle Command Line Interface (CLI).

### Data Access Requirements
To run the code as provided, users must:

1. Create a Kaggle account.
2. Generate a Kaggle API token (`kaggle.json`) from their Kaggle account settings.
3. Place the `kaggle.json` file in the appropriate local directory as required by Kaggle.
4. Ensure that the Kaggle CLI is installed and accessible from the system path.

For security reasons, the `kaggle.json` file is **not included** in this repository.

### Dataset Content
The dataset includes:
- Weekly Billboard chart rankings  
- Song titles and artist names  
- Full song lyrics  
- Release year metadata  

A binary target variable is constructed:
- **Top 50 (1)** – song ranked within the Billboard Top 50  
- **Not Top 50 (0)** – song ranked outside the Top 50  

Covering more than two decades, the dataset enables analysis of long-term trends in musical content, language use, and popularity as reflected in chart performance.
---

## Methods (Pipeline Overview)
A structured and reproducible data science pipeline is implemented:

1. **Data Loading and Cleaning**  
   The dataset is downloaded using the Kaggle API, relevant variables are selected, and missing values are handled. A binary target variable is constructed.

2. **Train–Test Split**  
   Stratified sampling is used to preserve class balance between Top 50 and non-Top 50 songs.

3. **Text Processing and Feature Engineering**  
   Song titles and lyrics are cleaned and combined. TF-IDF features are derived using training data only to prevent data leakage.

4. **Metadata Features**  
   Artist frequency and release year are computed from training data, standardised, and combined with TF-IDF features.

5. **Model Training and Evaluation**  
   Multiple models are trained and compared:
   - Ridge Logistic Regression  
   - Support Vector Machine (RBF kernel)  
   - Random Forest  
   - XGBoost  

   Performance is evaluated using confusion matrices and ROC/AUC metrics.

6. **Data Visualisation**  
   Dedicated visualisations support:
   - Model comparison and evaluation (IJC437)  
   - Interpretability and predictive uncertainty (IJC445)
   - 
Additional visual outputs generated during the analysis are available in the `outputs/figures/` directory.
<img width="716" height="482" alt="ijc437_comparison_models" src="https://github.com/user-attachments/assets/8e2177c9-d098-4946-8d46-5238acdef649" />

---


## Key Findings
- **Lyrical features provide meaningful predictive signal**, but do not fully separate Top 50 and non-Top 50 songs, indicating that lyrics alone are insufficient predictors of chart success.
- **Combining lyrics with metadata improves predictive performance**, with artist frequency emerging as a particularly influential contextual feature.
- **Ensemble-based models (Random Forest and XGBoost)** achieve the strongest overall performance, reflecting their ability to capture non-linear relationships.
- **Increased model complexity improves AUC**, but introduces trade-offs in interpretability and computational cost.
- Visualisations reveal substantial **prediction uncertainty**, highlighting overlap between successful and unsuccessful songs rather than deterministic outcomes.

---

## How to Run the Code
Run the scripts in the following order:

1. `00_setup.R` – Install and load required packages  
2. `01_data_loading.R` – Download and load the dataset  
3. `02_data_cleaning.R` – Clean data and create target variable  
4. `03_train_test_split.R` – Create stratified train/test sets  
5. `04_text_cleaning.R` – Clean and prepare text data  
6. `05_tfidf_feature_engineering.R` – Build TF-IDF features  
7. `06_metadata_features.R` – Create and combine metadata features  
8. `07_model_training_and_evaluation.R` – Train models and evaluate performance  
9. `08_visualisation_IJC437.R` – Model evaluation visualisations  
10. `09_visualisation_IJC445.R` – Interpretability and uncertainty visualisations  

> **Note:** Downloading the dataset via Kaggle requires Kaggle API credentials.  
> Alternatively, the CSV file can be manually placed in the `billboard_data/` directory.

---

## Repository Structure
```text
CODE/
 ┣ 00_setup.R
 ┣ 01_data_loading.R
 ┣ 02_data_cleaning.R
 ┣ 03_train_test_split.R
 ┣ 04_text_cleaning.R
 ┣ 05_tfidf_feature_engineering.R
 ┣ 06_metadata_features.R
 ┣ 07_model_training_and_evaluation.R
 ┣ 08_visualisation_IJC437.R
 ┗ 09_visualisation_IJC445.R

script.R   # Original unmodified working script (backup)

## Author
Student: **Sara Ahmadova**  
Modules: **IJC437 – Introduction to Data Science** and **IJC445 – Data Visualisation**  
University of Sheffield



