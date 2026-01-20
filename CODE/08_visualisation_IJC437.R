############################################################
# 08_visualisation_IJC437.R
#
# Purpose:
# - Visualise and compare the performance of different
#   predictive models used in the analysis
# - Support model evaluation for the IJC437 assessment
#
# Visualisations included:
# - ROC curve comparison
# - AUC comparison bar plot
# - Accuracy vs balanced accuracy
# - Random Forest feature importance
# - Model complexity vs performance
############################################################


############################################################
# VISUALISATIONS – IJC437 (Model evaluation)
############################################################

# ---------------------------------------------------------
# DS1 – ROC Curve Comparison
# ---------------------------------------------------------
# Compares the discriminative ability of all trained models

plot(
  roc_ridge,
  col = "#DD8452",
  lwd = 2,
  main = "ROC Curves – Model Comparison",
  legacy.axes = TRUE
)

plot(roc_svm, col = "#55A868", lwd = 2, add = TRUE)
plot(roc_rf,  col = "#4C72B0", lwd = 2, add = TRUE)
plot(roc_xgb, col = "#C44E52", lwd = 2, add = TRUE)

legend(
  "bottomright",
  legend = c(
    paste0("Ridge (AUC=", round(auc(roc_ridge), 2), ")"),
    paste0("SVM (AUC=",   round(auc(roc_svm),   2), ")"),
    paste0("RF (AUC=",    round(auc(roc_rf),    2), ")"),
    paste0("XGB (AUC=",   round(auc(roc_xgb),   2), ")")
  ),
  col = c("#DD8452", "#55A868", "#4C72B0", "#C44E52"),
  lwd = 2
)


# ---------------------------------------------------------
# DS2 – AUC Comparison Bar Plot
# ---------------------------------------------------------
# Summarises model performance using a single evaluation metric

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


# ---------------------------------------------------------
# DS3 – Accuracy vs Balanced Accuracy
# ---------------------------------------------------------
# Illustrates the effect of class imbalance on evaluation metrics

metric_df <- tibble(
  Model = c("Ridge", "SVM", "RF", "XGB"),
  Accuracy = c(0.599, 0.642, 0.664, 0.670),
  Balanced_Accuracy = c(0.597, 0.643, 0.664, 0.670)
) %>%
  pivot_longer(
    -Model,
    names_to = "Metric",
    values_to = "Value"
  )

ggplot(metric_df, aes(Model, Value, fill = Metric)) +
  geom_col(position = "dodge") +
  labs(
    title = "Accuracy vs Balanced Accuracy",
    y = "Score"
  ) +
  theme_minimal()


# ---------------------------------------------------------
# DS4 – Feature Importance (Random Forest)
# ---------------------------------------------------------
# Provides insight into the most influential features

varImpPlot(
  rf_model,
  main = "Random Forest Variable Importance",
  n.var = 10
)


# ---------------------------------------------------------
# DS5 – Model Complexity vs Performance
# ---------------------------------------------------------
# Illustrates the trade-off between model complexity and AUC

complexity_df <- tibble(
  Model = c("Ridge", "SVM", "Random Forest", "XGBoost"),
  Complexity = c(1, 2, 3, 4),
  AUC = c(
    auc(roc_ridge),
    auc(roc_svm),
    auc(roc_rf),
    auc(roc_xgb)
  )
)

ggplot(complexity_df, aes(Complexity, AUC, label = Model)) +
  geom_point(size = 4, color = "#4C72B0") +
  geom_text(vjust = -1) +
  scale_x_continuous(
    breaks = 1:4,
    labels = c("Linear", "Kernel", "Ensemble", "Boosted")
  ) +
  labs(
    title = "Model Complexity vs Predictive Performance",
    x = "Model Complexity",
    y = "AUC"
  ) +
  theme_minimal()
