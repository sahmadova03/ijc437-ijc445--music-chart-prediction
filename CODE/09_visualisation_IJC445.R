############################################################
# 09_visualisation_IJC445.R
#
# Purpose:
# - Provide interpretability and uncertainty-focused
#   visualisations for the IJC445 Data Visualisation assessment
# - Explore how textual and metadata features contribute
#   to predictive outcomes
# - Visualise prediction uncertainty and model confidence
#
# Visualisations included:
# - TF-IDF signal distribution by chart class
# - Random Forest feature importance (Top 10)
# - Relative contribution of lyrics vs metadata
# - Prediction confidence distribution

############################################################


############################################################
# VISUALISATIONS – IJC445 (Interpretability & uncertainty)
############################################################

# ---------------------------------------------------------
# DV1 – TF-IDF Signal Distribution
# ---------------------------------------------------------
# Compares overall lexical signal strength between
# Top 50 and Not Top 50 songs

tfidf_summary <- Matrix::rowSums(X_train_tfidf)

df_plot <- tibble(
  tfidf_sum = tfidf_summary,
  chart_class = factor(df_train$top50,
                       labels = c("Not Top 50", "Top 50"))
)

ggplot(df_plot, aes(x = tfidf_sum, fill = chart_class)) +
  geom_density(alpha = 0.45, color = "grey20",
               linewidth = 0.6, adjust = 1.1) +
  geom_vline(
    data = df_plot %>%
      group_by(chart_class) %>%
      summarise(med = median(tfidf_sum)),
    aes(xintercept = med, color = chart_class),
    linetype = "dashed",
    linewidth = 0.8,
    show.legend = FALSE
  ) +
  scale_fill_manual(
    values = c("Not Top 50" = "#E76F51",
               "Top 50"     = "#2A9D8F")
  ) +
  scale_color_manual(
    values = c("Not Top 50" = "#E76F51",
               "Top 50"     = "#2A9D8F")
  ) +
  labs(
    title = "Overall TF-IDF Signal by Chart Class",
    x = "Sum of TF-IDF Weights",
    y = "Density",
    fill = "Chart Class"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0.5),
    axis.title = element_text(face = "bold"),
    legend.title = element_text(face = "bold"),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )


# ---------------------------------------------------------
# DV2 – Random Forest Feature Importance (Top 10)
# ---------------------------------------------------------
# Highlights the most influential features used by the model

rf_imp <- importance(rf_model)

rf_imp_df <- data.frame(
  Feature = rownames(rf_imp),
  MeanDecreaseGini = rf_imp[, "MeanDecreaseGini"]
) %>%
  arrange(desc(MeanDecreaseGini)) %>%
  slice_head(n = 10)

ggplot(
  rf_imp_df,
  aes(x = reorder(Feature, MeanDecreaseGini),
      y = MeanDecreaseGini)
) +
  geom_col(fill = "#2A6F97", width = 0.75) +
  geom_text(
    aes(label = round(MeanDecreaseGini, 2)),
    hjust = -0.15,
    size = 3.8,
    color = "grey20"
  ) +
  coord_flip() +
  expand_limits(
    y = max(rf_imp_df$MeanDecreaseGini) * 1.15
  ) +
  labs(
    title = "Top 10 Most Important Features (Random Forest)",
    subtitle = "Lyrics TF-IDF + Metadata",
    x = NULL,
    y = "Mean Decrease in Gini"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 11, color = "grey30"),
    axis.text.y = element_text(face = "bold"),
    panel.grid.major.y = element_blank()
  )


# ---------------------------------------------------------
# DV3 – Lyrics vs Metadata Contribution
# ---------------------------------------------------------
# Shows the relative contribution of feature types
# among the most important Random Forest predictors

rf_imp_df %>%
  mutate(
    Feature_Type = ifelse(
      Feature %in% c("year", "artist_freq"),
      "Metadata",
      "Lyrics"
    )
  ) %>%
  count(Feature_Type) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(x = Feature_Type, y = n, fill = Feature_Type)) +
  geom_col(
    width = 0.65,
    alpha = 0.9,
    color = "grey20",
    linewidth = 0.4
  ) +
  geom_text(
    aes(label = paste0(n, " (",
                       scales::percent(pct, accuracy = 1), ")")),
    vjust = -0.5,
    fontface = "bold",
    size = 4
  ) +
  scale_fill_manual(
    values = c("Lyrics" = "#E76F51",
               "Metadata" = "#2A9D8F")
  ) +
  labs(
    title = "Relative Contribution of Lyrics vs Metadata",
    subtitle = "Share among the Top 10 Random Forest features",
    x = "Feature Type",
    y = "Count among Top Features"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(color = "grey30"),
    panel.grid.major.x = element_blank(),
    legend.position = "none"
  )


# ---------------------------------------------------------
# DV4 – Prediction Confidence Distribution
# ---------------------------------------------------------
# Visualises model uncertainty via predicted probabilities

pred_df <- tibble(
  prob = rf_prob,
  true = factor(y_test,
                labels = c("Not Top 50", "Top 50"))
)

ggplot(pred_df, aes(x = prob, fill = true)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 30,
    alpha = 0.4,
    position = "identity",
    color = NA
  ) +
  geom_density(alpha = 0.25, linewidth = 1) +
  scale_fill_manual(
    values = c("Not Top 50" = "#8E9AAF",
               "Top 50"     = "#6A4C93")
  ) +
  labs(
    title = "Prediction Confidence Distribution",
    subtitle = "Overlap indicates regions of higher model uncertainty",
    x = "Predicted Probability (Top 50)",
    y = "Density",
    fill = "True Class"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 11, color = "grey30"),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )
