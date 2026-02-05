#!/usr/bin/env Rscript
# ============================================================================
# Actual vs Predicted Visualizations
# ============================================================================

library(ggplot2)
library(readr)
library(dplyr)
library(scales)

# Configuration
BASE_DIR <- "/Users/jay/Desktop/Illness Prediction"
VIZ_DIR <- file.path(BASE_DIR, "results/experimental/visualizations")
illness_name <- "Acute Laryngopharyngitis"
model_type <- "XGBoost (Experimental)"

# Load predictions
cat("\n=== Loading Predictions ===\n")
predictions_df <- read_csv(
  file.path(VIZ_DIR, "predictions_for_viz.csv"),
  show_col_types = FALSE
)

# Convert DateTime
predictions_df <- predictions_df %>%
  mutate(DateTime = as.Date(DateTime))

cat(sprintf("Loaded %s predictions\n", comma(nrow(predictions_df))))
cat(sprintf("Date range: %s to %s\n", 
           min(predictions_df$DateTime), 
           max(predictions_df$DateTime)))

# Calculate metrics
metrics_df <- predictions_df %>%
  filter(!is.na(Actual)) %>%
  summarise(
    R2 = cor(Actual, Predicted)^2,
    RMSE = sqrt(mean((Actual - Predicted)^2)),
    MAE = mean(abs(Actual - Predicted))
  )

cat(sprintf("\nModel Performance:\n"))
cat(sprintf("  R² = %.4f\n", metrics_df$R2))
cat(sprintf("  RMSE = %.2f\n", metrics_df$RMSE))
cat(sprintf("  MAE = %.2f\n", metrics_df$MAE))

# === Plot 1: Full Time Series ===
cat("\n=== Creating Full Time Series Plot ===\n")

p1 <- ggplot(predictions_df, aes(x = DateTime)) +
  geom_line(aes(y = Actual, color = "Actual"), alpha = 0.7, size = 0.6) +
  geom_line(aes(y = Predicted, color = "Predicted"), alpha = 0.7, size = 0.6) +
  scale_color_manual(
    values = c("Actual" = "#1F77B4", "Predicted" = "#FF7F0E"),
    name = NULL
  ) +
  scale_y_continuous(labels = comma) +
  labs(
    title = sprintf("Actual vs Predicted Daily Case Counts - %s", illness_name),
    subtitle = sprintf("R² = %.4f | RMSE = %.2f | MAE = %.2f | Model: %s",
                      metrics_df$R2, metrics_df$RMSE, metrics_df$MAE, model_type),
    x = "Date",
    y = "Daily Case Count",
    caption = sprintf("Total samples: %s | Period: %s to %s",
                     comma(nrow(predictions_df)),
                     format(min(predictions_df$DateTime), "%Y-%m-%d"),
                     format(max(predictions_df$DateTime), "%Y-%m-%d"))
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    legend.position = "top",
    legend.text = element_text(size = 11),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(VIZ_DIR, "actual_vs_predicted_timeseries.png"),
  plot = p1,
  width = 14,
  height = 7,
  dpi = 300
)

cat(sprintf("✓ Saved: actual_vs_predicted_timeseries.png\n"))

# === Plot 2: 2022 Sample (for clarity) ===
cat("\n=== Creating 2022 Sample Plot ===\n")

sample_2022 <- predictions_df %>%
  filter(DateTime >= as.Date("2022-01-01") & DateTime <= as.Date("2022-12-31"))

if(nrow(sample_2022) > 0) {
  p2 <- ggplot(sample_2022, aes(x = DateTime)) +
    geom_line(aes(y = Actual, color = "Actual"), size = 1, alpha = 0.8) +
    geom_line(aes(y = Predicted, color = "Predicted"), size = 1, alpha = 0.8) +
    scale_color_manual(
      values = c("Actual" = "#1F77B4", "Predicted" = "#FF7F0E"),
      name = NULL
    ) +
    scale_y_continuous(labels = comma) +
    scale_x_date(date_breaks = "1 month", date_labels = "%b") +
    labs(
      title = sprintf("Model Performance Detail - Year 2022 | %s", illness_name),
      subtitle = "Closer view showing daily prediction accuracy",
      x = "Month (2022)",
      y = "Daily Case Count",
      caption = "Blue = Actual cases | Orange = Model predictions"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "gray40"),
      legend.position = "top",
      legend.text = element_text(size = 11),
      panel.grid.minor = element_blank()
    )
  
  ggsave(
    filename = file.path(VIZ_DIR, "actual_vs_predicted_2022_sample.png"),
    plot = p2,
    width = 14,
    height = 6,
    dpi = 300
  )
  
  cat(sprintf("✓ Saved: actual_vs_predicted_2022_sample.png\n"))
}

# === Plot 3: Scatter Plot ===
cat("\n=== Creating Scatter Plot ===\n")

p3 <- ggplot(predictions_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.3, color = "#1F77B4", size = 1.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
  geom_smooth(method = "lm", color = "#FF7F0E", se = TRUE, alpha = 0.2, size = 1) +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  coord_fixed() +
  labs(
    title = sprintf("Actual vs Predicted - Scatter Plot | %s", illness_name),
    subtitle = sprintf("R² = %.4f | Perfect prediction (red dashed) vs fitted line (orange with confidence band)",
                      metrics_df$R2),
    x = "Actual Case Count",
    y = "Predicted Case Count",
    caption = sprintf("Each point = one day's prediction | Model: %s", model_type)
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(VIZ_DIR, "actual_vs_predicted_scatter.png"),
  plot = p3,
  width = 10,
  height = 10,
  dpi = 300
)

cat(sprintf("✓ Saved: actual_vs_predicted_scatter.png\n"))

# === Plot 4: Residuals ===
cat("\n=== Creating Residuals Plot ===\n")

residuals_df <- predictions_df %>%
  filter(!is.na(Actual)) %>%
  mutate(Residual = Actual - Predicted)

p4 <- ggplot(residuals_df, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.3, color = "#1F77B4", size = 1.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", size = 1) +
  geom_smooth(method = "loess", color = "#FF7F0E", se = TRUE, alpha = 0.2, size = 1) +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(
    title = sprintf("Residual Plot | %s", illness_name),
    subtitle = "Checking for systematic prediction errors (residuals should be randomly distributed around zero)",
    x = "Predicted Case Count",
    y = "Residual (Actual - Predicted)",
    caption = "Red line = zero error | Orange curve = residual trend | Positive = underprediction, Negative = overprediction"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(VIZ_DIR, "residuals_plot.png"),
  plot = p4,
  width = 10,
  height = 7,
  dpi = 300
)

cat(sprintf("✓ Saved: residuals_plot.png\n"))

# === Plot 5: Distribution Comparison ===
cat("\n=== Creating Distribution Plot ===\n")

dist_df <- predictions_df %>%
  select(Actual, Predicted) %>%
  pivot_longer(everything(), names_to = "Type", values_to = "Cases")

p5 <- ggplot(dist_df, aes(x = Cases, fill = Type)) +
  geom_density(alpha = 0.5, size = 1) +
  scale_fill_manual(
    values = c("Actual" = "#1F77B4", "Predicted" = "#FF7F0E"),
    name = NULL
  ) +
  scale_x_continuous(labels = comma, limits = c(0, NA)) +
  labs(
    title = sprintf("Distribution of Actual vs Predicted Case Counts | %s", illness_name),
    subtitle = "Comparing the overall distribution of predictions with actual values",
    x = "Daily Case Count",
    y = "Density",
    caption = "Overlapping distributions indicate good model calibration"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    legend.position = "top",
    legend.text = element_text(size = 11),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(VIZ_DIR, "distribution_comparison.png"),
  plot = p5,
  width = 10,
  height = 6,
  dpi = 300
)

cat(sprintf("✓ Saved: distribution_comparison.png\n"))

cat("\n" %+% strrep("=", 70) %+% "\n")
cat("✓ ALL PREDICTION VISUALIZATIONS COMPLETE\n")
cat(strrep("=", 70) %+% "\n")
cat(sprintf("\nAll plots saved to:\n  %s\n\n", VIZ_DIR))
