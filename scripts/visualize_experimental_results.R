#!/usr/bin/env Rscript
# ============================================================================
# Visualization Script for Experimental XGBoost Model Results
# ============================================================================
# Purpose: Generate publication-quality visualizations for experimental model
# Author: Generated for Illness Prediction Research
# Date: February 4, 2026
# ============================================================================

# Load required libraries
required_packages <- c("ggplot2", "readr", "dplyr", "tidyr", "scales", "reticulate")

# Install missing packages
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  install.packages(new_packages, repos = "https://cloud.r-project.org/")
}

# Load libraries
library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)
library(scales)
library(reticulate)

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR <- "/Users/jay/Desktop/Illness Prediction"
RESULTS_DIR <- file.path(BASE_DIR, "results/experimental/models/experimental/Acute_laryngopharyngitis")
OUTPUT_DIR <- file.path(BASE_DIR, "results/experimental/visualizations")

# Create output directory
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

illness_name <- "Acute Laryngopharyngitis"
model_type <- "XGBoost (Experimental)"

# ============================================================================
# 1. FEATURE IMPORTANCE VISUALIZATION
# ============================================================================

cat("\n=== Loading Feature Importance Data ===\n")

# Load feature importance
feature_importance <- read_csv(
  file.path(RESULTS_DIR, "feature_importance_experimental_Acute_laryngopharyngitis_lag0.csv"),
  show_col_types = FALSE
)

cat(sprintf("Loaded %d features\n", nrow(feature_importance)))

# Add feature categories
feature_importance <- feature_importance %>%
  mutate(
    category = case_when(
      grepl("^RegionName_", feature) ~ "Regional",
      grepl("^Season_", feature) ~ "Seasonal",
      feature %in% c("Year", "Month", "DayOfWeek", "WeekOfYear", "DayOfYear") ~ "Temporal",
      grepl("_lag_", feature) ~ "Experimental TS (Lag)",
      grepl("_rolling_", feature) ~ "Rolling Statistics",
      feature %in% c("SO2", "CO", "O3", "NO2", "PM10", "PM25") ~ "Air Quality",
      grepl("Temp|Snow|Rain|Wind|Humidity|Pressure", feature) ~ "Climate",
      grepl("Soil|Ground", feature) ~ "Soil/Ground",
      TRUE ~ "Other"
    )
  )

# === Plot 1: Top 20 Features ===
cat("\n=== Creating Top 20 Features Plot ===\n")

top20 <- feature_importance %>%
  arrange(desc(importance)) %>%
  head(20) %>%
  mutate(feature = reorder(feature, importance))

p1 <- ggplot(top20, aes(x = importance, y = feature, fill = category)) +
  geom_col(width = 0.7) +
  scale_fill_brewer(palette = "Set2") +
  scale_x_continuous(labels = scales::percent_format(accuracy = 0.1)) +
  labs(
    title = sprintf("Top 20 Most Important Features - %s", illness_name),
    subtitle = sprintf("Model: %s | Total Features: %d selected", model_type, nrow(feature_importance)),
    x = "Feature Importance (%)",
    y = NULL,
    fill = "Category",
    caption = "Feature importance based on XGBoost gain metric"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right"
  )

ggsave(
  filename = file.path(OUTPUT_DIR, "feature_importance_top20.png"),
  plot = p1,
  width = 12,
  height = 8,
  dpi = 300
)

cat(sprintf("✓ Saved: %s\n", file.path(OUTPUT_DIR, "feature_importance_top20.png")))

# === Plot 2: Feature Importance by Category ===
cat("\n=== Creating Feature Importance by Category Plot ===\n")

category_importance <- feature_importance %>%
  group_by(category) %>%
  summarise(
    total_importance = sum(importance),
    n_features = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(total_importance)) %>%
  mutate(
    category = reorder(category, total_importance),
    pct_label = sprintf("%.1f%%\n(%d features)", total_importance * 100, n_features)
  )

p2 <- ggplot(category_importance, aes(x = total_importance, y = category, fill = category)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_text(aes(label = pct_label), hjust = -0.1, size = 3.5) +
  scale_x_continuous(
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.15))
  ) +
  scale_fill_brewer(palette = "Spectral") +
  labs(
    title = sprintf("Feature Importance by Category - %s", illness_name),
    subtitle = sprintf("Model: %s | %d total features across %d categories", 
                      model_type, nrow(feature_importance), nrow(category_importance)),
    x = "Total Importance (%)",
    y = NULL,
    caption = "Importance aggregated by feature category"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray40"),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(OUTPUT_DIR, "feature_importance_by_category.png"),
  plot = p2,
  width = 10,
  height = 7,
  dpi = 300
)

cat(sprintf("✓ Saved: %s\n", file.path(OUTPUT_DIR, "feature_importance_by_category.png")))

# === Plot 3: Experimental Time Series Features Only ===
cat("\n=== Creating Experimental TS Features Plot ===\n")

experimental_features <- feature_importance %>%
  filter(grepl("_lag_.*_rolling_", feature) | grepl("_rolling_mean_", feature)) %>%
  arrange(desc(importance)) %>%
  mutate(
    feature = reorder(feature, importance),
    feature_type = ifelse(grepl("_lag_.*_rolling_", feature), 
                          "Lag + Rolling Mean", 
                          "Rolling Mean (Base)")
  )

if(nrow(experimental_features) > 0) {
  p3 <- ggplot(experimental_features, aes(x = importance, y = feature, fill = feature_type)) +
    geom_col(width = 0.7) +
    scale_fill_manual(values = c("Lag + Rolling Mean" = "#E41A1C", 
                                  "Rolling Mean (Base)" = "#377EB8")) +
    scale_x_continuous(labels = scales::percent_format(accuracy = 0.01)) +
    labs(
      title = "Experimental Time Series Features (NEW)",
      subtitle = sprintf("%d features with rolling statistics on lag features", nrow(experimental_features)),
      x = "Feature Importance (%)",
      y = NULL,
      fill = "Feature Type",
      caption = "These granular lag features (2, 5, 8, 12 days) are unique to the experimental model"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14, color = "#E41A1C"),
      plot.subtitle = element_text(size = 11, color = "gray40"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "right"
    )
  
  ggsave(
    filename = file.path(OUTPUT_DIR, "experimental_features_importance.png"),
    plot = p3,
    width = 10,
    height = 6,
    dpi = 300
  )
  
  cat(sprintf("✓ Saved: %s\n", file.path(OUTPUT_DIR, "experimental_features_importance.png")))
}

# ============================================================================
# 2. MODEL PERFORMANCE: ACTUAL VS PREDICTED
# ============================================================================

cat("\n=== Loading Model Predictions ===\n")

# Load the trained model and generate predictions using Python
use_python("/usr/bin/python3", required = FALSE)

tryCatch({
  # Python code to load model and generate predictions
  py_run_string("
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
base_dir = '/Users/jay/Desktop/Illness Prediction'
model_path = f'{base_dir}/results/experimental/models/experimental/Acute_laryngopharyngitis/model_experimental_Acute_laryngopharyngitis_lag0.pkl'
data_path = f'{base_dir}/Processed Data/Illness & Environmental/Grouped/experimental/merged_data_Acute_laryngopharyngitis_lag0_comprehensive_ts.csv'

# Load model
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)
    
model = model_data['model']
selected_features = model_data['selected_features']

# Load data
df = pd.read_csv(data_path)

# Prepare features
X = df[selected_features]
y = df['CaseCount'] if 'CaseCount' in df.columns else None

# Generate predictions
predictions = model.predict(X)

# Create results dataframe
results = pd.DataFrame({
    'DateTime': df['DateTime'] if 'DateTime' in df.columns else range(len(predictions)),
    'Region': df['RegionName'] if 'RegionName' in df.columns else 'Unknown',
    'Actual': y if y is not None else np.nan,
    'Predicted': predictions
})

# Save to CSV for R to read
results.to_csv(f'{base_dir}/results/experimental/visualizations/predictions_for_viz.csv', index=False)
print(f'Predictions generated: {len(results)} samples')
  ")
  
  # Read predictions from Python output
  predictions_df <- read_csv(
    file.path(OUTPUT_DIR, "predictions_for_viz.csv"),
    show_col_types = FALSE
  )
  
  # Convert DateTime if present
  if("DateTime" %in% names(predictions_df)) {
    predictions_df <- predictions_df %>%
      mutate(DateTime = as.Date(DateTime))
  } else {
    predictions_df <- predictions_df %>%
      mutate(DateTime = as.Date("2019-01-01") + row_number())
  }
  
  cat(sprintf("✓ Loaded %d predictions\n", nrow(predictions_df)))
  
  # === Plot 4: Time Series - All Data ===
  cat("\n=== Creating Overall Time Series Plot ===\n")
  
  # Calculate metrics
  metrics_df <- predictions_df %>%
    filter(!is.na(Actual)) %>%
    summarise(
      R2 = cor(Actual, Predicted)^2,
      RMSE = sqrt(mean((Actual - Predicted)^2)),
      MAE = mean(abs(Actual - Predicted))
    )
  
  p4 <- ggplot(predictions_df, aes(x = DateTime)) +
    geom_line(aes(y = Actual, color = "Actual"), alpha = 0.7, size = 0.8) +
    geom_line(aes(y = Predicted, color = "Predicted"), alpha = 0.7, size = 0.8) +
    scale_color_manual(
      values = c("Actual" = "#1F77B4", "Predicted" = "#FF7F0E"),
      name = NULL
    ) +
    scale_y_continuous(labels = comma) +
    labs(
      title = sprintf("Actual vs Predicted Case Counts - %s", illness_name),
      subtitle = sprintf("R² = %.3f | RMSE = %.2f | MAE = %.2f | Model: %s",
                        metrics_df$R2, metrics_df$RMSE, metrics_df$MAE, model_type),
      x = "Date",
      y = "Daily Case Count",
      caption = sprintf("Total samples: %s | Time period: %s to %s",
                       comma(nrow(predictions_df)),
                       format(min(predictions_df$DateTime, na.rm = TRUE), "%Y-%m-%d"),
                       format(max(predictions_df$DateTime, na.rm = TRUE), "%Y-%m-%d"))
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "gray40"),
      legend.position = "top",
      panel.grid.minor = element_blank()
    )
  
  ggsave(
    filename = file.path(OUTPUT_DIR, "actual_vs_predicted_timeseries.png"),
    plot = p4,
    width = 14,
    height = 7,
    dpi = 300
  )
  
  cat(sprintf("✓ Saved: %s\n", file.path(OUTPUT_DIR, "actual_vs_predicted_timeseries.png")))
  
  # === Plot 5: Scatter Plot - Actual vs Predicted ===
  cat("\n=== Creating Scatter Plot ===\n")
  
  p5 <- ggplot(predictions_df, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.3, color = "#1F77B4", size = 1.5) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
    geom_smooth(method = "lm", color = "#FF7F0E", se = TRUE, alpha = 0.2) +
    scale_x_continuous(labels = comma) +
    scale_y_continuous(labels = comma) +
    labs(
      title = sprintf("Actual vs Predicted - Scatter Plot | %s", illness_name),
      subtitle = sprintf("R² = %.3f | Perfect prediction line (red dashed) vs fitted line (orange)",
                        metrics_df$R2),
      x = "Actual Case Count",
      y = "Predicted Case Count",
      caption = sprintf("Each point = one day | Model: %s", model_type)
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "gray40"),
      panel.grid.minor = element_blank()
    )
  
  ggsave(
    filename = file.path(OUTPUT_DIR, "actual_vs_predicted_scatter.png"),
    plot = p5,
    width = 10,
    height = 10,
    dpi = 300
  )
  
  cat(sprintf("✓ Saved: %s\n", file.path(OUTPUT_DIR, "actual_vs_predicted_scatter.png")))
  
  # === Plot 6: By Region (if available) ===
  if("Region" %in% names(predictions_df)) {
    cat("\n=== Creating Regional Comparison Plot ===\n")
    
    # Sample regions for cleaner visualization
    top_regions <- predictions_df %>%
      group_by(Region) %>%
      summarise(total_cases = sum(Actual, na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(total_cases)) %>%
      head(4) %>%
      pull(Region)
    
    regional_data <- predictions_df %>%
      filter(Region %in% top_regions) %>%
      filter(!is.na(DateTime) & DateTime >= as.Date("2021-01-01") & DateTime <= as.Date("2021-12-31"))
    
    if(nrow(regional_data) > 0) {
      p6 <- ggplot(regional_data, aes(x = DateTime)) +
        geom_line(aes(y = Actual, color = "Actual"), alpha = 0.6) +
        geom_line(aes(y = Predicted, color = "Predicted"), alpha = 0.6) +
        facet_wrap(~Region, scales = "free_y", ncol = 2) +
        scale_color_manual(
          values = c("Actual" = "#1F77B4", "Predicted" = "#FF7F0E"),
          name = NULL
        ) +
        scale_y_continuous(labels = comma) +
        labs(
          title = sprintf("Regional Performance - Top 4 Regions | %s", illness_name),
          subtitle = "Year 2021 sample | Model predictions vs actual case counts by region",
          x = "Date",
          y = "Daily Case Count",
          caption = "Regions selected by highest total case counts"
        ) +
        theme_minimal(base_size = 11) +
        theme(
          plot.title = element_text(face = "bold", size = 13),
          plot.subtitle = element_text(size = 10, color = "gray40"),
          legend.position = "top",
          strip.text = element_text(face = "bold", size = 11),
          panel.grid.minor = element_blank()
        )
      
      ggsave(
        filename = file.path(OUTPUT_DIR, "actual_vs_predicted_by_region.png"),
        plot = p6,
        width = 14,
        height = 10,
        dpi = 300
      )
      
      cat(sprintf("✓ Saved: %s\n", file.path(OUTPUT_DIR, "actual_vs_predicted_by_region.png")))
    }
  }
  
  # === Plot 7: Residuals Analysis ===
  cat("\n=== Creating Residuals Plot ===\n")
  
  residuals_df <- predictions_df %>%
    filter(!is.na(Actual)) %>%
    mutate(Residual = Actual - Predicted)
  
  p7 <- ggplot(residuals_df, aes(x = Predicted, y = Residual)) +
    geom_point(alpha = 0.3, color = "#1F77B4", size = 1.5) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed", size = 1) +
    geom_smooth(method = "loess", color = "#FF7F0E", se = TRUE, alpha = 0.2) +
    scale_x_continuous(labels = comma) +
    scale_y_continuous(labels = comma) +
    labs(
      title = sprintf("Residual Plot | %s", illness_name),
      subtitle = "Checking for systematic prediction errors (residuals should be randomly distributed around zero)",
      x = "Predicted Case Count",
      y = "Residual (Actual - Predicted)",
      caption = "Red line = zero error | Orange curve = residual trend"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "gray40"),
      panel.grid.minor = element_blank()
    )
  
  ggsave(
    filename = file.path(OUTPUT_DIR, "residuals_plot.png"),
    plot = p7,
    width = 10,
    height = 7,
    dpi = 300
  )
  
  cat(sprintf("✓ Saved: %s\n", file.path(OUTPUT_DIR, "residuals_plot.png")))
  
}, error = function(e) {
  cat(sprintf("\n⚠ Error generating predictions: %s\n", e$message))
  cat("Skipping actual vs predicted plots.\n")
  cat("This may occur if Python/pickle dependencies are not available.\n")
})

# ============================================================================
# Summary
# ============================================================================

cat("\n" %+% strrep("=", 70) %+% "\n")
cat("✓ VISUALIZATION COMPLETE\n")
cat(strrep("=", 70) %+% "\n")
cat(sprintf("\nAll visualizations saved to:\n  %s\n", OUTPUT_DIR))
cat("\nGenerated plots:\n")
cat("  1. feature_importance_top20.png\n")
cat("  2. feature_importance_by_category.png\n")
cat("  3. experimental_features_importance.png\n")
cat("  4. actual_vs_predicted_timeseries.png\n")
cat("  5. actual_vs_predicted_scatter.png\n")
cat("  6. actual_vs_predicted_by_region.png\n")
cat("  7. residuals_plot.png\n")
cat("\n")
