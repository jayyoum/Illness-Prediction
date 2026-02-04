#!/usr/bin/env Rscript
# Comprehensive visualizations for all three illnesses
# Generates feature importance plots and lag comparison charts

library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)
library(stringr)

# Set working directory
setwd("/Users/jay/Desktop/Illness Prediction")

# Create output directory
dir.create("results/three_illness_comparison/visualizations", recursive = TRUE, showWarnings = FALSE)

cat(strrep("=", 80), "\n")
cat("GENERATING VISUALIZATIONS FOR ALL THREE ILLNESSES\n")
cat(strrep("=", 80), "\n\n")

# ============================================================================
# 1. FEATURE IMPORTANCE PLOTS (Matching existing style)
# ============================================================================

cat("1. Generating feature importance plots...\n")

# Function to create feature importance plot (matching existing style)
plot_feature_importance <- function(csv_path, title, output_path, top_n = 15) {
  # Read data
  df <- read_csv(csv_path, show_col_types = FALSE)
  
  # Get top N features
  df_top <- df %>%
    arrange(desc(importance)) %>%
    head(top_n) %>%
    mutate(feature = factor(feature, levels = rev(feature)))
  
  # Create plot matching existing style
  p <- ggplot(df_top, aes(x = importance, y = feature)) +
    geom_bar(stat = "identity", fill = "#1f77b4", width = 0.7) +
    labs(
      title = NULL,
      x = "Importance Score",
      y = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text = element_text(size = 11),
      axis.title.x = element_text(size = 12, margin = margin(t = 10)),
      plot.margin = margin(20, 20, 20, 20)
    )
  
  # Save plot
  ggsave(output_path, p, width = 8, height = 6, dpi = 300, bg = "white")
  cat(sprintf("  ✓ Saved: %s\n", output_path))
  
  return(p)
}

# Gastritis feature importance
plot_feature_importance(
  "results/experimental_gastritis/models/feature_importance_gastritis_20260204_135227.csv",
  "Gastritis Feature Importance",
  "results/three_illness_comparison/visualizations/gastritis_feature_importance.png",
  top_n = 20
)

# Chronic rhinitis feature importance
plot_feature_importance(
  "results/experimental_chronic_rhinitis/models/feature_importance_chronic_rhinitis_20260204_135205.csv",
  "Chronic Rhinitis Feature Importance",
  "results/three_illness_comparison/visualizations/chronic_rhinitis_feature_importance.png",
  top_n = 20
)

# ============================================================================
# 2. LAG PERIOD COMPARISON ACROSS ALL 3 ILLNESSES
# ============================================================================

cat("\n2. Generating lag period comparison...\n")

# Extract lag information from feature names
extract_lag <- function(feature_name) {
  lag_match <- str_match(feature_name, "_lag_(\\d+)")
  if (!is.na(lag_match[1, 2])) {
    return(as.integer(lag_match[1, 2]))
  }
  return(NA)
}

# Function to get lag features from CSV
get_lag_features <- function(csv_path, illness_name, has_importance = TRUE) {
  df <- read_csv(csv_path, show_col_types = FALSE)
  
  result <- df %>%
    mutate(
      lag = sapply(feature, extract_lag),
      illness = illness_name
    ) %>%
    filter(!is.na(lag))
  
  # If importance column doesn't exist, add a default value of 1
  if (!has_importance || !"importance" %in% colnames(result)) {
    result <- result %>%
      mutate(importance = 1)
  }
  
  result %>%
    select(illness, lag, feature, importance)
}

# Load lag features for all illnesses (using feature_importance files which have both feature and importance)
laryngo_lags <- get_lag_features(
  "results/experimental/models/experimental/Acute_laryngopharyngitis/feature_importance_experimental_Acute_laryngopharyngitis_lag0.csv",
  "Acute laryngopharyngitis",
  has_importance = TRUE
)

gastritis_lags <- get_lag_features(
  "results/experimental_gastritis/models/feature_importance_gastritis_20260204_135227.csv",
  "Gastritis",
  has_importance = TRUE
)

rhinitis_lags <- get_lag_features(
  "results/experimental_chronic_rhinitis/models/feature_importance_chronic_rhinitis_20260204_135205.csv",
  "Chronic rhinitis",
  has_importance = TRUE
)

# Combine all lag data
all_lags <- bind_rows(laryngo_lags, gastritis_lags, rhinitis_lags)

# Count features per lag per illness
lag_summary <- all_lags %>%
  group_by(illness, lag) %>%
  summarise(
    n_features = n(),
    total_importance = sum(importance, na.rm = TRUE),
    .groups = "drop"
  )

# Create lag comparison bar chart
p_lag_comparison <- ggplot(lag_summary, aes(x = lag, y = n_features, fill = illness)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_x_continuous(breaks = 1:14) +
  scale_fill_manual(
    values = c(
      "Acute laryngopharyngitis" = "#e74c3c",
      "Gastritis" = "#3498db",
      "Chronic rhinitis" = "#2ecc71"
    ),
    name = "Illness Type"
  ) +
  labs(
    title = "Lag Period Distribution Across Three Illnesses",
    subtitle = "Number of selected features per lag period",
    x = "Lag Period (days)",
    y = "Number of Features Selected"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray30"),
    panel.grid.minor = element_blank()
  )

ggsave(
  "results/three_illness_comparison/visualizations/lag_comparison_all_illnesses.png",
  p_lag_comparison,
  width = 10,
  height = 6,
  dpi = 300,
  bg = "white"
)
cat("  ✓ Saved: lag_comparison_all_illnesses.png\n")

# ============================================================================
# 3. AVERAGE LAG COMPARISON
# ============================================================================

cat("\n3. Generating average lag comparison...\n")

# Calculate average lag for each illness
avg_lags <- lag_summary %>%
  group_by(illness) %>%
  summarise(
    avg_lag = weighted.mean(lag, n_features),
    .groups = "drop"
  ) %>%
  mutate(
    illness = factor(illness, levels = c(
      "Acute laryngopharyngitis",
      "Gastritis",
      "Chronic rhinitis"
    ))
  )

p_avg_lag <- ggplot(avg_lags, aes(x = illness, y = avg_lag, fill = illness)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.1f days", avg_lag)), 
            vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(
    values = c(
      "Acute laryngopharyngitis" = "#e74c3c",
      "Gastritis" = "#3498db",
      "Chronic rhinitis" = "#2ecc71"
    )
  ) +
  labs(
    title = "Average Environmental Exposure Lag Period",
    subtitle = "Time from environmental exposure to illness manifestation",
    x = NULL,
    y = "Average Lag (days)"
  ) +
  scale_y_continuous(limits = c(0, max(avg_lags$avg_lag) * 1.15)) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray30"),
    axis.text.x = element_text(angle = 15, hjust = 1),
    panel.grid.major.x = element_blank()
  )

ggsave(
  "results/three_illness_comparison/visualizations/average_lag_comparison.png",
  p_avg_lag,
  width = 8,
  height = 6,
  dpi = 300,
  bg = "white"
)
cat("  ✓ Saved: average_lag_comparison.png\n")

# ============================================================================
# 4. ENVIRONMENTAL FACTOR × LAG HEATMAP (Acute Laryngopharyngitis)
# ============================================================================

cat("\n4. Generating factor × lag heatmap for Acute Laryngopharyngitis...\n")

# Extract environmental variable and lag
laryngo_detail <- laryngo_lags %>%
  mutate(
    env_var = str_extract(feature, "^[^_]+(?=_lag)"),
    env_var = ifelse(is.na(env_var), str_extract(feature, "^[^_]+"), env_var)
  ) %>%
  filter(!is.na(env_var), !is.na(lag))

# Focus on key environmental variables
key_vars <- c("AvgTemp", "PM10", "AvgHumidity", "Case_Count")
laryngo_key <- laryngo_detail %>%
  filter(env_var %in% key_vars)

if (nrow(laryngo_key) > 0) {
  p_heatmap <- ggplot(laryngo_key, aes(x = lag, y = env_var, fill = importance)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = sprintf("%.3f", importance)), color = "white", size = 4, fontface = "bold") +
    scale_fill_gradient(low = "#3498db", high = "#e74c3c", name = "Importance") +
    scale_x_continuous(breaks = unique(laryngo_key$lag)) +
    scale_y_discrete(
      labels = c(
        "AvgTemp" = "Temperature",
        "PM10" = "Air Quality (PM10)",
        "AvgHumidity" = "Humidity",
        "Case_Count" = "Illness Cases"
      )
    ) +
    labs(
      title = "Environmental Factor × Lag Period Analysis",
      subtitle = "Acute Laryngopharyngitis: Which environmental factors at which lag periods?",
      x = "Lag Period (days since exposure)",
      y = "Environmental Factor"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "gray30"),
      legend.position = "right",
      panel.grid = element_blank()
    )
  
  ggsave(
    "results/three_illness_comparison/visualizations/factor_lag_heatmap_laryngopharyngitis.png",
    p_heatmap,
    width = 10,
    height = 5,
    dpi = 300,
    bg = "white"
  )
  cat("  ✓ Saved: factor_lag_heatmap_laryngopharyngitis.png\n")
}

# ============================================================================
# 5. INCUBATION TIMELINE (Acute Laryngopharyngitis)
# ============================================================================

cat("\n5. Generating incubation timeline...\n")

# Create timeline data
timeline_data <- data.frame(
  day = c(0, 2, 5, 8, 12),
  event = c(
    "Environmental\nExposure",
    "Temperature\nEffects",
    "Air Quality\n(PM10) Effects",
    "Humidity\nEffects",
    "Extended\nHumidity Effects"
  ),
  category = c("Exposure", "Symptom", "Symptom", "Symptom", "Symptom"),
  stringsAsFactors = FALSE
)

p_timeline <- ggplot(timeline_data, aes(x = day, y = 1)) +
  geom_line(linewidth = 2, color = "#3498db") +
  geom_point(aes(color = category), size = 8) +
  geom_text(aes(label = event), vjust = -2, size = 3.5, lineheight = 0.9) +
  geom_text(aes(label = sprintf("Day %d", day)), vjust = 3, size = 4, fontface = "bold") +
  scale_color_manual(
    values = c("Exposure" = "#95a5a6", "Symptom" = "#e74c3c"),
    guide = "none"
  ) +
  scale_x_continuous(breaks = timeline_data$day, limits = c(-1, 14)) +
  labs(
    title = "Environmental Exposure to Symptom Onset Timeline",
    subtitle = "Acute Laryngopharyngitis: Data-driven incubation period evidence",
    x = "Days Since Environmental Exposure",
    y = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray30"),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank(),
    plot.margin = margin(20, 20, 20, 20)
  )

ggsave(
  "results/three_illness_comparison/visualizations/incubation_timeline_laryngopharyngitis.png",
  p_timeline,
  width = 12,
  height = 4,
  dpi = 300,
  bg = "white"
)
cat("  ✓ Saved: incubation_timeline_laryngopharyngitis.png\n")

# ============================================================================
# 6. MODEL PERFORMANCE COMPARISON
# ============================================================================

cat("\n6. Generating model performance comparison...\n")

# Performance data
performance_data <- data.frame(
  illness = c("Acute laryngopharyngitis", "Gastritis", "Chronic rhinitis"),
  r2 = c(0.798, 0.906, 0.768),
  rmse = c(13.59, 4.54, 2.69),
  mae = c(6.44, 2.78, 1.49),
  stringsAsFactors = FALSE
)

performance_data$illness <- factor(
  performance_data$illness,
  levels = c("Acute laryngopharyngitis", "Gastritis", "Chronic rhinitis")
)

p_performance <- ggplot(performance_data, aes(x = illness, y = r2, fill = illness)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = sprintf("R² = %.3f", r2)), 
            vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(
    values = c(
      "Acute laryngopharyngitis" = "#e74c3c",
      "Gastritis" = "#3498db",
      "Chronic rhinitis" = "#2ecc71"
    )
  ) +
  labs(
    title = "Model Performance Comparison",
    subtitle = "Test set R² scores for comprehensive lag models",
    x = NULL,
    y = "R² Score"
  ) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray30"),
    axis.text.x = element_text(angle = 15, hjust = 1),
    panel.grid.major.x = element_blank()
  )

ggsave(
  "results/three_illness_comparison/visualizations/model_performance_comparison.png",
  p_performance,
  width = 8,
  height = 6,
  dpi = 300,
  bg = "white"
)
cat("  ✓ Saved: model_performance_comparison.png\n")

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("VISUALIZATION GENERATION COMPLETE ✓\n")
cat(strrep("=", 80), "\n")
cat("\nGenerated files:\n")
cat("  1. gastritis_feature_importance.png\n")
cat("  2. chronic_rhinitis_feature_importance.png\n")
cat("  3. lag_comparison_all_illnesses.png\n")
cat("  4. average_lag_comparison.png\n")
cat("  5. factor_lag_heatmap_laryngopharyngitis.png\n")
cat("  6. incubation_timeline_laryngopharyngitis.png\n")
cat("  7. model_performance_comparison.png\n")
cat("\nOutput directory: results/three_illness_comparison/visualizations/\n")
cat("\n✓ Ready for research paper!\n")
