#!/usr/bin/env Rscript
# SHAP Analysis Visualization
# Generates comprehensive ggplot2 visualizations of SHAP results

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)

# Set paths
project_root <- "/Users/jay/Desktop/Illness Prediction"
shap_dir <- file.path(project_root, "results/shap_analysis")
output_dir <- file.path(project_root, "results/optimization_visualizations")

cat(strrep("=", 80), "\n")
cat("GENERATING SHAP VISUALIZATIONS\n")
cat(strrep("=", 80), "\n\n")

# White background theme
theme_white <- theme(
  plot.background = element_rect(fill = "white", color = NA),
  panel.background = element_rect(fill = "white")
)

# Define illnesses
illnesses <- data.frame(
  safe_name = c("Acute_laryngopharyngitis", "Gastritis_unspecified", "Chronic_rhinitis"),
  full_name = c("Acute laryngopharyngitis", "Gastritis, unspecified", "Chronic rhinitis"),
  stringsAsFactors = FALSE
)

# ============================================================================
# PLOT 1: SHAP Feature Importance (Top 20) - Directional colored
# ============================================================================

cat("1. Creating SHAP feature importance plots (directional)...\n")

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  shap_file <- file.path(shap_dir, safe_name, "shap_values_summary.csv")
  if (!file.exists(shap_file)) {
    cat(sprintf("   ⚠️  Skipping %s - file not found\n", full_name))
    next
  }
  
  shap_df <- read_csv(shap_file, show_col_types = FALSE) %>%
    head(20)
  
  # Create plot with directional coloring
  p <- ggplot(shap_df, aes(x = reorder(Feature, Mean_Abs_SHAP), y = Mean_Abs_SHAP, fill = Direction)) +
    geom_bar(stat = "identity", width = 0.7) +
    geom_text(aes(label = sprintf("%.3f", Mean_Abs_SHAP)), 
              hjust = -0.1, size = 3) +
    coord_flip() +
    scale_fill_manual(
      values = c("Increase" = "#e74c3c", "Decrease" = "#3498db"),
      labels = c("Increase Illness", "Decrease Illness")
    ) +
    labs(
      title = paste("SHAP Feature Importance:", full_name),
      subtitle = "Top 20 features with directional effect on illness cases",
      x = "Feature",
      y = "Mean |SHAP Value|",
      fill = "Effect Direction"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray30"),
      axis.text.y = element_text(size = 9),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "top"
    ) +
    theme_white +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
  
  filename <- file.path(output_dir, paste0(safe_name, "_shap_importance_directional.png"))
  ggsave(filename, p, width = 12, height = 10, dpi = 300, bg = "white")
  cat(sprintf("   ✓ Saved: %s_shap_importance_directional.png\n", safe_name))
}

# ============================================================================
# PLOT 2: Direction Summary - Increase vs Decrease Features
# ============================================================================

cat("2. Creating direction summary plot...\n")

direction_data <- data.frame()

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  dir_file <- file.path(shap_dir, safe_name, "shap_direction_summary.csv")
  if (file.exists(dir_file)) {
    df <- read_csv(dir_file, show_col_types = FALSE)
    df$Illness <- full_name
    direction_data <- rbind(direction_data, df)
  }
}

if (nrow(direction_data) > 0) {
  p2 <- ggplot(direction_data, aes(x = Illness, y = Count, fill = Direction)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_text(aes(label = Count), 
              position = position_dodge(width = 0.7), 
              vjust = -0.5, size = 4) +
    scale_fill_manual(
      values = c("Increase" = "#e74c3c", "Decrease" = "#3498db"),
      labels = c("Increase Illness", "Decrease Illness")
    ) +
    labs(
      title = "SHAP Directional Feature Count",
      subtitle = "Number of features that increase vs decrease illness cases",
      x = "Illness",
      y = "Number of Features",
      fill = "Effect Direction"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray30"),
      axis.text.x = element_text(angle = 15, hjust = 1),
      legend.position = "top",
      panel.grid.major.x = element_blank()
    ) +
    theme_white
  
  ggsave(file.path(output_dir, "shap_direction_summary.png"), 
         p2, width = 10, height = 6, dpi = 300, bg = "white")
  cat("   ✓ Saved: shap_direction_summary.png\n")
}

# ============================================================================
# PLOT 3: Category-wise SHAP Importance
# ============================================================================

cat("3. Creating category SHAP importance plots...\n")

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  cat_file <- file.path(shap_dir, safe_name, "shap_category_summary.csv")
  if (!file.exists(cat_file)) {
    next
  }
  
  cat_df <- read_csv(cat_file, show_col_types = FALSE)
  cat_df$Category <- rownames(cat_df)
  if ("...1" %in% colnames(cat_df)) {
    cat_df$Category <- cat_df$...1
  }
  
  p <- ggplot(cat_df, aes(x = reorder(Category, Mean_Abs_SHAP), y = Mean_Abs_SHAP)) +
    geom_bar(stat = "identity", fill = "#9b59b6", width = 0.6) +
    geom_text(aes(label = sprintf("%.2f", Mean_Abs_SHAP)), 
              hjust = -0.1, size = 3.5) +
    coord_flip() +
    labs(
      title = paste("SHAP Importance by Environmental Category:", full_name),
      subtitle = "Total SHAP contribution by category",
      x = "Environmental Category",
      y = "Total Mean |SHAP Value|"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray30"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank()
    ) +
    theme_white +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
  
  filename <- file.path(output_dir, paste0(safe_name, "_shap_category.png"))
  ggsave(filename, p, width = 10, height = 6, dpi = 300, bg = "white")
  cat(sprintf("   ✓ Saved: %s_shap_category.png\n", safe_name))
}

# ============================================================================
# PLOT 4: Lag vs Base SHAP Importance
# ============================================================================

cat("4. Creating lag vs base SHAP importance plot...\n")

lag_data <- data.frame()

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  lag_file <- file.path(shap_dir, safe_name, "shap_lag_vs_base.csv")
  if (file.exists(lag_file)) {
    df <- read_csv(lag_file, show_col_types = FALSE)
    if ("...1" %in% colnames(df)) {
      df <- df %>% rename(Type = ...1)
    }
    df$Illness <- full_name
    lag_data <- rbind(lag_data, df)
  }
}

if (nrow(lag_data) > 0) {
  p4 <- ggplot(lag_data, aes(x = Illness, y = Mean_Abs_SHAP, fill = Type)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_text(aes(label = sprintf("%.2f", Mean_Abs_SHAP)), 
              position = position_dodge(width = 0.7), 
              vjust = -0.5, size = 3.5) +
    scale_fill_manual(values = c("Base Features" = "#2ecc71", "Lag Features" = "#f39c12")) +
    labs(
      title = "SHAP: Base vs Lag Feature Importance",
      subtitle = "Average SHAP importance for base vs lagged environmental features",
      x = "Illness",
      y = "Mean |SHAP Value|",
      fill = "Feature Type"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray30"),
      axis.text.x = element_text(angle = 15, hjust = 1),
      legend.position = "top",
      panel.grid.major.x = element_blank()
    ) +
    theme_white
  
  ggsave(file.path(output_dir, "shap_lag_vs_base.png"), 
         p4, width = 10, height = 6, dpi = 300, bg = "white")
  cat("   ✓ Saved: shap_lag_vs_base.png\n")
}

# ============================================================================
# PLOT 5: Mean SHAP Values (Directional) - Top 15
# ============================================================================

cat("5. Creating mean SHAP value plots (directional)...\n")

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  shap_file <- file.path(shap_dir, safe_name, "shap_values_summary.csv")
  if (!file.exists(shap_file)) {
    next
  }
  
  shap_df <- read_csv(shap_file, show_col_types = FALSE) %>%
    arrange(desc(Abs_Mean_SHAP)) %>%
    head(15)
  
  # Create plot with mean SHAP (positive/negative)
  p <- ggplot(shap_df, aes(x = reorder(Feature, Mean_SHAP), y = Mean_SHAP, fill = Direction)) +
    geom_bar(stat = "identity", width = 0.7) +
    geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.3f", Mean_SHAP)), 
              hjust = ifelse(shap_df$Mean_SHAP > 0, -0.1, 1.1), size = 3) +
    coord_flip() +
    scale_fill_manual(
      values = c("Increase" = "#e74c3c", "Decrease" = "#3498db"),
      labels = c("Increase Illness", "Decrease Illness")
    ) +
    labs(
      title = paste("Mean SHAP Values:", full_name),
      subtitle = "Average directional effect of top 15 features on illness prediction",
      x = "Feature",
      y = "Mean SHAP Value",
      fill = "Effect Direction"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray30"),
      axis.text.y = element_text(size = 9),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "top"
    ) +
    theme_white +
    scale_y_continuous(expand = expansion(mult = c(0.15, 0.15)))
  
  filename <- file.path(output_dir, paste0(safe_name, "_shap_mean_values.png"))
  ggsave(filename, p, width = 12, height = 8, dpi = 300, bg = "white")
  cat(sprintf("   ✓ Saved: %s_shap_mean_values.png\n", safe_name))
}

# ============================================================================
# PLOT 6: Cross-Illness Top Features Comparison (SHAP)
# ============================================================================

cat("6. Creating cross-illness SHAP comparison...\n")

all_shap <- data.frame()

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  shap_file <- file.path(shap_dir, safe_name, "shap_values_summary.csv")
  if (file.exists(shap_file)) {
    df <- read_csv(shap_file, show_col_types = FALSE)
    df$Illness <- full_name
    all_shap <- rbind(all_shap, df)
  }
}

if (nrow(all_shap) > 0) {
  # Get top 10 features by average SHAP
  top_features <- all_shap %>%
    group_by(Feature) %>%
    summarise(Avg_SHAP = mean(Mean_Abs_SHAP)) %>%
    arrange(desc(Avg_SHAP)) %>%
    head(10) %>%
    pull(Feature)
  
  top_shap_data <- all_shap %>%
    filter(Feature %in% top_features)
  
  p6 <- ggplot(top_shap_data, aes(x = reorder(Feature, Mean_Abs_SHAP), 
                                   y = Mean_Abs_SHAP, fill = Illness)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    coord_flip() +
    scale_fill_manual(values = c(
      "Acute laryngopharyngitis" = "#e74c3c",
      "Gastritis, unspecified" = "#3498db",
      "Chronic rhinitis" = "#2ecc71"
    )) +
    labs(
      title = "Top 10 SHAP Features Across All Illnesses",
      subtitle = "Comparison of SHAP importance by illness type",
      x = "Feature",
      y = "Mean |SHAP Value|",
      fill = "Illness"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray30"),
      legend.position = "right",
      panel.grid.major.y = element_blank()
    ) +
    theme_white
  
  ggsave(file.path(output_dir, "shap_cross_illness_top10.png"), 
         p6, width = 12, height = 8, dpi = 300, bg = "white")
  cat("   ✓ Saved: shap_cross_illness_top10.png\n")
}

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n", strrep("=", 80), "\n")
cat("SHAP VISUALIZATION COMPLETE\n")
cat(strrep("=", 80), "\n")
cat("\nGenerated visualizations saved to:\n")
cat(output_dir, "\n\n")

cat("Files created:\n")
cat("  1. [illness]_shap_importance_directional.png - Top 20 features with direction (3 files)\n")
cat("  2. shap_direction_summary.png - Increase vs decrease feature counts\n")
cat("  3. [illness]_shap_category.png - Category importance (3 files)\n")
cat("  4. shap_lag_vs_base.png - Base vs lag SHAP comparison\n")
cat("  5. [illness]_shap_mean_values.png - Mean SHAP values (directional, 3 files)\n")
cat("  6. shap_cross_illness_top10.png - Cross-illness comparison\n")
cat("\nTotal: ~13 SHAP visualizations\n")
cat(strrep("=", 80), "\n")
