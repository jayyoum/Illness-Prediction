#!/usr/bin/env Rscript
# SHAP Violin Plot Visualizations
# Shows distribution of SHAP values across samples for deeper insights

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(ggridges)

# Set paths
project_root <- "/Users/jay/Desktop/Illness Prediction"
shap_dir <- file.path(project_root, "results/shap_analysis")
output_dir <- file.path(project_root, "results/optimization_visualizations")

cat(strrep("=", 80), "\n")
cat("GENERATING SHAP VIOLIN PLOT VISUALIZATIONS\n")
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
# PLOT 1: Violin Plots - Top 15 Features with Distribution
# ============================================================================

cat("1. Creating SHAP value distribution violin plots (top 15)...\n")

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  # Load detailed SHAP values
  detail_file <- file.path(shap_dir, safe_name, "shap_values_detailed_top30.csv")
  summary_file <- file.path(shap_dir, safe_name, "shap_values_summary.csv")
  
  if (!file.exists(detail_file) || !file.exists(summary_file)) {
    cat(sprintf("   ⚠️  Skipping %s - files not found\n", full_name))
    next
  }
  
  # Load data
  shap_detail <- read_csv(detail_file, show_col_types = FALSE)
  shap_summary <- read_csv(summary_file, show_col_types = FALSE)
  
  # Get top 15 features by mean absolute SHAP
  top_features <- shap_summary %>%
    head(15) %>%
    pull(Feature)
  
  # Reshape to long format
  shap_long <- shap_detail %>%
    select(all_of(top_features)) %>%
    pivot_longer(cols = everything(), names_to = "Feature", values_to = "SHAP_Value")
  
  # Add direction based on mean
  shap_long <- shap_long %>%
    group_by(Feature) %>%
    mutate(
      Mean_SHAP = mean(SHAP_Value),
      Direction = ifelse(Mean_SHAP > 0, "Increase", "Decrease")
    ) %>%
    ungroup()
  
  # Order by mean absolute SHAP
  feature_order <- shap_summary %>%
    filter(Feature %in% top_features) %>%
    arrange(desc(Mean_Abs_SHAP)) %>%
    pull(Feature)
  
  shap_long$Feature <- factor(shap_long$Feature, levels = feature_order)
  
  # Create violin plot
  p <- ggplot(shap_long, aes(x = Feature, y = SHAP_Value, fill = Direction)) +
    geom_violin(alpha = 0.7, scale = "width", trim = FALSE) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
    stat_summary(fun = median, geom = "point", size = 2, color = "black") +
    coord_flip() +
    scale_fill_manual(
      values = c("Increase" = "#e74c3c", "Decrease" = "#3498db"),
      labels = c("Increase Illness", "Decrease Illness")
    ) +
    labs(
      title = paste("SHAP Value Distribution:", full_name),
      subtitle = "Top 15 features - Violin plots show distribution across samples (black dot = median)",
      x = "Feature",
      y = "SHAP Value",
      fill = "Mean Direction"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 9, hjust = 0.5, color = "gray30"),
      axis.text.y = element_text(size = 9),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "top"
    ) +
    theme_white
  
  filename <- file.path(output_dir, paste0(safe_name, "_shap_violin.png"))
  ggsave(filename, p, width = 12, height = 10, dpi = 300, bg = "white")
  cat(sprintf("   ✓ Saved: %s_shap_violin.png\n", safe_name))
}

# ============================================================================
# PLOT 2: Violin + Box Plot - Top 10 Features
# ============================================================================

cat("2. Creating violin + box combo plots (top 10)...\n")

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  detail_file <- file.path(shap_dir, safe_name, "shap_values_detailed_top30.csv")
  summary_file <- file.path(shap_dir, safe_name, "shap_values_summary.csv")
  
  if (!file.exists(detail_file) || !file.exists(summary_file)) {
    next
  }
  
  shap_detail <- read_csv(detail_file, show_col_types = FALSE)
  shap_summary <- read_csv(summary_file, show_col_types = FALSE)
  
  # Get top 10 features
  top_features <- shap_summary %>%
    head(10) %>%
    pull(Feature)
  
  # Reshape to long format
  shap_long <- shap_detail %>%
    select(all_of(top_features)) %>%
    pivot_longer(cols = everything(), names_to = "Feature", values_to = "SHAP_Value")
  
  # Add direction
  shap_long <- shap_long %>%
    group_by(Feature) %>%
    mutate(
      Mean_SHAP = mean(SHAP_Value),
      Direction = ifelse(Mean_SHAP > 0, "Increase", "Decrease")
    ) %>%
    ungroup()
  
  # Order features
  feature_order <- shap_summary %>%
    filter(Feature %in% top_features) %>%
    arrange(desc(Mean_Abs_SHAP)) %>%
    pull(Feature)
  
  shap_long$Feature <- factor(shap_long$Feature, levels = feature_order)
  
  # Create violin + box plot
  p <- ggplot(shap_long, aes(x = Feature, y = SHAP_Value, fill = Direction)) +
    geom_violin(alpha = 0.5, scale = "width", trim = FALSE) +
    geom_boxplot(width = 0.15, alpha = 0.8, outlier.size = 0.5, 
                 position = position_dodge(width = 0.9)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
    coord_flip() +
    scale_fill_manual(
      values = c("Increase" = "#e74c3c", "Decrease" = "#3498db"),
      labels = c("Increase Illness", "Decrease Illness")
    ) +
    labs(
      title = paste("SHAP Distribution with Quartiles:", full_name),
      subtitle = "Top 10 features - Violin shows distribution, box shows quartiles and outliers",
      x = "Feature",
      y = "SHAP Value",
      fill = "Mean Direction"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 9, hjust = 0.5, color = "gray30"),
      axis.text.y = element_text(size = 9),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "top"
    ) +
    theme_white
  
  filename <- file.path(output_dir, paste0(safe_name, "_shap_violin_box.png"))
  ggsave(filename, p, width = 12, height = 8, dpi = 300, bg = "white")
  cat(sprintf("   ✓ Saved: %s_shap_violin_box.png\n", safe_name))
}

# ============================================================================
# PLOT 3: Density Ridge Plot - Top 12 Features (Alternative visualization)
# ============================================================================

cat("3. Creating density ridge plots (top 12)...\n")

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  detail_file <- file.path(shap_dir, safe_name, "shap_values_detailed_top30.csv")
  summary_file <- file.path(shap_dir, safe_name, "shap_values_summary.csv")
  
  if (!file.exists(detail_file) || !file.exists(summary_file)) {
    next
  }
  
  shap_detail <- read_csv(detail_file, show_col_types = FALSE)
  shap_summary <- read_csv(summary_file, show_col_types = FALSE)
  
  # Get top 12 features
  top_features <- shap_summary %>%
    head(12) %>%
    pull(Feature)
  
  # Reshape
  shap_long <- shap_detail %>%
    select(all_of(top_features)) %>%
    pivot_longer(cols = everything(), names_to = "Feature", values_to = "SHAP_Value")
  
  # Add direction and mean
  shap_long <- shap_long %>%
    left_join(
      shap_summary %>% select(Feature, Mean_SHAP, Direction),
      by = "Feature"
    )
  
  # Order features
  feature_order <- shap_summary %>%
    filter(Feature %in% top_features) %>%
    arrange(desc(Mean_Abs_SHAP)) %>%
    pull(Feature)
  
  shap_long$Feature <- factor(shap_long$Feature, levels = rev(feature_order))
  
  # Create density plot
  p <- ggplot(shap_long, aes(x = SHAP_Value, y = Feature, fill = Direction)) +
    geom_density_ridges(
      alpha = 0.7, 
      scale = 0.9,
      rel_min_height = 0.01
    ) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
    scale_fill_manual(
      values = c("Increase" = "#e74c3c", "Decrease" = "#3498db"),
      labels = c("Increase Illness", "Decrease Illness")
    ) +
    labs(
      title = paste("SHAP Value Density Distribution:", full_name),
      subtitle = "Top 12 features - Ridge plot shows probability density of SHAP values",
      x = "SHAP Value",
      y = "Feature",
      fill = "Mean Direction"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 9, hjust = 0.5, color = "gray30"),
      axis.text.y = element_text(size = 9),
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank(),
      legend.position = "top"
    ) +
    theme_white
  
  filename <- file.path(output_dir, paste0(safe_name, "_shap_density_ridge.png"))
  ggsave(filename, p, width = 12, height = 8, dpi = 300, bg = "white")
  cat(sprintf("   ✓ Saved: %s_shap_density_ridge.png\n", safe_name))
}

# ============================================================================
# PLOT 4: Violin Plot - Cross-Illness Comparison (Top 8 Features)
# ============================================================================

cat("4. Creating cross-illness violin comparison...\n")

all_shap_detail <- data.frame()

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  detail_file <- file.path(shap_dir, safe_name, "shap_values_detailed_top30.csv")
  
  if (file.exists(detail_file)) {
    shap_detail <- read_csv(detail_file, show_col_types = FALSE)
    
    # Reshape to long
    shap_long <- shap_detail %>%
      pivot_longer(cols = everything(), names_to = "Feature", values_to = "SHAP_Value") %>%
      mutate(Illness = full_name)
    
    all_shap_detail <- rbind(all_shap_detail, shap_long)
  }
}

if (nrow(all_shap_detail) > 0) {
  # Get top 8 features across all illnesses
  top_features <- all_shap_detail %>%
    group_by(Feature) %>%
    summarise(Mean_Abs_SHAP = mean(abs(SHAP_Value))) %>%
    arrange(desc(Mean_Abs_SHAP)) %>%
    head(8) %>%
    pull(Feature)
  
  # Filter to top features
  plot_data <- all_shap_detail %>%
    filter(Feature %in% top_features)
  
  # Order features
  feature_order <- all_shap_detail %>%
    filter(Feature %in% top_features) %>%
    group_by(Feature) %>%
    summarise(Mean_Abs_SHAP = mean(abs(SHAP_Value))) %>%
    arrange(desc(Mean_Abs_SHAP)) %>%
    pull(Feature)
  
  plot_data$Feature <- factor(plot_data$Feature, levels = feature_order)
  
  # Create plot
  p4 <- ggplot(plot_data, aes(x = Feature, y = SHAP_Value, fill = Illness)) +
    geom_violin(alpha = 0.7, scale = "width", trim = FALSE, 
                position = position_dodge(width = 0.9)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
    stat_summary(fun = median, geom = "point", size = 1.5, color = "black",
                 position = position_dodge(width = 0.9)) +
    coord_flip() +
    scale_fill_manual(values = c(
      "Acute laryngopharyngitis" = "#e74c3c",
      "Gastritis, unspecified" = "#3498db",
      "Chronic rhinitis" = "#2ecc71"
    )) +
    labs(
      title = "SHAP Distribution Across Illnesses: Top 8 Features",
      subtitle = "Violin plots compare SHAP value distributions (black dot = median)",
      x = "Feature",
      y = "SHAP Value",
      fill = "Illness"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 9, hjust = 0.5, color = "gray30"),
      axis.text.y = element_text(size = 9),
      panel.grid.major.y = element_blank(),
      legend.position = "top"
    ) +
    theme_white
  
  ggsave(file.path(output_dir, "shap_violin_cross_illness.png"), 
         p4, width = 12, height = 8, dpi = 300, bg = "white")
  cat("   ✓ Saved: shap_violin_cross_illness.png\n")
}

# ============================================================================
# PLOT 5: Jitter + Violin for Extreme Effects (Top 10)
# ============================================================================

cat("5. Creating jitter + violin plots (showing individual samples)...\n")

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  detail_file <- file.path(shap_dir, safe_name, "shap_values_detailed_top30.csv")
  summary_file <- file.path(shap_dir, safe_name, "shap_values_summary.csv")
  
  if (!file.exists(detail_file) || !file.exists(summary_file)) {
    next
  }
  
  shap_detail <- read_csv(detail_file, show_col_types = FALSE)
  shap_summary <- read_csv(summary_file, show_col_types = FALSE)
  
  # Get top 10
  top_features <- shap_summary %>%
    head(10) %>%
    pull(Feature)
  
  shap_long <- shap_detail %>%
    select(all_of(top_features)) %>%
    pivot_longer(cols = everything(), names_to = "Feature", values_to = "SHAP_Value")
  
  # Add direction
  shap_long <- shap_long %>%
    group_by(Feature) %>%
    mutate(
      Mean_SHAP = mean(SHAP_Value),
      Direction = ifelse(Mean_SHAP > 0, "Increase", "Decrease")
    ) %>%
    ungroup()
  
  feature_order <- shap_summary %>%
    filter(Feature %in% top_features) %>%
    arrange(desc(Mean_Abs_SHAP)) %>%
    pull(Feature)
  
  shap_long$Feature <- factor(shap_long$Feature, levels = feature_order)
  
  # Create violin + jitter
  p <- ggplot(shap_long, aes(x = Feature, y = SHAP_Value, color = Direction, fill = Direction)) +
    geom_violin(alpha = 0.3, scale = "width", trim = FALSE) +
    geom_jitter(alpha = 0.2, size = 0.5, width = 0.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
    stat_summary(fun = mean, geom = "point", size = 3, color = "black", shape = 23, fill = "yellow") +
    coord_flip() +
    scale_color_manual(
      values = c("Increase" = "#e74c3c", "Decrease" = "#3498db"),
      labels = c("Increase Illness", "Decrease Illness")
    ) +
    scale_fill_manual(
      values = c("Increase" = "#e74c3c", "Decrease" = "#3498db"),
      labels = c("Increase Illness", "Decrease Illness")
    ) +
    labs(
      title = paste("SHAP Sample-Level Variation:", full_name),
      subtitle = "Top 10 features - Individual samples (dots), distribution (violin), mean (yellow diamond)",
      x = "Feature",
      y = "SHAP Value",
      color = "Mean Direction",
      fill = "Mean Direction"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 9, hjust = 0.5, color = "gray30"),
      axis.text.y = element_text(size = 9),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "top"
    ) +
    theme_white
  
  filename <- file.path(output_dir, paste0(safe_name, "_shap_violin_jitter.png"))
  ggsave(filename, p, width = 12, height = 8, dpi = 300, bg = "white")
  cat(sprintf("   ✓ Saved: %s_shap_violin_jitter.png\n", safe_name))
}

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n", strrep("=", 80), "\n")
cat("SHAP VIOLIN VISUALIZATION COMPLETE\n")
cat(strrep("=", 80), "\n")
cat("\nGenerated visualizations saved to:\n")
cat(output_dir, "\n\n")

cat("Files created:\n")
cat("  1. [illness]_shap_violin.png - Distribution violin plots (top 15, 3 files)\n")
cat("  2. [illness]_shap_violin_box.png - Violin + box combo (top 10, 3 files)\n")
cat("  3. [illness]_shap_density_ridge.png - Density ridge plots (top 12, 3 files)\n")
cat("  4. shap_violin_cross_illness.png - Cross-illness comparison (top 8)\n")
cat("  5. [illness]_shap_violin_jitter.png - Individual samples + violin (top 10, 3 files)\n")
cat("\nTotal: ~13 violin plot visualizations\n")
cat(strrep("=", 80), "\n")
