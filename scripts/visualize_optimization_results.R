#!/usr/bin/env Rscript
# Comprehensive Visualization of Advanced Optimization Results
# Generates publication-quality ggplot2 visualizations

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(scales)

# Set paths
project_root <- "/Users/jay/Desktop/Illness Prediction"
results_dir <- file.path(project_root, "results/advanced_optimization")
gridsearch_dir <- file.path(project_root, "results/xgb_gridsearch_lagged")
lag_analysis_dir <- file.path(project_root, "results/lag_vs_nonlag_analysis")
output_dir <- file.path(project_root, "results/optimization_visualizations")

# Create output directory
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat(strrep("=", 80), "\n")
cat("GENERATING OPTIMIZATION VISUALIZATIONS\n")
cat(strrep("=", 80), "\n\n")

# White background for all plots
theme_white <- theme(
  plot.background = element_rect(fill = "white", color = NA),
  panel.background = element_rect(fill = "white")
)

# Normalize feature importance to sum to 1 when stored in raw scale (e.g. XGBoost gain)
normalize_importance <- function(imp_df) {
  total <- sum(imp_df$Importance, na.rm = TRUE)
  if (total > 1.5) {
    imp_df$Importance <- imp_df$Importance / total
  }
  imp_df
}

# Define illnesses
illnesses <- data.frame(
  safe_name = c("Acute_laryngopharyngitis", "Gastritis_unspecified", "Chronic_rhinitis"),
  full_name = c("Acute laryngopharyngitis", "Gastritis, unspecified", "Chronic rhinitis"),
  stringsAsFactors = FALSE
)

# ============================================================================
# PLOT 1: Performance Comparison - GridSearch vs Optuna (CV R²)
# ============================================================================

cat("1. Creating performance comparison plot (CV R²)...\n")

perf_data <- data.frame()

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  # GridSearch
  grid_file <- file.path(gridsearch_dir, safe_name, paste0("xgb_gridsearch_", safe_name, "_summary.csv"))
  if (file.exists(grid_file)) {
    grid_df <- read_csv(grid_file, show_col_types = FALSE)
    perf_data <- rbind(perf_data, data.frame(
      Illness = full_name,
      Method = "GridSearch",
      CV_R2 = grid_df$Best_R2_CV[1],
      RMSE = grid_df$Final_RMSE[1]
    ))
  }
  
  # Optuna
  optuna_file <- file.path(results_dir, safe_name, "best_model_summary.csv")
  if (file.exists(optuna_file)) {
    optuna_df <- read_csv(optuna_file, show_col_types = FALSE)
    perf_data <- rbind(perf_data, data.frame(
      Illness = full_name,
      Method = "Optuna",
      CV_R2 = optuna_df$CV_R2[1],
      RMSE = optuna_df$RMSE[1]
    ))
  }
}

# Plot CV R² comparison
p1 <- ggplot(perf_data, aes(x = Illness, y = CV_R2, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", CV_R2)), 
            position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("GridSearch" = "#3498db", "Optuna" = "#e74c3c")) +
  labs(
    title = "Model Performance Comparison: GridSearch vs Optuna",
    subtitle = "Cross-Validation R² (higher is better)",
    x = "Illness",
    y = "Cross-Validation R²",
    fill = "Optimization Method"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray30"),
    axis.text.x = element_text(angle = 15, hjust = 1),
    legend.position = "top",
    panel.grid.major.x = element_blank()
  ) +
  theme_white +
  scale_y_continuous(limits = c(0, 0.65), breaks = seq(0, 0.6, 0.1))

ggsave(file.path(output_dir, "performance_comparison_cv_r2.png"), 
       p1, width = 10, height = 6, dpi = 300, bg = "white")
cat("   ✓ Saved: performance_comparison_cv_r2.png\n")

# ============================================================================
# PLOT 2: RMSE Comparison
# ============================================================================

cat("2. Creating RMSE comparison plot...\n")

p2 <- ggplot(perf_data, aes(x = Illness, y = RMSE, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = sprintf("%.2f", RMSE)), 
            position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("GridSearch" = "#3498db", "Optuna" = "#e74c3c")) +
  labs(
    title = "RMSE Comparison: GridSearch vs Optuna",
    subtitle = "Root Mean Squared Error (lower is better)",
    x = "Illness",
    y = "RMSE",
    fill = "Optimization Method"
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

ggsave(file.path(output_dir, "rmse_comparison.png"), 
       p2, width = 10, height = 6, dpi = 300, bg = "white")
cat("   ✓ Saved: rmse_comparison.png\n")

# ============================================================================
# PLOT 3: Improvement Percentage
# ============================================================================

cat("3. Creating improvement percentage plot...\n")

improvement_data <- perf_data %>%
  group_by(Illness) %>%
  summarise(
    GridSearch = CV_R2[Method == "GridSearch"],
    Optuna = CV_R2[Method == "Optuna"],
    Improvement_Pct = ((Optuna - GridSearch) / GridSearch) * 100
  )

p3 <- ggplot(improvement_data, aes(x = reorder(Illness, -Improvement_Pct), y = Improvement_Pct)) +
  geom_bar(stat = "identity", fill = "#27ae60", width = 0.6) +
  geom_text(aes(label = sprintf("+%.1f%%", Improvement_Pct)), 
            vjust = -0.5, size = 4, fontface = "bold") +
  geom_hline(yintercept = 0, linetype = "solid", color = "black", linewidth = 0.3) +
  labs(
    title = "Performance Improvement: Optuna vs GridSearch",
    subtitle = "Percentage increase in Cross-Validation R²",
    x = "Illness",
    y = "Improvement (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray30"),
    axis.text.x = element_text(angle = 15, hjust = 1),
    panel.grid.major.x = element_blank()
  ) +
  theme_white +
  scale_y_continuous(limits = c(0, 30), breaks = seq(0, 30, 5))

ggsave(file.path(output_dir, "improvement_percentage.png"), 
       p3, width = 10, height = 6, dpi = 300, bg = "white")
cat("   ✓ Saved: improvement_percentage.png\n")

# ============================================================================
# PLOT 4: Raw Feature Importance - All Variables Combined (Top 20)
# ============================================================================

cat("4. Creating combined feature importance plots...\n")

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  importance_file <- file.path(results_dir, safe_name, "feature_importance.csv")
  if (!file.exists(importance_file)) {
    cat(sprintf("   ⚠️  Skipping %s - file not found\n", full_name))
    next
  }
  
  imp_df <- read_csv(importance_file, show_col_types = FALSE) %>%
    normalize_importance() %>%
    arrange(desc(Importance)) %>%
    head(20)
  
  # Create plot
  p <- ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "#3498db", width = 0.7) +
    geom_text(aes(label = sprintf("%.4f", Importance)), 
              hjust = -0.1, size = 3) +
    coord_flip() +
    labs(
      title = paste("Feature Importance:", full_name),
      subtitle = "Top 20 Features (Optuna-Optimized XGBoost Model)",
      x = "Feature",
      y = "Importance Score"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray30"),
      axis.text.y = element_text(size = 9),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank()
    ) +
    theme_white +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
  
  filename <- file.path(output_dir, paste0(safe_name, "_feature_importance_top20.png"))
  ggsave(filename, p, width = 10, height = 8, dpi = 300, bg = "white")
  cat(sprintf("   ✓ Saved: %s_feature_importance_top20.png\n", safe_name))
}

# ============================================================================
# PLOT 5: Lag vs Non-Lag Distribution
# ============================================================================

cat("5. Creating lag vs non-lag distribution plot...\n")

lag_summary <- data.frame()

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  analysis_file <- file.path(lag_analysis_dir, paste0(safe_name, "_base_vs_lag_summary.csv"))
  if (file.exists(analysis_file)) {
    df <- read_csv(analysis_file, show_col_types = FALSE)
    df$Illness <- full_name
    lag_summary <- rbind(lag_summary, df)
  }
}

if (nrow(lag_summary) > 0) {
  p5 <- ggplot(lag_summary, aes(x = Illness, y = Pct_Importance, fill = Category)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_text(aes(label = sprintf("%.1f%%", Pct_Importance)), 
              position = position_dodge(width = 0.7), 
              vjust = -0.5, size = 3.5) +
    scale_fill_manual(values = c("Base Features" = "#2ecc71", "Lag Features" = "#f39c12")) +
    labs(
      title = "Base vs Lag Feature Importance Distribution",
      subtitle = "Percentage contribution to total feature importance",
      x = "Illness",
      y = "Importance (%)",
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
    theme_white +
    scale_y_continuous(limits = c(0, 80), breaks = seq(0, 80, 10))
  
  ggsave(file.path(output_dir, "base_vs_lag_distribution.png"), 
         p5, width = 10, height = 6, dpi = 300, bg = "white")
  cat("   ✓ Saved: base_vs_lag_distribution.png\n")
}

# ============================================================================
# PLOT 6: Lag Day Importance Heatmap
# ============================================================================

cat("6. Creating lag day importance heatmap...\n")

lag_day_data <- data.frame()

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  lag_day_file <- file.path(lag_analysis_dir, paste0(safe_name, "_lag_day_importance.csv"))
  if (file.exists(lag_day_file)) {
    df <- read_csv(lag_day_file, show_col_types = FALSE)
    df$Illness <- full_name
    lag_day_data <- rbind(lag_day_data, df)
  }
}

if (nrow(lag_day_data) > 0) {
  p6 <- ggplot(lag_day_data, aes(x = factor(Lag_Day), y = Illness, fill = Total_Importance)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = sprintf("%.3f", Total_Importance)), 
              color = "white", size = 3.5, fontface = "bold") +
    scale_fill_gradient(low = "#3498db", high = "#e74c3c", name = "Importance") +
    labs(
      title = "Lag Day Importance Heatmap",
      subtitle = "Total importance of features for each lag period",
      x = "Lag Day",
      y = "Illness"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray30"),
      axis.text.x = element_text(size = 10),
      axis.text.y = element_text(size = 10),
      legend.position = "right",
      panel.grid = element_blank()
    ) +
    theme_white
  
  ggsave(file.path(output_dir, "lag_day_importance_heatmap.png"), 
         p6, width = 12, height = 5, dpi = 300, bg = "white")
  cat("   ✓ Saved: lag_day_importance_heatmap.png\n")
}

# ============================================================================
# PLOT 7: Feature Count Distribution (Base vs Lag)
# ============================================================================

cat("7. Creating feature count distribution...\n")

if (nrow(lag_summary) > 0) {
  p7 <- ggplot(lag_summary, aes(x = Illness, y = Count, fill = Category)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    geom_text(aes(label = Count), 
              position = position_dodge(width = 0.7), 
              vjust = -0.5, size = 4) +
    scale_fill_manual(values = c("Base Features" = "#2ecc71", "Lag Features" = "#f39c12")) +
    labs(
      title = "Feature Count: Base vs Lag Features",
      subtitle = "Number of features selected by model",
      x = "Illness",
      y = "Number of Features",
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
  
  ggsave(file.path(output_dir, "feature_count_distribution.png"), 
         p7, width = 10, height = 6, dpi = 300, bg = "white")
  cat("   ✓ Saved: feature_count_distribution.png\n")
}

# ============================================================================
# PLOT 8: Variable-Specific Analysis (Base + Lag combined)
# ============================================================================

cat("8. Creating variable-specific importance plots...\n")

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  var_analysis_file <- file.path(lag_analysis_dir, paste0(safe_name, "_variable_analysis.csv"))
  if (!file.exists(var_analysis_file)) {
    next
  }
  
  var_df <- read_csv(var_analysis_file, show_col_types = FALSE) %>%
    arrange(desc(Total_Importance)) %>%
    head(15)
  
  # Reshape for stacked bar chart
  var_long <- var_df %>%
    select(Variable, Base_Importance, Lag_Importance) %>%
    pivot_longer(cols = c(Base_Importance, Lag_Importance), 
                 names_to = "Type", values_to = "Importance") %>%
    mutate(Type = str_replace(Type, "_Importance", ""))
  
  p <- ggplot(var_long, aes(x = reorder(Variable, Importance), y = Importance, fill = Type)) +
    geom_bar(stat = "identity", width = 0.7) +
    coord_flip() +
    scale_fill_manual(values = c("Base" = "#2ecc71", "Lag" = "#f39c12")) +
    labs(
      title = paste("Variable Importance Breakdown:", full_name),
      subtitle = "Top 15 variables - Base vs Lag contribution",
      x = "Variable",
      y = "Total Importance",
      fill = "Feature Type"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray30"),
      legend.position = "top",
      panel.grid.major.y = element_blank()
    ) +
    theme_white
  
  filename <- file.path(output_dir, paste0(safe_name, "_variable_breakdown.png"))
  ggsave(filename, p, width = 10, height = 8, dpi = 300, bg = "white")
  cat(sprintf("   ✓ Saved: %s_variable_breakdown.png\n", safe_name))
}

# ============================================================================
# PLOT 9: Combined Feature Importance Across All Illnesses (Top 10)
# ============================================================================

cat("9. Creating combined cross-illness feature importance...\n")

all_importance <- data.frame()

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  importance_file <- file.path(results_dir, safe_name, "feature_importance.csv")
  if (file.exists(importance_file)) {
    imp_df <- read_csv(importance_file, show_col_types = FALSE) %>%
      normalize_importance()
    imp_df$Illness <- full_name
    all_importance <- rbind(all_importance, imp_df)
  }
}

if (nrow(all_importance) > 0) {
  # Get top 10 features by average importance
  top_features <- all_importance %>%
    group_by(Feature) %>%
    summarise(Avg_Importance = mean(Importance)) %>%
    arrange(desc(Avg_Importance)) %>%
    head(10) %>%
    pull(Feature)
  
  top_imp_data <- all_importance %>%
    filter(Feature %in% top_features)
  
  p9 <- ggplot(top_imp_data, aes(x = reorder(Feature, Importance), y = Importance, fill = Illness)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    coord_flip() +
    scale_fill_manual(values = c(
      "Acute laryngopharyngitis" = "#e74c3c",
      "Gastritis, unspecified" = "#3498db",
      "Chronic rhinitis" = "#2ecc71"
    )) +
    labs(
      title = "Top 10 Most Important Features Across All Illnesses",
      subtitle = "Comparison of feature importance by illness type",
      x = "Feature",
      y = "Importance Score",
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
  
  ggsave(file.path(output_dir, "cross_illness_top10_features.png"), 
         p9, width = 12, height = 8, dpi = 300, bg = "white")
  cat("   ✓ Saved: cross_illness_top10_features.png\n")
}

# ============================================================================
# PLOT 10: Average Importance by Feature Category
# ============================================================================

cat("10. Creating feature category analysis...\n")

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  
  importance_file <- file.path(results_dir, safe_name, "feature_importance.csv")
  if (!file.exists(importance_file)) {
    next
  }
  
  imp_df <- read_csv(importance_file, show_col_types = FALSE) %>%
    normalize_importance()
  
  # Categorize features
  imp_df <- imp_df %>%
    mutate(
      Category = case_when(
        str_detect(Feature, "PM10|PM25|SO2|CO|O3|NO2") ~ "Air Quality",
        str_detect(Feature, "Temp") ~ "Temperature",
        str_detect(Feature, "Humidity|Vapor") ~ "Humidity",
        str_detect(Feature, "Wind") ~ "Wind",
        str_detect(Feature, "Pressure") ~ "Pressure",
        str_detect(Feature, "Rain|Cloud|Sunshine") ~ "Weather",
        TRUE ~ "Other"
      ),
      Is_Lag = str_detect(Feature, "_lag_")
    )
  
  # Aggregate by category
  category_summary <- imp_df %>%
    group_by(Category, Is_Lag) %>%
    summarise(
      Total_Importance = sum(Importance),
      Avg_Importance = mean(Importance),
      Count = n(),
      .groups = "drop"
    ) %>%
    mutate(Feature_Type = ifelse(Is_Lag, "Lag", "Base"))
  
  p <- ggplot(category_summary, aes(x = reorder(Category, -Total_Importance), 
                                     y = Total_Importance, fill = Feature_Type)) +
    geom_bar(stat = "identity", width = 0.7) +
    geom_text(aes(label = sprintf("%.3f", Total_Importance)), 
              position = position_stack(vjust = 0.5), 
              size = 3.5, color = "white", fontface = "bold") +
    scale_fill_manual(values = c("Base" = "#2ecc71", "Lag" = "#f39c12")) +
    labs(
      title = paste("Feature Category Importance:", full_name),
      subtitle = "Total importance by environmental category (Base + Lag stacked)",
      x = "Environmental Category",
      y = "Total Importance",
      fill = "Feature Type"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5, color = "gray30"),
      axis.text.x = element_text(angle = 15, hjust = 1),
      legend.position = "top",
      panel.grid.major.x = element_blank()
    ) +
    theme_white
  
  filename <- file.path(output_dir, paste0(safe_name, "_category_importance.png"))
  ggsave(filename, p, width = 10, height = 6, dpi = 300, bg = "white")
  cat(sprintf("   ✓ Saved: %s_category_importance.png\n", safe_name))
}

# ============================================================================
# PLOT 11: Model Comparison Summary Dashboard
# ============================================================================

cat("11. Creating comprehensive dashboard plot...\n")

# Prepare data for multi-panel plot
perf_summary <- perf_data %>%
  group_by(Illness) %>%
  summarise(
    GridSearch_R2 = CV_R2[Method == "GridSearch"],
    Optuna_R2 = CV_R2[Method == "Optuna"],
    GridSearch_RMSE = RMSE[Method == "GridSearch"],
    Optuna_RMSE = RMSE[Method == "Optuna"],
    R2_Improvement = ((Optuna_R2 - GridSearch_R2) / GridSearch_R2) * 100,
    RMSE_Reduction = ((GridSearch_RMSE - Optuna_RMSE) / GridSearch_RMSE) * 100
  )

# Create long format for faceting
perf_long <- perf_summary %>%
  select(Illness, GridSearch_R2, Optuna_R2, R2_Improvement) %>%
  pivot_longer(cols = c(GridSearch_R2, Optuna_R2), 
               names_to = "Method", values_to = "CV_R2") %>%
  mutate(Method = str_replace(Method, "_R2", ""))

p11 <- ggplot(perf_long, aes(x = Illness, y = CV_R2, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", CV_R2)), 
            position = position_dodge(width = 0.7), 
            vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("GridSearch" = "#95a5a6", "Optuna" = "#e74c3c"),
                    labels = c("GridSearch (Baseline)", "Optuna (Optimized)")) +
  labs(
    title = "Optimization Impact Summary",
    subtitle = "Cross-Validation R² comparison across all illnesses",
    x = "",
    y = "Cross-Validation R²",
    fill = "Method"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray30", margin = margin(b = 15)),
    axis.text.x = element_text(angle = 20, hjust = 1, size = 11),
    legend.position = "top",
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 11),
    panel.grid.major.x = element_blank(),
    plot.margin = margin(20, 20, 20, 20)
  ) +
  theme_white +
  scale_y_continuous(limits = c(0, 0.65), breaks = seq(0, 0.6, 0.1))

ggsave(file.path(output_dir, "optimization_impact_summary.png"), 
       p11, width = 12, height = 7, dpi = 300, bg = "white")
cat("   ✓ Saved: optimization_impact_summary.png\n")

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n", strrep("=", 80), "\n")
cat("VISUALIZATION COMPLETE\n")
cat(strrep("=", 80), "\n")
cat("\nGenerated visualizations saved to:\n")
cat(output_dir, "\n\n")

cat("Files created:\n")
cat("  1. performance_comparison_cv_r2.png - CV R² comparison\n")
cat("  2. rmse_comparison.png - RMSE comparison\n")
cat("  3. improvement_percentage.png - Improvement percentages\n")
cat("  4. [illness]_feature_importance_top20.png - Top 20 features per illness (3 files)\n")
cat("  5. base_vs_lag_distribution.png - Base vs lag importance distribution\n")
cat("  6. lag_day_importance_heatmap.png - Lag day heatmap\n")
cat("  7. feature_count_distribution.png - Feature counts\n")
cat("  8. [illness]_variable_breakdown.png - Variable breakdown per illness (2 files)\n")
cat("  9. cross_illness_top10_features.png - Top 10 features across all illnesses\n")
cat(" 10. [illness]_category_importance.png - Category importance per illness (2 files)\n")
cat(" 11. optimization_impact_summary.png - Comprehensive dashboard\n")
cat("\nTotal: ~15 publication-quality visualizations\n")
cat(strrep("=", 80), "\n")
