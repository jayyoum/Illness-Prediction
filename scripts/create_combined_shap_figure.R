#!/usr/bin/env Rscript
# Create Combined SHAP Feature Importance Figure
# Layout: A and B on top, C below (like the sample image)

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(gridExtra)
library(grid)

# Set paths
project_root <- "/Users/jay/Desktop/Illness Prediction"
shap_dir <- file.path(project_root, "results/shap_analysis")
output_dir <- file.path(project_root, "results/optimization_visualizations")

cat("Creating combined SHAP feature importance figure...\n")

# White background theme
theme_white <- theme(
  plot.background = element_rect(fill = "white", color = NA),
  panel.background = element_rect(fill = "white")
)

# Define illnesses in order: A, B, C
illnesses <- data.frame(
  safe_name = c("Acute_laryngopharyngitis", "Gastritis_unspecified", "Chronic_rhinitis"),
  full_name = c("Acute laryngopharyngitis", "Gastritis, unspecified", "Chronic rhinitis"),
  panel_label = c("A", "B", "C"),
  stringsAsFactors = FALSE
)

# Create individual plots without titles
plots <- list()

for (i in 1:nrow(illnesses)) {
  safe_name <- illnesses$safe_name[i]
  full_name <- illnesses$full_name[i]
  panel_label <- illnesses$panel_label[i]
  
  shap_file <- file.path(shap_dir, safe_name, "shap_values_summary.csv")
  
  if (!file.exists(shap_file)) {
    cat(sprintf("   ⚠️  Skipping %s - file not found\n", full_name))
    next
  }
  
  shap_df <- read_csv(shap_file, show_col_types = FALSE) %>%
    head(20)
  
  # Create plot WITHOUT title/subtitle
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
      x = "Feature",
      y = "Mean |SHAP Value|",
      fill = "Effect Direction"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      axis.text.y = element_text(size = 9),
      axis.text.x = element_text(size = 9),
      axis.title = element_text(size = 10),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "top",
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      plot.margin = margin(t = 5, r = 10, b = 5, l = 5)
    ) +
    theme_white +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
  
  plots[[i]] <- p
  plots[[paste0(i, "_label")]] <- panel_label
  cat(sprintf("   ✓ Created plot %s: %s\n", panel_label, full_name))
}

# Arrange in layout: A and B on top, C below (same size, centered)
# Create the combined layout
cat("\nCombining plots into final figure...\n")

# Create top row (A and B side by side)
top_row <- arrangeGrob(plots[[1]], plots[[2]], ncol = 2, widths = unit(c(1, 1), "null"))

# Create bottom row (C centered with same width as A or B)
# To center C with same width: use widths 1:2:1 so C takes 50% (same as A or B)
bottom_row <- arrangeGrob(
  nullGrob(), plots[[3]], nullGrob(),
  ncol = 3,
  widths = unit(c(1, 2, 1), "null")
)

# Combine top and bottom rows
combined_plot <- arrangeGrob(
  top_row, bottom_row,
  nrow = 2,
  heights = unit(c(1, 1), "null")
)

# Save the combined figure
output_file <- file.path(output_dir, "SHAP_feature_importance_combined.png")

# Save with high resolution and large size to preserve individual plot quality
png(output_file, width = 18, height = 14, units = "in", res = 300, bg = "white")

# Draw the plots
grid.draw(combined_plot)

# Add panel labels OUTSIDE the plots
# Top row: A at left (0%), B at right (starts at 50%)
# Bottom row: C centered (starts at 25% since it's centered with 1:2:1 spacing)
grid.text("A", x = unit(0.02, "npc"), y = unit(0.98, "npc"), 
          just = c("left", "top"), gp = gpar(fontsize = 18, fontface = "bold"))
grid.text("B", x = unit(0.52, "npc"), y = unit(0.98, "npc"), 
          just = c("left", "top"), gp = gpar(fontsize = 18, fontface = "bold"))
grid.text("C", x = unit(0.27, "npc"), y = unit(0.48, "npc"), 
          just = c("left", "top"), gp = gpar(fontsize = 18, fontface = "bold"))

dev.off()

cat(sprintf("\n✓ Combined figure saved: %s\n", output_file))
cat(sprintf("   Dimensions: 18\" x 14\" at 300 DPI\n"))
cat(sprintf("   Layout: A and B (top row), C (bottom row, same size, centered)\n"))
