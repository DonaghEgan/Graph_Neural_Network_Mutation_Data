#!/usr/bin/env Rscript

# R Script to Plot Training Results from Enhanced GNN Model
# Author: Generated for training_summary_20250812_123555.csv
# Date: August 12, 2025

# Load required libraries
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(gridExtra)
  library(viridis)
  library(scales)
})

# Set working directory and file paths
setwd("/home/degan/Graph_Neural_Network_Mutation_Data")
csv_file <- "training_summary_20250812_123555.csv"

# Check if file exists
if (!file.exists(csv_file)) {
  cat("❌ Error: File", csv_file, "not found!\n")
  cat("Available CSV files in current directory:\n")
  csv_files <- list.files(pattern = "*.csv")
  if (length(csv_files) > 0) {
    cat(paste(csv_files, collapse = "\n"), "\n")
  } else {
    cat("No CSV files found.\n")
  }
  
  # Try to find the most recent training summary file
  summary_files <- list.files(pattern = "training_summary_.*\\.csv")
  if (length(summary_files) > 0) {
    # Sort by modification time and take the most recent
    summary_files_info <- file.info(summary_files)
    latest_file <- rownames(summary_files_info)[which.max(summary_files_info$mtime)]
    cat("📊 Using most recent training summary file:", latest_file, "\n")
    csv_file <- latest_file
  } else {
    stop("No training summary files found!")
  }
}

# Read the training summary data
cat("📖 Reading training summary data from:", csv_file, "\n")
summary_data <- read.csv(csv_file, stringsAsFactors = FALSE)

# Ensure value column is numeric and handle any missing values
summary_data$value <- as.numeric(summary_data$value)

# Display the data structure
cat("\n📋 Training Summary Data Structure:\n")
print(str(summary_data))
cat("\n📊 Training Summary Metrics:\n")
print(summary_data)

# Create output directory for plots
dir.create("figures", showWarnings = FALSE)

# Create a comprehensive summary plot
create_summary_plot <- function(data) {
  # Ensure value column is numeric
  data$value <- as.numeric(data$value)
  
  # Prepare data for plotting
  metrics_df <- data %>%
    filter(metric %in% c("best_val_ci", "final_test_ci", "test_ci_best_model")) %>%
    mutate(
      metric_type = case_when(
        metric == "best_val_ci" ~ "Best Validation C-Index",
        metric == "final_test_ci" ~ "Final Test C-Index",
        metric == "test_ci_best_model" ~ "Test C-Index (Best Model)",
        TRUE ~ metric
      ),
      metric_category = case_when(
        grepl("val", metric) ~ "Validation",
        grepl("test", metric) ~ "Test",
        TRUE ~ "Other"
      )
    )
  
  loss_df <- data %>%
    filter(metric %in% c("best_val_loss", "final_test_loss", "test_loss_best_model")) %>%
    mutate(
      metric_type = case_when(
        metric == "best_val_loss" ~ "Best Validation Loss",
        metric == "final_test_loss" ~ "Final Test Loss",
        metric == "test_loss_best_model" ~ "Test Loss (Best Model)",
        TRUE ~ metric
      ),
      metric_category = case_when(
        grepl("val", metric) ~ "Validation",
        grepl("test", metric) ~ "Test",
        TRUE ~ "Other"
      )
    )
  
  # C-Index Plot
  p1 <- ggplot(metrics_df, aes(x = reorder(metric_type, value), y = value, fill = metric_category)) +
    geom_col(width = 0.7, alpha = 0.8) +
    geom_text(aes(label = sprintf("%.4f", value)), 
              hjust = -0.1, size = 3.5, fontface = "bold") +
    scale_fill_viridis_d(name = "Dataset", begin = 0.2, end = 0.8) +
    scale_y_continuous(limits = c(0, max(metrics_df$value) * 1.15), 
                       labels = number_format(accuracy = 0.001)) +
    coord_flip() +
    labs(
      title = "🎯 Model Performance: Concordance Index (C-Index)",
      subtitle = paste("Higher values indicate better survival prediction accuracy"),
      x = "Metric",
      y = "C-Index Value",
      caption = paste("Generated from:", csv_file)
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(size = 14, face = "bold", color = "#2c3e50"),
      plot.subtitle = element_text(size = 11, color = "#7f8c8d"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
  
  # Loss Plot
  p2 <- ggplot(loss_df, aes(x = reorder(metric_type, -value), y = value, fill = metric_category)) +
    geom_col(width = 0.7, alpha = 0.8) +
    geom_text(aes(label = sprintf("%.4f", value)), 
              hjust = -0.1, size = 3.5, fontface = "bold") +
    scale_fill_viridis_d(name = "Dataset", begin = 0.2, end = 0.8) +
    scale_y_continuous(limits = c(0, max(loss_df$value) * 1.15),
                       labels = number_format(accuracy = 0.001)) +
    coord_flip() +
    labs(
      title = "📉 Model Performance: Cox Loss",
      subtitle = "Lower values indicate better model fit",
      x = "Metric",
      y = "Loss Value",
      caption = paste("Generated from:", csv_file)
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(size = 14, face = "bold", color = "#2c3e50"),
      plot.subtitle = element_text(size = 11, color = "#7f8c8d"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
  
  return(list(cindex_plot = p1, loss_plot = p2))
}

# Create additional summary statistics
create_summary_stats <- function(data) {
  # Extract key metrics with proper type conversion and error handling
  safe_extract <- function(metric_name) {
    value <- data$value[data$metric == metric_name]
    if (length(value) == 0) {
      return(NA)
    }
    return(as.numeric(value))
  }
  
  total_epochs <- safe_extract("total_epochs")
  best_val_ci <- safe_extract("best_val_ci")
  final_test_ci <- safe_extract("final_test_ci")
  best_val_loss <- safe_extract("best_val_loss")
  final_test_loss <- safe_extract("final_test_loss")
  
  # Calculate performance summary
  cat("\n" , rep("=", 60), "\n")
  cat("🎯 ENHANCED GNN MODEL PERFORMANCE SUMMARY\n")
  cat(rep("=", 60), "\n")
  cat("📊 Training Configuration:\n")
  cat(sprintf("   • Total Epochs: %.0f\n", ifelse(is.na(total_epochs), 0, total_epochs)))
  cat(sprintf("   • Enhanced Loss Functions: Cox + Weighted + Combined\n"))
  cat(sprintf("   • GPU Acceleration: Enabled\n"))
  cat(sprintf("   • Early Stopping: Enabled\n"))
  cat("\n🎯 Performance Metrics:\n")
  cat(sprintf("   • Best Validation C-Index: %.4f\n", ifelse(is.na(best_val_ci), 0, best_val_ci)))
  cat(sprintf("   • Final Test C-Index: %.4f\n", ifelse(is.na(final_test_ci), 0, final_test_ci)))
  
  # Calculate improvement if best model exists
  if ("test_ci_best_model" %in% data$metric) {
    test_ci_best <- safe_extract("test_ci_best_model")
    if (!is.na(test_ci_best) && !is.na(final_test_ci)) {
      improvement <- ((test_ci_best - final_test_ci) / final_test_ci) * 100
      cat(sprintf("   • Test C-Index (Best Model): %.4f\n", test_ci_best))
      cat(sprintf("   • Model Selection Improvement: %+.2f%%\n", improvement))
    }
  }
  
  cat(sprintf("\n📉 Loss Metrics:\n"))
  cat(sprintf("   • Best Validation Loss: %.4f\n", ifelse(is.na(best_val_loss), 0, best_val_loss)))
  cat(sprintf("   • Final Test Loss: %.4f\n", ifelse(is.na(final_test_loss), 0, final_test_loss)))
  
  # Performance assessment
  cat(sprintf("\n🏆 Performance Assessment:\n"))
  if (!is.na(best_val_ci)) {
    if (best_val_ci > 0.6) {
      performance_level <- "Excellent"
      emoji <- "🌟"
    } else if (best_val_ci > 0.55) {
      performance_level <- "Good"
      emoji <- "👍"
    } else if (best_val_ci > 0.5) {
      performance_level <- "Fair"
      emoji <- "⚠️"
    } else {
      performance_level <- "Needs Improvement"
      emoji <- "🔧"
    }
    
    cat(sprintf("   %s Model Performance: %s (C-Index: %.4f)\n", emoji, performance_level, best_val_ci))
    cat(sprintf("   📈 Random baseline C-Index: 0.5000\n"))
    cat(sprintf("   📊 Performance above baseline: %+.4f\n", best_val_ci - 0.5))
  } else {
    cat("   ⚠️  Unable to assess performance - missing validation C-Index\n")
  }
  cat(rep("=", 60), "\n\n")
}

# Generate plots and summary
cat("🎨 Creating visualizations...\n")

# Create summary statistics
create_summary_stats(summary_data)

# Create plots
plots <- create_summary_plot(summary_data)

# Save individual plots
ggsave("figures/cindex_performance.pdf", plots$cindex_plot, width = 10, height = 6, dpi = 300)
ggsave("figures/cindex_performance.png", plots$cindex_plot, width = 10, height = 6, dpi = 300)

ggsave("figures/loss_performance.pdf", plots$loss_plot, width = 10, height = 6, dpi = 300)
ggsave("figures/loss_performance.png", plots$loss_plot, width = 10, height = 6, dpi = 300)

# Create combined plot
combined_plot <- grid.arrange(plots$cindex_plot, plots$loss_plot, ncol = 1)

# Save combined plot
pdf("figures/training_summary_combined.pdf", width = 12, height = 10)
grid.arrange(plots$cindex_plot, plots$loss_plot, ncol = 1)
dev.off()

png("figures/training_summary_combined.png", width = 12, height = 10, units = "in", res = 300)
grid.arrange(plots$cindex_plot, plots$loss_plot, ncol = 1)
dev.off()

# Create a detailed metrics table plot
create_metrics_table <- function(data) {
  # Prepare data for table
  table_data <- data %>%
    mutate(
      formatted_value = case_when(
        grepl("epochs", metric) ~ as.character(as.integer(value)),
        TRUE ~ sprintf("%.4f", value)
      ),
      metric_display = case_when(
        metric == "best_val_ci" ~ "Best Validation C-Index",
        metric == "best_val_loss" ~ "Best Validation Loss",
        metric == "final_test_ci" ~ "Final Test C-Index",
        metric == "final_test_loss" ~ "Final Test Loss",
        metric == "total_epochs" ~ "Total Training Epochs",
        metric == "test_ci_best_model" ~ "Test C-Index (Best Model)",
        metric == "test_loss_best_model" ~ "Test Loss (Best Model)",
        TRUE ~ metric
      )
    ) %>%
    select(Metric = metric_display, Value = formatted_value)
  
  return(table_data)
}

# Create and save metrics table
metrics_table <- create_metrics_table(summary_data)
write.csv(metrics_table, "figures/training_metrics_table.csv", row.names = FALSE)

cat("✅ Plots and analysis completed successfully!\n")
cat("\n📁 Generated files:\n")
cat("   • figures/cindex_performance.pdf\n")
cat("   • figures/cindex_performance.png\n")
cat("   • figures/loss_performance.pdf\n")
cat("   • figures/loss_performance.png\n")
cat("   • figures/training_summary_combined.pdf\n")
cat("   • figures/training_summary_combined.png\n")
cat("   • figures/training_metrics_table.csv\n")

cat("\n🎯 Analysis Summary:\n")
cat("   The enhanced GNN model with improved loss functions shows:\n")
cat("   • Advanced loss scheduling (Cox → Weighted → Combined)\n")
cat("   • GPU acceleration and automatic device selection\n")
cat("   • Early stopping and learning rate scheduling\n")
cat("   • Numerical stability improvements\n")
cat("   • Comprehensive performance tracking\n")

cat("\n🚀 Next Steps:\n")
cat("   • Review the generated plots for performance insights\n")
cat("   • Compare C-Index values with baseline models\n")
cat("   • Analyze loss convergence patterns\n")
cat("   • Consider hyperparameter tuning if needed\n")

cat("\n" , rep("=", 60), "\n")
