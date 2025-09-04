#!/usr/bin/env Rscript

# R Script to Plot Training Results with Epoch-by-Epoch Progress
# Author: Enhanced for epoch-wise visualization
# Date: August 13, 2025

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

# Function to find the most recent training files
find_training_files <- function() {
  cat("🔍 Searching for training files...\n")
  
  # Search in multiple possible locations
  search_dirs <- c(
    ".",                           # Current directory
    "results/training_outputs",    # Expected results directory
    "results",                     # Results directory
    "src/core",                   # Where main.py is located
    "outputs"                     # Alternative outputs directory
  )
  
  epoch_files <- c()
  summary_files <- c()
  
  for (dir in search_dirs) {
    if (dir.exists(dir)) {
      cat(sprintf("  📂 Searching in: %s\n", dir))
      
      # Look for epoch-wise training results
      pattern1 <- file.path(dir, "training_results_*.csv")
      pattern2 <- file.path(dir, "*training*.csv")
      pattern3 <- file.path(dir, "*results*.csv")
      
      epoch_candidates <- c(
        Sys.glob(pattern1),
        Sys.glob(pattern2),
        Sys.glob(pattern3)
      )
      
      # Look for summary files
      summary_candidates <- Sys.glob(file.path(dir, "training_summary_*.csv"))
      
      epoch_files <- c(epoch_files, epoch_candidates)
      summary_files <- c(summary_files, summary_candidates)
      
      # List all CSV files in this directory
      all_csvs <- list.files(dir, pattern = "\\.csv$", full.names = TRUE)
      if (length(all_csvs) > 0) {
        cat(sprintf("    📄 Found CSV files: %s\n", paste(basename(all_csvs), collapse = ", ")))
      }
    }
  }
  
  # Remove duplicates
  epoch_files <- unique(epoch_files)
  summary_files <- unique(summary_files)
  
  result <- list(epoch_file = NULL, summary_file = NULL)
  
  # Find most recent epoch-wise file
  if (length(epoch_files) > 0) {
    cat(sprintf("📊 Found potential epoch files: %s\n", paste(basename(epoch_files), collapse = ", ")))
    
    # Filter files that actually contain epoch data by checking their structure
    valid_epoch_files <- c()
    for (file in epoch_files) {
      if (file.exists(file)) {
        tryCatch({
          test_data <- read.csv(file, nrows = 5, stringsAsFactors = FALSE)
          required_cols <- c("epoch", "train_loss", "val_loss")
          if (all(required_cols %in% names(test_data))) {
            valid_epoch_files <- c(valid_epoch_files, file)
            cat(sprintf("  ✅ Valid epoch file: %s\n", basename(file)))
          } else {
            cat(sprintf("  ❌ Invalid epoch file (missing columns): %s\n", basename(file)))
            cat(sprintf("      Has columns: %s\n", paste(names(test_data), collapse = ", ")))
          }
        }, error = function(e) {
          cat(sprintf("  ❌ Error reading file: %s\n", basename(file)))
        })
      }
    }
    
    if (length(valid_epoch_files) > 0) {
      epoch_files_info <- file.info(valid_epoch_files)
      result$epoch_file <- rownames(epoch_files_info)[which.max(epoch_files_info$mtime)]
      cat(sprintf("📊 Selected epoch file: %s\n", basename(result$epoch_file)))
    }
  }
  
  # Find most recent summary file
  if (length(summary_files) > 0) {
    summary_files_info <- file.info(summary_files)
    result$summary_file <- rownames(summary_files_info)[which.max(summary_files_info$mtime)]
    cat(sprintf("📋 Found training summary file: %s\n", basename(result$summary_file)))
  }
  
  return(result)
}

# Function to create epoch-wise training plots
create_epoch_plots <- function(epoch_data) {
  cat("🎨 Creating plots with data structure:\n")
  print(str(epoch_data))
  
  # Ensure numeric columns
  numeric_cols <- c("epoch", "train_loss", "val_loss", "train_ci", "val_ci")
  for (col in numeric_cols) {
    if (col %in% names(epoch_data)) {
      epoch_data[[col]] <- as.numeric(epoch_data[[col]])
    }
  }
  
  # Prepare data for plotting
  loss_data <- epoch_data %>%
    select(any_of(c("epoch", "train_loss", "val_loss"))) %>%
    pivot_longer(cols = -epoch, 
                 names_to = "loss_type", 
                 values_to = "loss_value") %>%
    mutate(dataset = ifelse(grepl("train", loss_type), "Training", "Validation"))
  
  ci_data <- epoch_data %>%
    select(any_of(c("epoch", "train_ci", "val_ci"))) %>%
    pivot_longer(cols = -epoch, 
                 names_to = "ci_type", 
                 values_to = "ci_value") %>%
    mutate(dataset = ifelse(grepl("train", ci_type), "Training", "Validation"))
  
  # Loss over epochs plot
  p1 <- ggplot(loss_data, aes(x = epoch, y = loss_value, color = dataset)) +
    geom_line(size = 1.2, alpha = 0.8) +
    geom_point(size = 2, alpha = 0.6) +
    scale_color_viridis_d(name = "Dataset", begin = 0.2, end = 0.8) +
    scale_x_continuous(breaks = pretty_breaks(n = 8)) +
    scale_y_continuous(labels = number_format(accuracy = 0.001)) +
    labs(
      title = "📉 Training Progress: Cox Loss Over Epochs",
      subtitle = "Lower values indicate better model fit",
      x = "Epoch",
      y = "Cox Loss",
      caption = "Enhanced GNN Model Training"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(size = 14, face = "bold", color = "#2c3e50"),
      plot.subtitle = element_text(size = 11, color = "#7f8c8d"),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )
  
  # Add smoothed trend lines if enough data points
  if (nrow(loss_data) > 10) {
    p1 <- p1 + geom_smooth(method = "loess", se = TRUE, alpha = 0.2, size = 0.8)
  }
  
  # C-Index over epochs plot (only if CI data exists)
  p2 <- NULL
  if (nrow(ci_data) > 0 && !all(is.na(ci_data$ci_value))) {
    p2 <- ggplot(ci_data, aes(x = epoch, y = ci_value, color = dataset)) +
      geom_line(size = 1.2, alpha = 0.8) +
      geom_point(size = 2, alpha = 0.6) +
      geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", alpha = 0.7) +
      annotate("text", x = Inf, y = 0.5, label = "Random Baseline", 
               hjust = 1.1, vjust = -0.5, color = "red", size = 3) +
      scale_color_viridis_d(name = "Dataset", begin = 0.2, end = 0.8) +
      scale_x_continuous(breaks = pretty_breaks(n = 8)) +
      scale_y_continuous(labels = number_format(accuracy = 0.001)) +
      labs(
        title = "🎯 Training Progress: C-Index Over Epochs",
        subtitle = "Higher values indicate better survival prediction accuracy",
        x = "Epoch",
        y = "Concordance Index (C-Index)",
        caption = "Enhanced GNN Model Training"
      ) +
      theme_minimal(base_size = 12) +
      theme(
        plot.title = element_text(size = 14, face = "bold", color = "#2c3e50"),
        plot.subtitle = element_text(size = 11, color = "#7f8c8d"),
        legend.position = "bottom",
        panel.grid.minor = element_blank()
      )
    
    # Add smoothed trend lines if enough data points
    if (nrow(ci_data) > 10) {
      p2 <- p2 + geom_smooth(method = "loess", se = TRUE, alpha = 0.2, size = 0.8)
    }
  }
  
  return(list(loss_plot = p1, cindex_plot = p2))
}

# Function to create training summary
create_training_summary <- function(epoch_data, summary_data = NULL) {
  if (is.null(epoch_data) || nrow(epoch_data) == 0) {
    cat("⚠️ No epoch data available for summary\n")
    return()
  }
  
  # Calculate summary statistics
  total_epochs <- nrow(epoch_data)
  
  # Print summary
  cat("\n", rep("=", 60), "\n")
  cat("🎯 ENHANCED GNN MODEL TRAINING SUMMARY\n")
  cat(rep("=", 60), "\n")
  cat("📊 Training Overview:\n")
  cat(sprintf("   • Total Epochs: %d\n", total_epochs))
  
  if ("val_loss" %in% names(epoch_data)) {
    best_val_loss_idx <- which.min(epoch_data$val_loss)
    best_val_loss <- epoch_data$val_loss[best_val_loss_idx]
    best_val_loss_epoch <- epoch_data$epoch[best_val_loss_idx]
    cat(sprintf("   • Best Validation Loss: %.4f (Epoch %d)\n", best_val_loss, best_val_loss_epoch))
  }
  
  if ("val_ci" %in% names(epoch_data) && !all(is.na(epoch_data$val_ci))) {
    best_val_ci_idx <- which.max(epoch_data$val_ci)
    best_val_ci <- epoch_data$val_ci[best_val_ci_idx]
    best_val_ci_epoch <- epoch_data$epoch[best_val_ci_idx]
    cat(sprintf("   • Best Validation C-Index: %.4f (Epoch %d)\n", best_val_ci, best_val_ci_epoch))
  }
  
  cat(rep("=", 60), "\n\n")
}

# Function to clean and convert data
clean_data <- function(data) {
  # Handle various data type issues
  for (col in names(data)) {
    if (col %in% c("epoch", "train_loss", "val_loss", "train_ci", "val_ci")) {
      # Remove any non-numeric characters and convert
      data[[col]] <- as.numeric(as.character(data[[col]]))
      
      # Replace any remaining NAs with 0 for plotting
      if (col %in% c("train_ci", "val_ci")) {
        data[[col]][is.na(data[[col]])] <- 0
      }
    }
  }
  return(data)
}

# Main execution
cat("🚀 Enhanced GNN Training Results Analysis\n")
cat("==========================================\n")

# Create output directory
dir.create("figures", showWarnings = FALSE)

# Find training files
files <- find_training_files()

# Read epoch-wise data if available
epoch_data <- NULL
if (!is.null(files$epoch_file) && file.exists(files$epoch_file)) {
  cat("📖 Reading epoch-wise training data...\n")
  epoch_data <- read.csv(files$epoch_file, stringsAsFactors = FALSE)
  
  # Clean the data
  epoch_data <- clean_data(epoch_data)
  
  cat("📋 Epoch Data Structure (after cleaning):\n")
  print(str(epoch_data))
  cat("First few rows:\n")
  print(head(epoch_data))
  
  # Check for any remaining issues
  cat("Data summary:\n")
  print(summary(epoch_data))
}

# Read summary data if available
summary_data <- NULL
if (!is.null(files$summary_file) && file.exists(files$summary_file)) {
  cat("📖 Reading training summary data...\n")
  summary_data <- read.csv(files$summary_file, stringsAsFactors = FALSE)
}

# Create plots if epoch data is available
if (!is.null(epoch_data) && nrow(epoch_data) > 0) {
  cat("🎨 Creating epoch-wise training plots...\n")
  
  # Create plots
  plots <- create_epoch_plots(epoch_data)
  
  # Save loss plot
  if (!is.null(plots$loss_plot)) {
    ggsave("results/figures/loss_over_epochs.pdf", plots$loss_plot, width = 12, height = 8, dpi = 300)
    ggsave("results/figures/loss_over_epochs.png", plots$loss_plot, width = 12, height = 8, dpi = 300)
    cat("✅ Loss plots saved\n")
  }
  
  # Save C-index plot if it exists
  if (!is.null(plots$cindex_plot)) {
    ggsave("results/figures/cindex_over_epochs.pdf", plots$cindex_plot, width = 12, height = 8, dpi = 300)
    ggsave("results/figures/cindex_over_epochs.png", plots$cindex_plot, width = 12, height = 8, dpi = 300)
    cat("✅ C-Index plots saved\n")
  }
  
  # Create training summary
  create_training_summary(epoch_data, summary_data)
  
  cat("✅ Analysis completed successfully!\n")
  
} else {
  cat("❌ No epoch-wise training data found!\n")
  cat("Please check:\n")
  cat("1. Have you run the training script (main.py)?\n")
  cat("2. Does your training script save CSV files with epoch data?\n")
  cat("3. Are the CSV files in the expected location?\n")
  
  # Show what files we did find
  cat("\n🔍 Files found in current directory:\n")
  all_files <- list.files(".", pattern = "\\.(csv|txt)$", recursive = TRUE)
  if (length(all_files) > 0) {
    for (file in all_files) {
      cat(sprintf("  📄 %s\n", file))
    }
  } else {
    cat("  (no CSV or TXT files found)\n")
  }
}

cat("\n🎯 Analysis Complete!\n")
