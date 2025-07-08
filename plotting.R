library(ggplot2)

df_plot <- read.csv("/home/degan/msk/data/umap_embedding_for_R.csv")

pdf("/home/degan/msk/figures/umap_sample_embedding.pdf")
ggplot(df_plot, aes(x = UMAP1, y = UMAP2, color = CANCER_TYPE)) +
  geom_point(alpha = 0.7, size = 1.5) +
  labs(title = "UMAP of Sample Embeddings", color = "Cancer Type") +
  theme_minimal(base_size = 10) +
  theme(legend.position = "top") +
  guides(color = guide_legend(ncol = 3))
dev.off()

