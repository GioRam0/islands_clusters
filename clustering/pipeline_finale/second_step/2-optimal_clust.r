if (!require("NbClust")) {
  install.packages("NbClust")
  library(NbClust)
} else {
  library(NbClust)
}

if (!require("here")) {
  install.packages("here")
  library(here)
} else {
  library(here)
}

# Percorso al file CSV nella cartella "data"
file_path <- here("clustering/pipeline_finale/first_step/dataframes/df_norm_first_step.csv")
data <- read.csv(file_path, header = TRUE, sep = ",")
txt_folder <- here("clustering/pipeline_finale/second_step/results/optimal_clust")
if (!dir.exists(txt_folder)) {
  dir.create(txt_folder, recursive = TRUE)
}

columns <- list(
  c('solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi'),
  c('solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro'),
  c('solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential')
)
names <- c('base', 'hydro', 'geothermal_potential')

max_cluster <- max(data$cluster, na.rm = TRUE)
for (i in (0:max_cluster+1)) {
  filtered_data <- subset(data, data$cluster == i-1)
  txt_path <- file.path(txt_folder, paste0("optimal_clust_", "_cluster_", i-1, ".txt"))
  if (file.exists(txt_path)) {
    file.remove(txt_path)
  }
  for (j in 1:length(columns)) {
    write(paste("Gruppo di colonne", names[j], ":"), file = txt_path, append = TRUE)
    cols_to_select <- columns[[j]]
    selected_data <- filtered_data[cols_to_select]
    set.seed(123) # Per riproducibilità
    if (nrow(selected_data) < 50) {
      max = 10
    } else{
      max = 20
    }
    res <- NbClust(
      data = selected_data,
      distance = "euclidean",
      min.nc = 2,
      max.nc = max,
      method = "kmeans"
    )
    recommended_clusters <- res$Best.nc[1, ]
    cluster_counts <- table(recommended_clusters)
    cluster_counts_ordered <- cluster_counts[order(as.numeric(names(cluster_counts)))]
    write(paste("Among all indices:"), file = txt_path, append = TRUE)
    for (num_cluster in names(cluster_counts_ordered)) {
      count <- cluster_counts_ordered[num_cluster]
      write(paste(count, "proposed ", num_cluster, " as the best number of clusters"), file = txt_path, append = TRUE)
    }
    optimal_cluster_num <- as.numeric(names(which.max(cluster_counts)))
    write(paste("According to the majority rule, the best number of clusters is", optimal_cluster_num), file = txt_path, append = TRUE)
    cat("\n", file = txt_path, append = TRUE)
  }
}