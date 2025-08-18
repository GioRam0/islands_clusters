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
file_path <- here("clusterizzazione/pipeline_finale/dataframes/df_norm_first_step.csv")
data <- read.csv(file_path, header = TRUE, sep = ",")
txt_folder <- here("clusterizzazione/pipeline_finale/optimal_clust")
if (!dir.exists(txt_folder)) {
  dir.create(txt_folder, recursive = TRUE)
}

columns <- list(
  c('solar_pow', 'eolico', 'superficie_res'),
  c('solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std'),
  c('solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore'),
  c('solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi'),
  c('solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro'),
  c('solar_pow', 'eolico', 'superficie_res', 'solar_seas_ind', 'eolico_std', 'offshore', 'evi', 'hydro', 'geothermal_potential')
)
names <- c('base', 'varianze', 'offshore', 'evi', 'hydro', 'geothermal_potential')

max_clusters <- max(data$clusters, na.rm = TRUE)
for (i in (max_clusters-1:max_clusters+1)) {
  filtered_data <- subset(data, data$clusters == i)
  txt_path <- file.path(txt_folder, paste0("optimal_clust_", "_cluster_", i, ".txt"))
  if (file.exists(txt_path)) {
    file.remove(txt_path)
  }
  for (j in 1:length(columns)) {
    write(paste("Gruppo di colonne", names[j], ":"), file = txt_path, append = TRUE)
    cols_to_select <- columns[[j]]
    selected_data <- filtered_data[cols_to_select]
    set.seed(123) # Per riproducibilità
    res <- NbClust(
      data = selected_data,
      distance = "euclidean",
      min.nc = 2,
      max.nc = 20,
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