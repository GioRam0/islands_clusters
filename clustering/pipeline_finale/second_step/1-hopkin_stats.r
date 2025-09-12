if (!require("clustertend")) {
  install.packages("clustertend")
  library(clustertend)
} else {
  library(clustertend)
}

if (!require("here")) {
  install.packages("here")
  library(here)
} else {
  library(here)
}

if (!require("hopkins")) {
  install.packages("hopkins")
  library(hopkins)
} else {
  library(hopkins)
}

#percorso al file CSV
file_path <- here("clustering/pipeline_finale/first_step/dataframes/df_norm_first_step.csv")
data <- read.csv(file_path, header = TRUE, sep = ",")
txt_folder <- here("clustering/pipeline_finale/second_step/results/hopkins_stats")
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

for (i in 0:max_cluster) {
  txt_path <- file.path(txt_folder, paste0("hopkins_stats_cluster_", i, ".txt"))
  if (file.exists(txt_path)) {
    file.remove(txt_path)
  }
  filtered_data <- subset(data, cluster == i)
  m <- min(max(30, floor(0.10 * num_rows)), num_rows-1)
  print(m)
  write(paste("Valore di m utilizzato:", m), file = txt_path, append = TRUE)
  num_iterations <- 10
  for (j in 1:length(columns)) {
    cols_to_select <- columns[[j]]
    selected_data <- filtered_data[cols_to_select]
    if (j != 1) {
      n <- sum(filtered_data$hydro > 0)
      n1 <- sum(filtered_data$hydro == 0)
      write(paste("Cluster", i, ", elementi con hydro non nullo", n, ", elementi con hydro nullo", n1), file = txt_path, append = TRUE)
    }
    if (j == 3) {
      n <- sum(filtered_data$geothermal_potential > 0)
      n1 <- sum(filtered_data$geothermal_potential == 0)
      write(paste("Elementi con geot non nullo", n, ", elementi con geot nullo", n1), file = txt_path, append = TRUE)
    }
    cluster_hopkins_results <- numeric(num_iterations)
    clustertend_hopkins_results <- numeric(num_iterations)
    for (h in 1:num_iterations) {
      cluster_hopkins_results[h] <- hopkins(selected_data, m = m)
      set.seed(264+h)
      clustertend_hopkins_results[h] <- clustertend::hopkins(selected_data, n = m)$H
    }
    cluster_average_hopkins <- mean(cluster_hopkins_results)
    clustertend_average_hopkins <- mean(clustertend_hopkins_results)
    write(paste("Media della statistica di Hopkins (libreria 'cluster'):", round(cluster_average_hopkins, 6)), file = txt_path, append = TRUE)
    write(paste("Media della statistica di Hopkins (libreria 'clustertend'):", round(clustertend_average_hopkins, 6)), file = txt_path, append = TRUE)
    cat("\n", file = txt_path, append = TRUE)
  }
}