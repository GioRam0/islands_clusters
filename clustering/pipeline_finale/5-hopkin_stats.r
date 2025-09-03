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

# Percorso al file CSV nella cartella "data"
file_path <- here("clusterizzazione/pipeline_finale/dataframes/df_norm_first_step.csv")
data <- read.csv(file_path, header = TRUE, sep = ",")
txt_folder <- here("clusterizzazione/pipeline_finale/hopkins_stats")
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

for (j in 1:length(columns)) {
  txt_path <- file.path(txt_folder, paste0("hopkins_stats_", names[j], ".txt"))
  if (file.exists(txt_path)) {
    file.remove(txt_path)
  }
  cols_to_select <- columns[[j]]
  print(names[j])
  for (i in 0:max_clusters) {
    filtered_data <- subset(data, clusters == i)
    selected_data <- filtered_data[cols_to_select]
    m_values <- c(10)
    if (nrow(selected_data) > 50) {
      m_values <- unique(c(m_values, 50))
    }
    if (nrow(selected_data) > 100) {
      m_values <- unique(c(m_values, seq(100, nrow(selected_data), by = 100)))
    } else {
      m_values <- unique(c(m_values, nrow(selected_data)-1))
    }
    for (m in m_values) {
      hopkins_stat <- hopkins(selected_data, m = m)
      write(paste("Cluster", i, "- Hopkins statistic (m =", m, "):", hopkins_stat), file = txt_path, append = TRUE)
    }
    cat("\n", file = txt_path, append = TRUE)
    for (m in m_values) {
      set.seed(264)
      hopkins_stat <- clustertend::hopkins(selected_data, n = m)
      write(paste("Cluster", i, "- Hopkins statistic (clustertend) (m =", m, "):", hopkins_stat), file = txt_path, append = TRUE)
    }
    cat("\n", file = txt_path, append = TRUE)
  }
}