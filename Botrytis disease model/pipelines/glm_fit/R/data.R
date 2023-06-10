read_sb_data <- function(input_data) {
  df <- read_csv(input_data, col_types = cols()) %>%
    mutate(
      year = as.factor(year),
      pruning = as.factor(pruning),
      Site = as.factor(Site)
    ) %>%
    select(-c(
      Variety, Date, first_sev1, end_sev23, end_sev1, 
      first_sev23, end_sev23, doy, doy.july1st,
      previous.season, season
    ))
}