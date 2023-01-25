# Created by use_targets().
# Follow the comments below to fill in this target script.
# Then follow the manual to check and run the pipeline:
#   https://books.ropensci.org/targets/walkthrough.html#inspect-the-pipeline # nolint

# Load packages required to define the pipeline:
library(targets)
# library(tarchetypes) # Load other packages as needed. # nolint

# Set target options:
tar_option_set(
  packages = c("tidyverse", "lubridate", "readxl", "RcppRoll", "MASS", "ggpubr", "gtable", "grid", "ggthemes","viridis"), 
  # packages that your targets need to run
  format = "rds" # default storage format
  # Set other options as needed.
)

# tar_make_clustermq() configuration (okay to leave alone):
options(clustermq.scheduler = "multiprocess")

# tar_make_future() configuration (okay to leave alone):
# Install packages {{future}}, {{future.callr}}, and {{future.batchtools}} to allow use_targets() to configure tar_make_future() options.

# Run the R scripts in the R/ folder with your custom functions:
tar_source(
  source("Functions/Abundance_conidia.R"),
  source("Functions/Calculate_DAWS.R"),
  source("Functions/Compare_BBCH_DAWS.R"))
# source("other_functions.R") # Source other scripts as needed. # nolint


# Replace the target list below with your own:
list(

  tar_target(f1_BBCH, "Data/BBCH.xlsx", format = "file"),
  tar_target(f2_HrData, "Data/MRL.climate.data.hourly.csv", format = "file"),
#  tar_target(f3_ObsData, "Data/....csv", format = "file"),

  tar_target(BbchData, get_data(f1_BBCH)),
  tar_target(HrData, get_data(f2_HrData)),
#  tar_target(ObsData, get_data(f3_ObsData)),

  tar_target(Hr_model, CalHr_model(HrData)),
  tar_target(CISO_model, CalCISO_model(Hr_model)),
  tar_target(BBCH_model, BBCH_stagemodel(CISO_model)),
  tar_target(Risk_model, Risk_model(BBCH_model)),
  tar_target(Sev_model, Sev_model(Sev_model)),

#  tar_target(plot, plot_model(model, data))
)

