#test targets package

library(tidyverse)
library(lubridate)
library(readxl)
library(RcppRoll)
library(MASS)
library(targets)

library(ggplot2)
library(gtable)
library(tibble)

source("R/functions.R")
file <- "data.csv"
data <- get_data(file)
model <- fit_model(data)
plot <- plot_model(model, data)
  