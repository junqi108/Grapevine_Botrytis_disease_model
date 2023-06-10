# Files
INPUT_DATA <- here("data", "sauvignon_blanc_severity_calc.csv")

# Stan
N_CORES <- 4
N_CHAINS <- 4
FILE_REFIT <- "never"  
SAMPLING_ALGORITHM <- "sampling"  
MODEL_METRICS <- c("loo", "waic")
STAN_BACKEND <- "cmdstanr"
WITHIN_CHAIN_THREADS <- "3"
STAN_INIT_START <- "0"
STAN_ITER <- 2000
STAN_WARMUP <- 1000
STAN_ADAPT_DELTA <- 0.9
STAN_SAMPLE_PRIOR <- F
STAN_GAUSSIAN_FAMILY <- gaussian()
STAN_CONSTANT_PRIOR <- 'constant(1)'
STAN_NDRAWS = 25

# Random
SEED <- 100
