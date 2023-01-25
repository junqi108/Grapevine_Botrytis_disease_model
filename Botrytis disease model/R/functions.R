get_data <- function(file) {
  read.csv(file, col_types = cols(), header = TRUE, sep = ",") 
}

get2_data <- function(file) {
  read_excel(file, col_types = cols())}

CalHr_model <- function(data) {
  for(DOY in  1:365) {
    Temp_data = data[which(data$year == 2000 & data$day == DOY),]
    Moist_data = Temp_data[which(Temp_data$mean.rh >= 0.9),]
    Cal_data = data.frame(Temp_data[1,9:10],Temp_data[1,1],mean(Temp_data$mean.rh),
                        suppressWarnings(mean(Temp_data$mean.ta)),suppressWarnings(min(Temp_data$mean.ta)),
                        suppressWarnings(max(Temp_data$mean.ta)),count(Moist_data))
    colnames(Cal_data) = c('Year', 'Stn','DOY','RH','T','Tmin','Tmax','WD')
    Cal_daily = bind_rows(Cal_daily,Cal_data)
  }
  Cal_daily = na.omit(Cal_daily)
}


## Calculate CISO
CalCISO_model <- function(data) {
  Tmin_spor = 0
  Tmin_mygr = 0
  Tmax_spor = 35
  Tmax_mygr = 40 
  #CISO =  data.frame()
  for(id in 1:nrow(data)) {
    Temp_CISO = data[id,]
    Temp_CISO$Teq = Cal_Teq(Temp_CISO$T, Temp_CISO$Tmin, Temp_CISO$Tmax)
    Temp_CISO$Teq_spor = Cal_Teq(Temp_CISO$T, Tmin_spor, Tmax_spor)
    Temp_CISO$Teq_mygr = Cal_Teq(Temp_CISO$T, Tmin_mygr, Tmax_mygr)
    Temp_CISO$Mf = Temp_CISO$WD/24.0
    Temp_CISO$MYGR = Cal_MYGR(Temp_CISO$Teq_mygr, Temp_CISO$Mf)
    Temp_CISO$SPOR = Cal_SPOR(Temp_CISO$Teq_spor, Temp_CISO$RH)
    Temp_CISO$date = as.Date(Temp_CISO$DOY, origin = as.Date(paste0(Temp_CISO$Year,"-01-01")))
    CISO =  bind_rows(CISO,Temp_CISO)
    }
  CISO$CISO = Cal_CISO(CISO$MYGR, CISO$SPOR)
#rownames(CISO) = seq(1,nrow(CISO),1)
}


BBCH_stagemodel <- function(data) {
  ## Determine which stage today is in.
  #GS_data = data.frame()
  Date_pre <-
    data %>%
    mutate(year = year(date), doy= yday(date)) %>%
    do(DOY.from.July1st(.))
  for (idc in 1:nrow(Date_pre)) {
    Gs_CISO = Date_pre[idc,]
    Gs_CISO$GS = Compare_BBCH_DAWS(Gs_CISO)
    GS_data = bind_rows(GS_data,Gs_CISO)
    }  
#rownames(GS_data) = seq(1,nrow(GS_data),1)
}

Risk_model <- function(data) {
# The model calculations begin when grape inflorescences are clearly visible and ends when berries are ripe for harvest, with a time step of 1 day.
#Risk_data = data.frame()
  for (idr in 1:nrow(data)) {
    Temp_Risk = data[idr,]
    if (Temp_Risk$GS >= 53 & Temp_Risk$GS <= 73) {
    ## in the first infection window(stage 53-73),calculate an infection rate on inflorescences and young clusters
      Temp_Risk$RIS1 = Cal_RIS1(Temp_Risk$CISO, Temp_Risk$Teq, Temp_Risk$WD, Temp_Risk$GS)
      } else if (Temp_Risk$GS >= 79 & Temp_Risk$GS <= 89) {
    
    # (stage 79-89) in the second infection window,calculate two infection rates on ripening berries: one for conidial infection(INF2) and another for berry-to-berry infection(INF3)
    # Infection rate for conidia infection:
        Temp_Risk$RIS2 = Cal_RIS2(Temp_Risk$CISO, Temp_Risk$Teq, Temp_Risk$WD, Temp_Risk$GS)
    
    ## Infection rate for berry-to-berry infection:
        Temp_Risk$RIS3 = Cal_RIS3(Temp_Risk$MYGR, Temp_Risk$Teq, Temp_Risk$RH, Temp_Risk$GS)
        }
    Risk_data = bind_rows(Risk_data,Temp_Risk)
  }
}


Sev_model <- function(data) {
##Calculate accumulated severity
#SEV_data = Risk_data
  data$RIS1[is.na(data$RIS1)] = 0
  data$RIS2[is.na(data$RIS2)] = 0
  data$RIS3[is.na(data$RIS3)] = 0
  data$SEV1 = cumsum(data$RIS1)
  data$SEV23= cumsum(data$RIS2+data$RIS3)
  for (ids in 1:nrow(data)) {
    if (data$RIS1[ids] == 0 ) {
      data$SEV1[ids] = NA
      }
    if (data$SEV23[ids] == 0 ) {
      data$SEV23[ids] = NA
    }
  }
}
