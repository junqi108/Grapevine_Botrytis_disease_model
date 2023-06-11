## Create new management file

Create_New_Management <- function(file) {

# 
  
  Managements_new <- file %>%  
    slice(rep(1:n(),times = Duration)) %>% 
    group_by(Management_Action,Date) %>%
    mutate(year = year(Date),
           doy = yday(Date),
           doy_eff = doy+row_number()-1) 

#  if(leap_year(Managements_new$year) == TRUE && Managements_new$doy_eff > 366) {
#    Managements_new$doy_eff = Managements_new$doy_eff -366
#    Managements_new$doy.july1st_eff = Managements_new$doy.july1st_eff -366
#    Managements_new$year = Managements_new$year +1
#  } else {
#    Managements_new$doy_eff = Managements_new$doy_eff
#    Managements_new$doy.july1st_eff = Managements_new$doy.july1st_eff}
  
  New <- subset(Managements_new, select = -c(Application_Rate)) %>%
    rename(Fungicide_Date = Date)  %>%
    do(DOY.from.July1st(.) %>%
    mutate(doy.july1st = doy.july1st+row_number()-1) )
 

  return(New)
}

#Managements_old <- read_excel(file.path(InputData, 'Management.xlsx')) %>%
#  mutate(year = year(Date),doy = yday(Date))  %>%
#  do(DOY.from.July1st(.))

#Managements_new <- Managements_old %>%
#  slice(rep(1:n(),times = Duration)) %>%
#  group_by(`Management Action`,Date) %>%
#  mutate(doy0=doy+row_number()-1)
