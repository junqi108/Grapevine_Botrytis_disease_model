
## Find which stage today is in. # test BBCH stage data
Compare_BBCH_DAWS <- function(df_BBCH, date_input) {
#  df_BBCH = read_excel(filepath, file, sheet = "Sheet1")
  
  df_BBCH  <- 
    df_BBCH %>% 
    mutate(year = year(df_BBCH$Record), doy= yday(df_BBCH$Record)) %>% 
    do(DOY.from.July1st(.))

    df_BBCH$dif = date_input$doy.july1st - df_BBCH$doy.july1st
    GS =df_BBCH[which.min(df_BBCH$dif > 0),1]
    
    return(GS)
}

#    GS =df_BBCH[which.min(difftime(Date, df_BBCH$temp, units = "days") > 0),1]
