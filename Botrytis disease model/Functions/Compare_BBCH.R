## Find which stage today is in. # test BBCH stage data
Compare_BBCH <- function(date_input) {
  df_BBCH = data.frame(read_excel("Data/BBCH.xlsx", sheet = "Sheet1"))
  Date = as.Date(date_input$date)
#  Date = as.Date("2000-09-18")
  year(df_BBCH$Normal) = year(Date)
  year(df_BBCH$Record) = year(Date)
  
  for (i in nrow(df_BBCH) ) {

   if (!is.na(df_BBCH$Record[i])) {
     df_BBCH$temp = df_BBCH$Record
     df_BBCH$dif = difftime(Date, df_BBCH$Record, units = "days")
     } else{
       df_BBCH$temp = df_BBCH$Normal
       df_BBCH$dif = difftime(Date, df_BBCH$Normal, units = "days")
     }
    #DOY =which.min(difftime(Date, df_BBCH$temp, units = "days") > 0)
     GS =df_BBCH[which.min(difftime(Date, df_BBCH$temp, units = "days") > 0),1]
  }

  
  return(GS)
  }
