
DOY.from.July1st <- function(df) { 
  df.new <- 
    df %>% 
    mutate(doy.july1st =  ifelse(leap_year(year),ifelse(doy > 182,doy-182, doy+184),
                                 ifelse(doy > 181,doy-181, doy+184)), 
           season = ifelse(leap_year(year), 
                           ifelse(doy > 182, paste(year, year+1, sep = '-'), paste(year-1, year, sep = '-')),
                           ifelse(doy > 181, paste(year, year+1, sep = '-'), paste(year-1, year, sep = '-'))),
           previous.season = ifelse(leap_year(year), 
                                    ifelse(doy > 182, paste(year-1, year, sep = '-'), paste(year-2, year-1, sep = '-')),
                                    ifelse(doy > 181, paste(year-1, year, sep = '-'), paste(year-2, year-1, sep = '-'))))
  return(df.new)
}


calcDAWS <- function(year, doy, WinterSolsticeDOY=172) { 
  #calculating Days Since Winter Solstice (DAWS)
  #172
  daws =  ifelse(leap_year(year-1),
                 ifelse(doy >= WinterSolsticeDOY, doy - WinterSolsticeDOY, 366 - WinterSolsticeDOY + doy),
                 ifelse(doy >= WinterSolsticeDOY, doy - WinterSolsticeDOY, 365 - WinterSolsticeDOY + doy))
  return(daws)
}
