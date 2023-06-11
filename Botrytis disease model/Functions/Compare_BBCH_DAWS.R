#library(lubridate)
#library(tidyverse)
#library(data.table)
#library(compare)

## Find which stage today is in. using APSIM file
Compare_BBCH_DAWS <- function(data_input,Report_Code_BBCH) {
  #data_input <- Gs_CISO

  GS <- Report_Code_BBCH[which(Report_Code_BBCH$Clock.Today.Year == data_input$year & Report_Code_BBCH$DAWS == data_input$doy.july1st),]$Code_BBCH_Phenology
  
#Report_Code_BBCH$Clock.Today.Year == data_input$year &&   
 # GS <- ifelse(length(GS)> 0,GS,NA)
  return(ifelse(length(GS)> 0,GS,NA))
}

## Find which stage today is in. # test BBCH stage data
Compare_BBCH_DAWS_orign <- function(data_input) {
  # df_BBCH <- read_excel(file.path(InputData, 'BBCH.xlsx'))    %>% 
  
  Code <- c(0, 1, 3,5,6,7,9,11,12,13,14,15,16,17,18,19,53,55,57,
            60,61,62,63,64,65,66,67,68,69,71,73,75,77,78,79,
            80,81,83,84,85,86,87,88,89,91,92,93,95,97,99)
  
  Description <- c(1)
  Record <- as.Date(c(" ","14/09/2022","17/09/2022","20/09/2022","22/09/2022","24/09/2022","29/09/2022","10/10/2022",
                      "","","","","","","","","12/11/2022","18/11/2022","24/11/2022","28/11/2022","30/11/2022","1/12/2022",
                      "3/12/2022","4/12/2022","5/12/2022","6/12/2022","8/12/2022","10/12/2022","15/12/2022","19/12/2022",
                      "23/12/2022","30/12/2022","7/01/2023","12/01/2023","16/01/2023","22/01/2023","28/01/2023","30/01/2023",
                      "3/02/2023","5/02/2023","20/02/2023","20/03/2023","25/03/2023","30/03/2023","","","1/05/2023","",
                      "31/05/2023",""),format = "%d/%m/%Y")
  
  #BBCH_Base <- read_excel(file.path(InputData, 'BBCH.xlsx'))  
  df_BBCH <- data.frame(Code, Description,Record) %>%
    mutate(year = year(Record), doy= yday(Record)) %>% 
    do(DOY.from.July1st(.))
  
  df_BBCH$dif = data_input$doy.july1st - df_BBCH$doy.july1st
  GS =df_BBCH[which.min(df_BBCH$dif > 0),1]
    
    return(GS)
}

## Method 1 to Match BBCH stage data by using CurrentStageName and date in APSIM file 
Match_BBCH_Code_1 <- function(data_input) {
  
  Code <- c(0, 1, 3,5,6,7,9,11,12,13,14,15,16,17,18,19,53,55,57,
            60,61,62,63,64,65,66,67,68,69,71,73,75,77,78,79,
            80,81,83,84,85,86,87,88,89,91,92,93,95,97,99)
  Description <- c(1)
  Record <- as.Date(c(" ","14/09/2022","17/09/2022","20/09/2022","22/09/2022","24/09/2022","29/09/2022","10/10/2022",
                      "","","","","","","","","12/11/2022","18/11/2022","24/11/2022","28/11/2022","30/11/2022","1/12/2022",
                      "3/12/2022","4/12/2022","5/12/2022","6/12/2022","8/12/2022","10/12/2022","15/12/2022","19/12/2022",
                      "23/12/2022","30/12/2022","7/01/2023","12/01/2023","16/01/2023","22/01/2023","28/01/2023","30/01/2023",
                      "3/02/2023","5/02/2023","20/02/2023","20/03/2023","25/03/2023","30/03/2023","","","1/05/2023","",
                      "31/05/2023",""),format = "%d/%m/%Y")
  
  #BBCH_Base <- read_excel(file.path(InputData, 'BBCH.xlsx'))  
  df_BBCH <- data.frame(Code, Description,Record) %>%
    drop_na(Record)  %>%
    mutate(year = year(Record), 
           doy= yday(Record),
           StageName =  NA,
           Diff_stage = ifelse(c(NA,diff(doy))>=0,c(NA,diff(doy)),c(NA,diff(doy))+365),
           Diff_stage = ifelse(Code == 9 | Code == 61 | Code == 71 | Code == 81, 0, Diff_stage))  %>%
    mutate_if(~'POSIXt' %in% class(.x), as.Date)  %>%
    do(DOY.from.July1st(.)) %>%
    mutate(Diff_stage_July1st = ifelse(c(NA,diff(doy.july1st))>=0,c(NA,diff(doy.july1st)),c(NA,diff(doy.july1st))+365),
           Diff_stage_July1st = ifelse(Code == 9 | Code == 61 | Code == 71 | Code == 81, 0, Diff_stage_July1st),
           Acc_diff_July1st = ave(Diff_stage_July1st, rev(cumsum(rev(Diff_stage_July1st==0))), FUN=cumsum),
           Acc_diff_July1st = ifelse(Code == 9 | Code == 61 | Code == 71 | Code == 81, 0, Acc_diff_July1st))

  df_BBCH[df_BBCH$Code == 9,]$StageName  <- 'BudBurst'
  df_BBCH[df_BBCH$Code == 61,]$StageName <- 'Flowering'
  df_BBCH[df_BBCH$Code == 71,]$StageName <- 'FruitSet'
  df_BBCH[df_BBCH$Code == 81,]$StageName <- 'Veraison'  
  
  df_BBCH <- df_BBCH %>%
    mutate(StageName = ifelse(StageName != "", StageName, NA)) %>%
    fill(StageName) %>%
    filter(!is.na(StageName))
 
  #BBCH code data
  BudBurst_BBCH_Data <- df_BBCH  %>%
    filter(StageName == 'BudBurst')
  

  Flowering_BBCH_Data <- df_BBCH %>%
    filter(StageName == 'Flowering') 
  
  FruitSet_BBCH_Data <- df_BBCH %>%
    filter(StageName == 'FruitSet') 
  
  Veraison_BBCH_Data <- df_BBCH %>%
    filter(StageName == 'Veraison')
  
    ###############################
  #load Apsim file
  #data_input <- read.csv(file.path(InputData, 'Report0.csv'))   
  #Report <- data_input%>%
  Report <- data_input%>%  
    dplyr::select(SimulationID,Experiment,Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
                  FolderName,DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
    filter(SimulationID ==6, Experiment == 'Marlborough_2Cane', CurrentSeason == 2017,FolderName == 'Sauvignonblanc') %>%
    mutate(CurrentStageName = ifelse(CurrentStageName != "", CurrentStageName, NA)) %>%  
    fill(CurrentStageName)  %>%
    group_by(CurrentStageName) %>%
    mutate(Diff_DAWS = ifelse(c(NA,diff(DAWS))>=0,c(NA,diff(DAWS)),c(NA,diff(DAWS))+365),
           Acc_diff = ave(Diff_DAWS, rev(cumsum(rev(Diff_DAWS==0))), FUN=cumsum),
           Code_BBCH = NA) %>%
    mutate(Diff_DAWS = coalesce(Diff_DAWS, 0)) %>%
    mutate(Acc_diff = coalesce(Acc_diff, 0))
  
 
  #df_BBCH[df_BBCH$Code == 9,]$StageName  <- 'BudBurst'
  #df_BBCH[df_BBCH$Code == 61,]$StageName <- 'Flowering'
  #df_BBCH[df_BBCH$Code == 71,]$StageName <- 'FruitSet'
  #df_BBCH[df_BBCH$Code == 81,]$StageName <- 'Veraison'  
  

  ## BudBurst stage: fill the BBCH code at the BudBurst stage
  #  BudBurst_Data <- Report1 %>%
  #    filter(CurrentStageName == 'BudBurst')
  
  BudBurst_CodeofBBCH = data_frame()
  for(id in 1:nrow(BudBurst_Data<- Report %>%
                   filter(CurrentStageName == 'BudBurst'))) {
    temp_BudBurst = BudBurst_Data[id,]
    
    for (id_BBCH in 1:nrow(BudBurst_BBCH_Data)) {
      temp_BBCH = BudBurst_BBCH_Data[id_BBCH,] 
      
      if(isTRUE( temp_BudBurst$Acc_diff == temp_BBCH$Acc_diff_July1st) == TRUE )   {
        temp_BudBurst$Code_BBCH = temp_BBCH$Code
      } 
    }
    BudBurst_CodeofBBCH = rbind(BudBurst_CodeofBBCH, temp_BudBurst)
  }
  
  ## Flowering stage: fill the BBCH code at the Flowering stage
  
  Flowering_CodeofBBCH = data_frame()
  for(id in 1:nrow(Flowering_Data<- Report %>%
                   filter(CurrentStageName == 'Flowering'))) {
    temp_Flowering = Flowering_Data[id,]
    
    for (id_BBCH in 1:nrow(Flowering_BBCH_Data)) {
      temp_BBCH = Flowering_BBCH_Data[id_BBCH,] 
      
      if(isTRUE(temp_Flowering$Acc_diff == temp_BBCH$Acc_diff_July1st)==TRUE  )   {
        temp_Flowering$Code_BBCH = temp_BBCH$Code
      } 
    }
    Flowering_CodeofBBCH = rbind(Flowering_CodeofBBCH, temp_Flowering)
  }
  
  ## FruitSet stage: fill the BBCH code at the FruitSet stage
  
  FruitSet_CodeofBBCH = data_frame()
  for(id in 1:nrow(FruitSet_Data<- Report %>%
                   filter(CurrentStageName == 'FruitSet' ))) {
    temp_FruitSet = FruitSet_Data[id,]
    
    for (id_BBCH in 1:nrow(FruitSet_BBCH_Data)) {
      temp_BBCH = FruitSet_BBCH_Data[id_BBCH,] 
      
      if(isTRUE(temp_FruitSet$Acc_diff == temp_BBCH$Acc_diff_July1st)==TRUE )   {
        temp_FruitSet$Code_BBCH = temp_BBCH$Code
      }  
    }
    FruitSet_CodeofBBCH = rbind(FruitSet_CodeofBBCH, temp_FruitSet)
  }
  
  ## Veraison stage: fill the BBCH code at the Veraison stage
  
  Veraison_CodeofBBCH = data_frame()
  for(id in 1:nrow(Veraison_Data<- Report %>%
                   filter(CurrentStageName == 'Veraison'))) {
    temp_Veraison = Veraison_Data[id,]
    
    for (id_BBCH in 1:nrow(Veraison_BBCH_Data)) {
      temp_BBCH = Veraison_BBCH_Data[id_BBCH,] 
      
      if(isTRUE(temp_Veraison$Acc_diff == temp_BBCH$Acc_diff_July1st)==TRUE )   {
        temp_Veraison$Code_BBCH = temp_BBCH$Code
      }
      
    }
    Veraison_CodeofBBCH = rbind(Veraison_CodeofBBCH, temp_Veraison)
  }
  
  Report_CodeofBBCH <- rbind(BudBurst_CodeofBBCH,Flowering_CodeofBBCH,FruitSet_CodeofBBCH,Veraison_CodeofBBCH) %>% fill(Code_BBCH)
  
  #fill the blanks with previous value
  #Report_total <- Report_CodeofBBCH %>% fill(Code_BBCH)
  
  return(Report_CodeofBBCH)
}

## Method 2 to Match BBCH stage data by using the value of Grapevine.Phenology.Stage
Match_BBCH_APSIM <- function(data_input) {
  #load Apsim file
  #data_input <- read.csv(file.path(InputData, 'Report0.csv'))   
  
  Report <- data_input%>%  
    #    dplyr::select(SimulationID,Experiment,Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
    #                  FolderName,DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
    #   filter(SimulationID ==6, CurrentSeason == 2009, FolderName == 'Sauvignonblanc') %>%
    dplyr::select(Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
                  DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
#    filter(CurrentSeason == yearofmet) %>%
    mutate(Code_BBCH_Phenology = case_when(Grapevine.Phenology.Stage >= 3.00 & Grapevine.Phenology.Stage < 3.13 ~ 9,
                                           Grapevine.Phenology.Stage >= 3.13 & Grapevine.Phenology.Stage < 3.63 ~ 11,
                                           Grapevine.Phenology.Stage >= 3.63 & Grapevine.Phenology.Stage < 3.71 ~ 53,
                                           Grapevine.Phenology.Stage >= 3.71 & Grapevine.Phenology.Stage < 3.86 ~ 55,
                                           Grapevine.Phenology.Stage >= 3.86 & Grapevine.Phenology.Stage < 3.96 ~ 57,
                                           Grapevine.Phenology.Stage >= 3.96 & Grapevine.Phenology.Stage < 4.00 ~ 60,
                                           
                                           Grapevine.Phenology.Stage >= 4.00 & Grapevine.Phenology.Stage < 4.13 ~ 61,
                                           Grapevine.Phenology.Stage >= 4.13 & Grapevine.Phenology.Stage < 4.26 ~ 62,
                                           Grapevine.Phenology.Stage >= 4.26 & Grapevine.Phenology.Stage < 4.34 ~ 63,
                                           Grapevine.Phenology.Stage >= 4.34 & Grapevine.Phenology.Stage < 4.41 ~ 64,
                                           Grapevine.Phenology.Stage >= 4.41 & Grapevine.Phenology.Stage < 4.49 ~ 65,
                                           Grapevine.Phenology.Stage >= 4.49 & Grapevine.Phenology.Stage < 4.63 ~ 66,
                                           Grapevine.Phenology.Stage >= 4.63 & Grapevine.Phenology.Stage < 4.72 ~ 67,
                                           Grapevine.Phenology.Stage >= 4.72 & Grapevine.Phenology.Stage < 5.00 ~ 68,
                                           
                                           Grapevine.Phenology.Stage >= 5.00 & Grapevine.Phenology.Stage < 5.10 ~ 71,
                                           Grapevine.Phenology.Stage >= 5.10 & Grapevine.Phenology.Stage < 5.24 ~ 73,
                                           Grapevine.Phenology.Stage >= 5.24 & Grapevine.Phenology.Stage < 5.47 ~ 75,
                                           Grapevine.Phenology.Stage >= 5.47 & Grapevine.Phenology.Stage < 5.59 ~ 77,
                                           Grapevine.Phenology.Stage >= 5.59 & Grapevine.Phenology.Stage < 5.72 ~ 78,
                                           Grapevine.Phenology.Stage >= 5.72 & Grapevine.Phenology.Stage < 5.91 ~ 79,
                                           Grapevine.Phenology.Stage >= 5.91 & Grapevine.Phenology.Stage < 6.00 ~ 80,
                                           
                                           Grapevine.Phenology.Stage >= 6.00 & Grapevine.Phenology.Stage < 6.01 ~ 81,
                                           Grapevine.Phenology.Stage >= 6.01 & Grapevine.Phenology.Stage < 6.02 ~ 83,
                                           Grapevine.Phenology.Stage >= 6.02 & Grapevine.Phenology.Stage < 6.03 ~ 84,
                                           Grapevine.Phenology.Stage >= 6.03 & Grapevine.Phenology.Stage < 6.04 ~ 85,
                                           Grapevine.Phenology.Stage >= 6.04 & Grapevine.Phenology.Stage < 6.05 ~ 86,
                                           Grapevine.Phenology.Stage >= 6.05 & Grapevine.Phenology.Stage < 6.06 ~ 87,
                                           Grapevine.Phenology.Stage >= 6.06 & Grapevine.Phenology.Stage < 6.07 ~ 88,
                                           Grapevine.Phenology.Stage >= 6.07 & Grapevine.Phenology.Stage < 6.15 ~ 89
    )
    
    )
  
  return(Report)
}


## Method 2 to Match BBCH stage data by using the value of Grapevine.Phenology.Stage
Match_BBCH_APSIM2021 <- function(data_input) {
  #load Apsim file
  #data_input <- read.csv(file.path(InputData, 'Report0.csv'))   
  
  Report <- data_input%>%  
    #    dplyr::select(SimulationID,Experiment,Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
    #                  FolderName,DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
    #   filter(SimulationID ==6, CurrentSeason == 2009, FolderName == 'Sauvignonblanc') %>%
    dplyr::select(Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
                  DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
    filter(CurrentSeason == 2021) %>%
    mutate(Code_BBCH_Phenology = case_when(Grapevine.Phenology.Stage >= 3.00 & Grapevine.Phenology.Stage < 3.13 ~ 9,
                                           Grapevine.Phenology.Stage >= 3.13 & Grapevine.Phenology.Stage < 3.63 ~ 11,
                                           Grapevine.Phenology.Stage >= 3.63 & Grapevine.Phenology.Stage < 3.71 ~ 53,
                                           Grapevine.Phenology.Stage >= 3.71 & Grapevine.Phenology.Stage < 3.86 ~ 55,
                                           Grapevine.Phenology.Stage >= 3.86 & Grapevine.Phenology.Stage < 3.96 ~ 57,
                                           Grapevine.Phenology.Stage >= 3.96 & Grapevine.Phenology.Stage < 4.00 ~ 60,
                                           
                                           Grapevine.Phenology.Stage >= 4.00 & Grapevine.Phenology.Stage < 4.13 ~ 61,
                                           Grapevine.Phenology.Stage >= 4.13 & Grapevine.Phenology.Stage < 4.26 ~ 62,
                                           Grapevine.Phenology.Stage >= 4.26 & Grapevine.Phenology.Stage < 4.34 ~ 63,
                                           Grapevine.Phenology.Stage >= 4.34 & Grapevine.Phenology.Stage < 4.41 ~ 64,
                                           Grapevine.Phenology.Stage >= 4.41 & Grapevine.Phenology.Stage < 4.49 ~ 65,
                                           Grapevine.Phenology.Stage >= 4.49 & Grapevine.Phenology.Stage < 4.63 ~ 66,
                                           Grapevine.Phenology.Stage >= 4.63 & Grapevine.Phenology.Stage < 4.72 ~ 67,
                                           Grapevine.Phenology.Stage >= 4.72 & Grapevine.Phenology.Stage < 5.00 ~ 68,
                                           
                                           Grapevine.Phenology.Stage >= 5.00 & Grapevine.Phenology.Stage < 5.10 ~ 71,
                                           Grapevine.Phenology.Stage >= 5.10 & Grapevine.Phenology.Stage < 5.24 ~ 73,
                                           Grapevine.Phenology.Stage >= 5.24 & Grapevine.Phenology.Stage < 5.47 ~ 75,
                                           Grapevine.Phenology.Stage >= 5.47 & Grapevine.Phenology.Stage < 5.59 ~ 77,
                                           Grapevine.Phenology.Stage >= 5.59 & Grapevine.Phenology.Stage < 5.72 ~ 78,
                                           Grapevine.Phenology.Stage >= 5.72 & Grapevine.Phenology.Stage < 5.91 ~ 79,
                                           Grapevine.Phenology.Stage >= 5.91 & Grapevine.Phenology.Stage < 6.00 ~ 80,
                                           
                                           Grapevine.Phenology.Stage >= 6.00 & Grapevine.Phenology.Stage < 6.01 ~ 81,
                                           Grapevine.Phenology.Stage >= 6.01 & Grapevine.Phenology.Stage < 6.02 ~ 83,
                                           Grapevine.Phenology.Stage >= 6.02 & Grapevine.Phenology.Stage < 6.03 ~ 84,
                                           Grapevine.Phenology.Stage >= 6.03 & Grapevine.Phenology.Stage < 6.04 ~ 85,
                                           Grapevine.Phenology.Stage >= 6.04 & Grapevine.Phenology.Stage < 6.05 ~ 86,
                                           Grapevine.Phenology.Stage >= 6.05 & Grapevine.Phenology.Stage < 6.06 ~ 87,
                                           Grapevine.Phenology.Stage >= 6.06 & Grapevine.Phenology.Stage < 6.07 ~ 88,
                                           Grapevine.Phenology.Stage >= 6.07 & Grapevine.Phenology.Stage < 6.15 ~ 89
    )
    
    )
  
  return(Report)
}

## Method 2 to Match BBCH stage data by using the value of Grapevine.Phenology.Stage
Match_BBCH_Squv2021 <- function(data_input) {
  #load Apsim file
  #data_input <- read.csv(file.path(InputData, 'Report0.csv'))   
  
  Report <- data_input%>%  
    #    dplyr::select(SimulationID,Experiment,Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
    #                  FolderName,DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
    #   filter(SimulationID ==6, CurrentSeason == 2009, FolderName == 'Sauvignonblanc') %>%
    dplyr::select(Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
                  DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
    filter(CurrentSeason == 2021) %>%
    mutate(Code_BBCH_Phenology = case_when(Grapevine.Phenology.Stage >= 3.00 & Grapevine.Phenology.Stage < 3.13 ~ 9,
                                           Grapevine.Phenology.Stage >= 3.13 & Grapevine.Phenology.Stage < 3.63 ~ 11,
                                           Grapevine.Phenology.Stage >= 3.63 & Grapevine.Phenology.Stage < 3.71 ~ 53,
                                           Grapevine.Phenology.Stage >= 3.71 & Grapevine.Phenology.Stage < 3.86 ~ 55,
                                           Grapevine.Phenology.Stage >= 3.86 & Grapevine.Phenology.Stage < 3.96 ~ 57,
                                           Grapevine.Phenology.Stage >= 3.96 & Grapevine.Phenology.Stage < 4.00 ~ 60,
                                           
                                           Grapevine.Phenology.Stage >= 4.00 & Grapevine.Phenology.Stage < 4.13 ~ 61,
                                           Grapevine.Phenology.Stage >= 4.13 & Grapevine.Phenology.Stage < 4.26 ~ 62,
                                           Grapevine.Phenology.Stage >= 4.26 & Grapevine.Phenology.Stage < 4.34 ~ 63,
                                           Grapevine.Phenology.Stage >= 4.34 & Grapevine.Phenology.Stage < 4.41 ~ 64,
                                           Grapevine.Phenology.Stage >= 4.41 & Grapevine.Phenology.Stage < 4.49 ~ 65,
                                           Grapevine.Phenology.Stage >= 4.49 & Grapevine.Phenology.Stage < 4.63 ~ 66,
                                           Grapevine.Phenology.Stage >= 4.63 & Grapevine.Phenology.Stage < 4.72 ~ 67,
                                           Grapevine.Phenology.Stage >= 4.72 & Grapevine.Phenology.Stage < 5.00 ~ 68,
                                           
                                           Grapevine.Phenology.Stage >= 5.00 & Grapevine.Phenology.Stage < 5.10 ~ 71,
                                           Grapevine.Phenology.Stage >= 5.10 & Grapevine.Phenology.Stage < 5.24 ~ 73,
                                           Grapevine.Phenology.Stage >= 5.24 & Grapevine.Phenology.Stage < 5.47 ~ 75,
                                           Grapevine.Phenology.Stage >= 5.47 & Grapevine.Phenology.Stage < 5.59 ~ 77,
                                           Grapevine.Phenology.Stage >= 5.59 & Grapevine.Phenology.Stage < 5.72 ~ 78,
                                           Grapevine.Phenology.Stage >= 5.72 & Grapevine.Phenology.Stage < 5.91 ~ 79,
                                           Grapevine.Phenology.Stage >= 5.91 & Grapevine.Phenology.Stage < 6.00 ~ 80,
                                           
                                           Grapevine.Phenology.Stage >= 6.00 & Grapevine.Phenology.Stage < 6.01 ~ 81,
                                           Grapevine.Phenology.Stage >= 6.01 & Grapevine.Phenology.Stage < 6.02 ~ 83,
                                           Grapevine.Phenology.Stage >= 6.02 & Grapevine.Phenology.Stage < 6.03 ~ 84,
                                           Grapevine.Phenology.Stage >= 6.03 & Grapevine.Phenology.Stage < 6.04 ~ 85,
                                           Grapevine.Phenology.Stage >= 6.04 & Grapevine.Phenology.Stage < 6.05 ~ 86,
                                           Grapevine.Phenology.Stage >= 6.05 & Grapevine.Phenology.Stage < 6.06 ~ 87,
                                           Grapevine.Phenology.Stage >= 6.06 & Grapevine.Phenology.Stage < 6.07 ~ 88,
                                           Grapevine.Phenology.Stage >= 6.07 & Grapevine.Phenology.Stage < 6.15 ~ 89
    )
    
    )
  
  return(Report)
}

## Method 3 to Match BBCH stage data by using the value of Grapevine.Phenology.Stage of .db APSIM file
Match_BBCH_Chard2021 <- function(data_input) {
  #load Apsim file
  #data_input <- read.csv(file.path(InputData, 'Report0.csv'))   
  
  Report <- data_input%>%  
#    dplyr::select(SimulationID,Experiment,Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
#                  FolderName,DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
 #   filter(SimulationID ==6, CurrentSeason == 2009, FolderName == 'Sauvignonblanc') %>%
    dplyr::select(Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
                  DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
    filter(CurrentSeason == 2021) %>%
    mutate(Code_BBCH_Phenology = case_when(Grapevine.Phenology.Stage >= 3.00 & Grapevine.Phenology.Stage < 3.13 ~ 9,
                                           Grapevine.Phenology.Stage >= 3.13 & Grapevine.Phenology.Stage < 3.63 ~ 11,
                                           Grapevine.Phenology.Stage >= 3.63 & Grapevine.Phenology.Stage < 3.71 ~ 53,
                                           Grapevine.Phenology.Stage >= 3.71 & Grapevine.Phenology.Stage < 3.86 ~ 55,
                                           Grapevine.Phenology.Stage >= 3.86 & Grapevine.Phenology.Stage < 3.96 ~ 57,
                                           Grapevine.Phenology.Stage >= 3.96 & Grapevine.Phenology.Stage < 4.00 ~ 60,
                                           
                                           Grapevine.Phenology.Stage >= 4.00 & Grapevine.Phenology.Stage < 4.13 ~ 61,
                                           Grapevine.Phenology.Stage >= 4.13 & Grapevine.Phenology.Stage < 4.26 ~ 62,
                                           Grapevine.Phenology.Stage >= 4.26 & Grapevine.Phenology.Stage < 4.34 ~ 63,
                                           Grapevine.Phenology.Stage >= 4.34 & Grapevine.Phenology.Stage < 4.41 ~ 64,
                                           Grapevine.Phenology.Stage >= 4.41 & Grapevine.Phenology.Stage < 4.49 ~ 65,
                                           Grapevine.Phenology.Stage >= 4.49 & Grapevine.Phenology.Stage < 4.63 ~ 66,
                                           Grapevine.Phenology.Stage >= 4.63 & Grapevine.Phenology.Stage < 4.72 ~ 67,
                                           Grapevine.Phenology.Stage >= 4.72 & Grapevine.Phenology.Stage < 5.00 ~ 68,
                                           
                                           Grapevine.Phenology.Stage >= 5.00 & Grapevine.Phenology.Stage < 5.10 ~ 71,
                                           Grapevine.Phenology.Stage >= 5.10 & Grapevine.Phenology.Stage < 5.24 ~ 73,
                                           Grapevine.Phenology.Stage >= 5.24 & Grapevine.Phenology.Stage < 5.47 ~ 75,
                                           Grapevine.Phenology.Stage >= 5.47 & Grapevine.Phenology.Stage < 5.59 ~ 77,
                                           Grapevine.Phenology.Stage >= 5.59 & Grapevine.Phenology.Stage < 5.72 ~ 78,
                                           Grapevine.Phenology.Stage >= 5.72 & Grapevine.Phenology.Stage < 5.91 ~ 79,
                                           Grapevine.Phenology.Stage >= 5.91 & Grapevine.Phenology.Stage < 6.00 ~ 80,
                                           
                                           Grapevine.Phenology.Stage >= 6.00 & Grapevine.Phenology.Stage < 6.01 ~ 81,
                                           Grapevine.Phenology.Stage >= 6.01 & Grapevine.Phenology.Stage < 6.02 ~ 83,
                                           Grapevine.Phenology.Stage >= 6.02 & Grapevine.Phenology.Stage < 6.03 ~ 84,
                                           Grapevine.Phenology.Stage >= 6.03 & Grapevine.Phenology.Stage < 6.04 ~ 85,
                                           Grapevine.Phenology.Stage >= 6.04 & Grapevine.Phenology.Stage < 6.05 ~ 86,
                                           Grapevine.Phenology.Stage >= 6.05 & Grapevine.Phenology.Stage < 6.06 ~ 87,
                                           Grapevine.Phenology.Stage >= 6.06 & Grapevine.Phenology.Stage < 6.07 ~ 88,
                                           Grapevine.Phenology.Stage >= 6.07 & Grapevine.Phenology.Stage < 6.15 ~ 89
    )
    
    )
  
  return(Report)
}

## Method 4 to Match BBCH stage data by using the value of Grapevine.Phenology.Stage of .db APSIM file
Match_BBCH_Gris2021 <- function(data_input) {
  #load Apsim file
  #data_input <- read.csv(file.path(InputData, 'Report0.csv'))   
  
  Report <- data_input%>%  
    #    dplyr::select(SimulationID,Experiment,Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
    #                  FolderName,DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
    #   filter(SimulationID ==6, CurrentSeason == 2009, FolderName == 'Sauvignonblanc') %>%
    dplyr::select(Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
                  DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
    filter(CurrentSeason == 2021) %>%
    mutate(Code_BBCH_Phenology = case_when(Grapevine.Phenology.Stage >= 3.00 & Grapevine.Phenology.Stage < 3.13 ~ 9,
                                           Grapevine.Phenology.Stage >= 3.13 & Grapevine.Phenology.Stage < 3.63 ~ 11,
                                           Grapevine.Phenology.Stage >= 3.63 & Grapevine.Phenology.Stage < 3.71 ~ 53,
                                           Grapevine.Phenology.Stage >= 3.71 & Grapevine.Phenology.Stage < 3.86 ~ 55,
                                           Grapevine.Phenology.Stage >= 3.86 & Grapevine.Phenology.Stage < 3.96 ~ 57,
                                           Grapevine.Phenology.Stage >= 3.96 & Grapevine.Phenology.Stage < 4.00 ~ 60,
                                           
                                           Grapevine.Phenology.Stage >= 4.00 & Grapevine.Phenology.Stage < 4.13 ~ 61,
                                           Grapevine.Phenology.Stage >= 4.13 & Grapevine.Phenology.Stage < 4.26 ~ 62,
                                           Grapevine.Phenology.Stage >= 4.26 & Grapevine.Phenology.Stage < 4.34 ~ 63,
                                           Grapevine.Phenology.Stage >= 4.34 & Grapevine.Phenology.Stage < 4.41 ~ 64,
                                           Grapevine.Phenology.Stage >= 4.41 & Grapevine.Phenology.Stage < 4.49 ~ 65,
                                           Grapevine.Phenology.Stage >= 4.49 & Grapevine.Phenology.Stage < 4.63 ~ 66,
                                           Grapevine.Phenology.Stage >= 4.63 & Grapevine.Phenology.Stage < 4.72 ~ 67,
                                           Grapevine.Phenology.Stage >= 4.72 & Grapevine.Phenology.Stage < 5.00 ~ 68,
                                           
                                           Grapevine.Phenology.Stage >= 5.00 & Grapevine.Phenology.Stage < 5.10 ~ 71,
                                           Grapevine.Phenology.Stage >= 5.10 & Grapevine.Phenology.Stage < 5.24 ~ 73,
                                           Grapevine.Phenology.Stage >= 5.24 & Grapevine.Phenology.Stage < 5.47 ~ 75,
                                           Grapevine.Phenology.Stage >= 5.47 & Grapevine.Phenology.Stage < 5.59 ~ 77,
                                           Grapevine.Phenology.Stage >= 5.59 & Grapevine.Phenology.Stage < 5.72 ~ 78,
                                           Grapevine.Phenology.Stage >= 5.72 & Grapevine.Phenology.Stage < 5.91 ~ 79,
                                           Grapevine.Phenology.Stage >= 5.91 & Grapevine.Phenology.Stage < 6.00 ~ 80,
                                           
                                           Grapevine.Phenology.Stage >= 6.00 & Grapevine.Phenology.Stage < 6.01 ~ 81,
                                           Grapevine.Phenology.Stage >= 6.01 & Grapevine.Phenology.Stage < 6.02 ~ 83,
                                           Grapevine.Phenology.Stage >= 6.02 & Grapevine.Phenology.Stage < 6.03 ~ 84,
                                           Grapevine.Phenology.Stage >= 6.03 & Grapevine.Phenology.Stage < 6.04 ~ 85,
                                           Grapevine.Phenology.Stage >= 6.04 & Grapevine.Phenology.Stage < 6.05 ~ 86,
                                           Grapevine.Phenology.Stage >= 6.05 & Grapevine.Phenology.Stage < 6.06 ~ 87,
                                           Grapevine.Phenology.Stage >= 6.06 & Grapevine.Phenology.Stage < 6.07 ~ 88,
                                           Grapevine.Phenology.Stage >= 6.07 & Grapevine.Phenology.Stage < 6.15 ~ 89
    )
    
    )
  
  return(Report)
}

## Method 4 to Match BBCH stage data by using the value of Grapevine.Phenology.Stage of .db APSIM file
Match_BBCH_Noir2021 <- function(data_input) {
  #load Apsim file
  #data_input <- read.csv(file.path(InputData, 'Report0.csv'))   
  
  Report <- data_input%>%  
    #    dplyr::select(SimulationID,Experiment,Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
    #                  FolderName,DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
    #   filter(SimulationID ==6, CurrentSeason == 2009, FolderName == 'Sauvignonblanc') %>%
    dplyr::select(Clock.Today,Clock.Today.Year,Clock.Today.DayOfYear,
                  DAWS,CurrentSeason,Grapevine.Phenology.Stage,CurrentStageName) %>%
    filter(CurrentSeason == 2021) %>%
    mutate(Code_BBCH_Phenology = case_when(Grapevine.Phenology.Stage >= 3.00 & Grapevine.Phenology.Stage < 3.13 ~ 9,
                                           Grapevine.Phenology.Stage >= 3.13 & Grapevine.Phenology.Stage < 3.63 ~ 11,
                                           Grapevine.Phenology.Stage >= 3.63 & Grapevine.Phenology.Stage < 3.71 ~ 53,
                                           Grapevine.Phenology.Stage >= 3.71 & Grapevine.Phenology.Stage < 3.86 ~ 55,
                                           Grapevine.Phenology.Stage >= 3.86 & Grapevine.Phenology.Stage < 3.96 ~ 57,
                                           Grapevine.Phenology.Stage >= 3.96 & Grapevine.Phenology.Stage < 4.00 ~ 60,
                                           
                                           Grapevine.Phenology.Stage >= 4.00 & Grapevine.Phenology.Stage < 4.13 ~ 61,
                                           Grapevine.Phenology.Stage >= 4.13 & Grapevine.Phenology.Stage < 4.26 ~ 62,
                                           Grapevine.Phenology.Stage >= 4.26 & Grapevine.Phenology.Stage < 4.34 ~ 63,
                                           Grapevine.Phenology.Stage >= 4.34 & Grapevine.Phenology.Stage < 4.41 ~ 64,
                                           Grapevine.Phenology.Stage >= 4.41 & Grapevine.Phenology.Stage < 4.49 ~ 65,
                                           Grapevine.Phenology.Stage >= 4.49 & Grapevine.Phenology.Stage < 4.63 ~ 66,
                                           Grapevine.Phenology.Stage >= 4.63 & Grapevine.Phenology.Stage < 4.72 ~ 67,
                                           Grapevine.Phenology.Stage >= 4.72 & Grapevine.Phenology.Stage < 5.00 ~ 68,
                                           
                                           Grapevine.Phenology.Stage >= 5.00 & Grapevine.Phenology.Stage < 5.10 ~ 71,
                                           Grapevine.Phenology.Stage >= 5.10 & Grapevine.Phenology.Stage < 5.24 ~ 73,
                                           Grapevine.Phenology.Stage >= 5.24 & Grapevine.Phenology.Stage < 5.47 ~ 75,
                                           Grapevine.Phenology.Stage >= 5.47 & Grapevine.Phenology.Stage < 5.59 ~ 77,
                                           Grapevine.Phenology.Stage >= 5.59 & Grapevine.Phenology.Stage < 5.72 ~ 78,
                                           Grapevine.Phenology.Stage >= 5.72 & Grapevine.Phenology.Stage < 5.91 ~ 79,
                                           Grapevine.Phenology.Stage >= 5.91 & Grapevine.Phenology.Stage < 6.00 ~ 80,
                                           
                                           Grapevine.Phenology.Stage >= 6.00 & Grapevine.Phenology.Stage < 6.01 ~ 81,
                                           Grapevine.Phenology.Stage >= 6.01 & Grapevine.Phenology.Stage < 6.02 ~ 83,
                                           Grapevine.Phenology.Stage >= 6.02 & Grapevine.Phenology.Stage < 6.03 ~ 84,
                                           Grapevine.Phenology.Stage >= 6.03 & Grapevine.Phenology.Stage < 6.04 ~ 85,
                                           Grapevine.Phenology.Stage >= 6.04 & Grapevine.Phenology.Stage < 6.05 ~ 86,
                                           Grapevine.Phenology.Stage >= 6.05 & Grapevine.Phenology.Stage < 6.06 ~ 87,
                                           Grapevine.Phenology.Stage >= 6.06 & Grapevine.Phenology.Stage < 6.07 ~ 88,
                                           Grapevine.Phenology.Stage >= 6.07 & Grapevine.Phenology.Stage < 6.15 ~ 89
    )
    
    )
  
  return(Report)
}