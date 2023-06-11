
Cal_Teq <- function(T, Tmin, Tmax){
  # 
  Teq = (T - Tmin)/(Tmax - Tmin)
  return(Teq)
}

Cal_MYGR <- function(Teq, Mf){
  MYGR = (3.78*Teq^0.9*(1-Teq))^0.475*Mf
  return(MYGR)
}

Cal_SPOR <- function(Teq, RH){
  SPOR = (3.7*Teq^0.9*(1-Teq))^10.49*(-3.595+0.097*RH-0.0005*RH^2)
  if (!is.na(SPOR) && SPOR < 0) {
    SPOR = 0
  }
  if (!is.na(SPOR) && SPOR > 1) {
    SPOR = 1
  }
  return(SPOR)
}

Cal_CISO <- function(MYGR, SPOR){
  CISO = rollmean(MYGR + SPOR, k = 7, align = "right", fill = NA)
  return(CISO)
}



Cal_RIS1 <- function(CISO,Teq, WD, GS){

  SUS = -379.09*(GS/100)^3+671.25*(GS/100)^2-390.33*(GS/100)+75.209
  if (SUS > 1)   {
    SUS = 1
  }
  INF =  (3.56*Teq^0.99*(1-Teq))^0.71/(1+exp(1.85-0.19*WD))*SUS
  RIS = CISO*INF
  return(data.frame(SUS,INF,RIS))
}


Cal_RIS2 <- function(CISO, Teq, WD, GS){
  
  SUS = 5*10^-17*exp(0.4219*GS)
  if (SUS > 1)   {
    SUS = 1
  }
  INF =  (6.416*Teq^1.292*(1-Teq))^0.469*exp(-2.3*exp(-0.048*WD))*SUS
  RIS = CISO*INF

  return(data.frame(SUS,INF,RIS))
}

Cal_RIS3 <- function(MYGR, Teq, RH, GS){
  
  SUS = 0.0546*GS-3.87
  if (SUS > 1)   {
    SUS = 1
  }
  
  INF =  (7.75*Teq^2.14*(1-Teq))^0.469/(1+exp(35.36-40.26*(RH/100)))*SUS;
  RIS = MYGR*INF;
  return(data.frame(SUS,INF,RIS))
}

Cal_RIS1_Eff <- function(CISO, Teq, WD, GS, Factor) {
  SUS_Eff <- (-379.09*(GS/100)^3 + 671.25*(GS/100)^2 - 390.33*(GS/100) + 75.209) * Factor
  SUS_Eff <- min(SUS_Eff, 1)
  
  INF_Eff <- (6.416*Teq^1.292*(1-Teq))^0.469 * exp(-2.3*exp(-0.048*WD)) * SUS_Eff
  
  RIS_Eff <- CISO * INF_Eff
  
  data.frame(SUS_Eff, INF_Eff, RIS_Eff)
}


Cal_RIS2_Eff <- function(CISO,Teq, WD, GS,Factor){
  
  SUS_Eff = (5*10^-17*exp(0.4219*GS)) * Factor
  SUS_Eff <- min(SUS_Eff, 1)
  
  INF_Eff =  (6.416*Teq^1.292*(1-Teq))^0.469*exp(-2.3*exp(-0.048*WD))*SUS_Eff
  
  RIS_Eff = CISO * INF_Eff
  data.frame(SUS_Eff, INF_Eff, RIS_Eff)
}

Cal_RIS3_Eff <- function(MYGR,Teq, RH, GS,Factor){
  
  SUS_Eff = (0.0546*GS-3.87) * Factor
  SUS_Eff <- min(SUS_Eff, 1)
  
  INF_Eff =  (7.75*Teq^2.14*(1-Teq))^0.469/(1+exp(35.36-40.26*(RH/100)))*SUS_Eff  
  RIS_Eff = MYGR * INF_Eff
  data.frame(SUS_Eff, INF_Eff, RIS_Eff)
}
