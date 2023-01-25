
#*********************************************************
#data manipulation
#*********************************************************
# data manipulation can mainly be done by dplyr package
# select, filter, group_by, mutate, arrange, summarise, rename
# mutate_each, can replace all NA or other values to desired ones
# data_frame can create one dimensional local data frame tbl_df
# do If one of the specialised verbs doesn't do what you need, you can use do()

#saturation vapor pressure as a function of temperature
satvap <- function(temp) { #(kPa)
  return (0.611*exp(17.4*temp/(239+temp)));}  
# Vapour pressure of the air (kPa)
vap <- function(temp,RH) { # (kPa)
  return (RH * satvap(temp)); }

# VPD is vapour pressure deficit of the external air (kPa)
VPD <- function(temp,RH) { # (kPa)
  return (satvap(temp)-vap(temp,RH)); }

# converting factor variables into numeric variable
as.numeric.factor <- function(x) { 
  if(is.factor(x)) return(as.numeric(paste(x)))
  if(is.numeric(x)) return(x)
}

# this function chnage certain numeric numbers into NA
changetoNA <- function(colnum,df) {
  col <- df[,colnum]
  if (is.numeric(col)) {  #edit: verifying column is numeric
    col[col == 0 & is.numeric(col)] <- NA
  }
  return(col)
}
# logistic fitting functions
YIN_beta_function = function(t, Lmax, te, tm) {
  
  Lmax * (1 + (te - t)/(te-tm))* (t/te)^(te/(te-tm)) }

# this create an empty plot and add legend at outside the current figure
# add_legend <- function(...) {
#   opar <- par(fig=c(0, 1, 0, 1), oma=c(0, 0, 0, 0), 
#               mar=c(0, 0, 0, 0), new=TRUE)
#   on.exit(par(opar))
#   plot(0, 0, type='n', bty='n', xaxt='n', yaxt='n')
#   legend(...)
# }

# in tidyr, it offers tools for tidying the data
# gather, spread, complete, extract_numeric, separate
# mtcars %>%
#   unite(vs_am, vs, am) %>%
#   separate(vs_am, c("vs", "am"))
# library(zoo)
# na.locf Generic function for replacing each NA with the most recent non-NA prior to it.

#*********************************************************
# Specific functions for co2 response curve
#*********************************************************
cpa <- function(temp, ci.ppm) { #(ubar)
  return (ci.ppm*0.008314*(273.15+temp)*10/22.4136);
  }  # do not know how does this was converted

#*********************************************************
# fitting functions
#*********************************************************
# logistic fitting functions
PAR_Logistic_4 = function(x, a, b, c, d) {
  
  a + (b - a ) /(1 + exp(-c * (x - d))) }

logistic4.fit <- function(df) {
  
  #df = subset(RFR.dynamic, Row == 1)
     st <- coef(nls(log(RFR) ~ log( PAR_Logistic_4(Thermal.time, a, b, c,d)),
                  df, start = c(a = 0.3, b = 1.1, c = -0.03, d = 400)))
  # gnls is strong in adapting the same starting value
  mod<- gnls(RFR ~ PAR_Logistic_4(Thermal.time, a, b, c,d),  data = df, 
             start = st, #c(a = 0.3, b = 1.1, c = -0.03, d = 400),   
             control=gnlsControl(returnObject=TRUE),weights=varPower())

  return(coef(mod))
}

# multiphase fitting
function_unused <- function(.) {
# expexplinflat <- function(SDT,T0,R1,T1,R2,T2,T3,inlength) {
#   L <- SDT
#   
#   a <- inlength * exp(R1*(T1-T0))
#   c <- a        * exp(R2*(T2-T1))
#   d <- a * R2  * exp(R2*(T2-T1))
#   e <- c + d    * (T3-T2)
#   
#   
#   L[SDT < T0]             <- inlength[1]
#   L[SDT >= T0 & SDT < T1] <- inlength[1] * exp(R1[1]*(SDT[SDT >= T0 & SDT < T1]-T0[1])) 
#   L[SDT >= T1 & SDT < T2] <- a[1]        * exp(R2[1]*(SDT[SDT >= T1 & SDT < T2]-T1[1]))
#   L[SDT >= T2 & SDT < T3] <- c[1] + d[1] * (SDT[SDT >= T2 & SDT < T3]-T2[1])
#   L[SDT >= T3]            <- e[1]
#   
#   return(L)
# }
}
# self starting value function
#m1 <- nls( A_n~ a - b*exp(-c*GDP_per_capita), start = list(a= 2,b= 0.0001, c= 0.1), data=df)
#m1 <- nls( A_n~ SSasymp(GDP_per_capita,a,b,c), data=df)

# mixed linear model
function_unused <- function(.) {
# # # m2 random intercept model
# m2 <- lme(thermal_time ~ 1 + phytomer_rank*treatment, random = ~ 1 |plot/plant, 
#           method = "REML",data=df,na.action ='na.omit') 
# 
# ##model comparation
# AIC(m1,m2)
# anova(m1,m2)
# summary(m2)
# ## model validation
# ##residuals and fitted values 
# plot(m1)
# plot(m2)
# 
# fixed.effects(m2)
# ####standard error for the coef
# m2.se <- sqrt(diag(vcov(m2)))
}
#*********************************************************
# some file operation functions
#*********************************************************
# read all the data files in a folder
function_unused <- function(.) {
# read.files <- function(file.directory) {
#   file_list <- list.files(file.directory)
#   for (file in file_list){
#     if (!exists("dataset")){
#       dataset <- read.table(file.path(file.directory, file), nrows =1, header = FALSE)
#       dataset <- dataset[-1,1:dim(dataset)[2]]
#     }
#     # if the merged dataset does exist, append to it
#     if (exists("dataset")){
#       temp_dataset <-read.table(file.path(file.directory, file))
#       dataset<-rbind(dataset, temp_dataset)
#       rm(temp_dataset)
#     }
#   }
#   return(dataset)
# }

}

# Grab Maestra and put it into the working directory
function_unused <- function(.) {
# setMaestra <- function(filename = "Maestra.out", from = file.path(SourDir,"Maestra"),
#                        to = MaeDir)
# {
#   file.copy(from=file.path(from,filename),to=file.path(to,'Maestra.out'), overwrite= TRUE) 
# }
}


## Grab zip file with the meteorology and overwrite it
function_unused <- function(.) {
# setMet <- function(fileName = "DefaultMet.zip",from = MaeDir, 
#                    to = MaeDir, which = 'metDAY.dat')
# {
#   file.copy(from=file.path(from,fileName),to=file.path(to,fileName),overwrite=TRUE)
#   unzip(file.path(to,fileName),files = which,overwrite = TRUE, exdir = to)
#   file.remove(file.path(to,fileName))
#   if(file.exists(file.path(to,'met.dat'))) file.remove(file.path(to,'met.dat'))
#   file.copy(file.path(to,which),file.path(to,'met.dat'))
#   file.remove(file.path(to,which))
# }
}

#*********************************************************
#ggplot function
#*********************************************************
# library(cowplot)
# cowplot can use save_plot to save the arranged ggplot figures
# final.plot <-  plot_grid(p1, p2, p3, p4, p5, p6, ncol = 2, nrow = 3)
#   
# save_plot(file.path(FigureOutput, "carbon partition trial.pdf"), final.plot, base_height = 17, base_width = 15,
#             base_aspect_ratio = 1.2 # make room for figure legend
# for qucik plot we can use qplot to see the data first

# annotation should specify the place where to put the annotation by factor levels
# ann_text <- data.frame(PhytomerRank = factor(c(5,5,7,7)),
#                        variable = factor(c("Blade","Sheath","Blade","Sheath")),
#                        x = c(70,70,70,70), y = c(52,21,80,21),
#                        label.text = c("A Blade 5","B Sheath 5","C Blade 7","D Sheath 7"))
# limits.leaf.dry <- aes(ymax = total.leaf.dry + se, ymin = total.leaf.dry - se)
# ann_arrows <- data.frame(x = c(167,192,201, 215,222,287),
#                          y = c(0.3,0.3,0.3,0.3,0.3,0.3), 
#                          yend =c(-0.10,-0.1,-0.10,-0.10,-0.10,-0.10),
#                          xend =c(167,192,201, 215, 222,287))
#layout theme
# source(file="Layout_LegendOpts_ScenarioAnalysis.R")
###plotfunction
plot.water.flux <- function(data, #data.2,  # can import many datasets if there are
                            scale.xmin,scale.xmax, scale.xby,
                            scale.ymin,scale.ymax, scale.yby, 
                            xaxis.title,
                            yaxis.title, 
                            legend=FALSE, title=FALSE) {
  p <- ggplot(data = data, aes(x = hours))+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    # facet_wrap(~PhytomerRank + variable, scales = "free") + 
    # geom_text(data = ann_text,aes(x = x, y = y, label = label.text),hjust=0, giude = "none", size = 5 ) +
    # point figure, can use subset to derive part of the data
    # ase map the data, like how do you divide the data and how to show the legend
    # geom_point(data = subset(data, Version == "Obs"), aes(y = value,shape = factor(Treatment)), size = 3) +
    # control the shape of the points manually
    # scale_shape_manual(name ="", values=c(1,2)) +
    
    # control the line shape
    # , colour = Version, only defined it in aes then you can change the shape or colour, like defining the map first
  geom_line(aes(y = water.flux, linetype = factor(type)),size = 0.5) +
    # control the shape of the line
    # can suppress the legend display by  guide = "none"
    scale_linetype_manual(name = "",values=c(1,2,3))+
    # errorbar
    # geom_errorbar(data = data,limits.leaf.dry, width=0.7) +
    # scale_size_manual(name = "",values=c(1.5,1,1,1,1,1,1))+   # size of each line
    # add arrows
    #     geom_segment(data = ann_arrows,aes(x = x, y = y, xend = xend, yend = yend), 
    #                  arrow = arrow(length = unit(0.2, "cm"))) +
    #     #scale_colour_manual("", values=c("red", "blue")) + 
    
  theme(panel.margin = unit(1, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin, scale.xmax),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=scale.ymin, to=scale.ymax, by=scale.yby)) +   
    
    # theme of the layout
    lib.opts.layout 
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}



#*********************************************************
# use stat_summary for ploting mean, errorbar
#*********************************************************
function_unused <- function(.) {
# stat_sum_single <- function(fun, geom="point", ...) {
#   stat_summary(fun.y=fun, colour="red", geom=geom, size = 1, ...)
# }
# 
# errorUpper <- function(x){ 
#   x.mean <- mean(x) 
#   x.sd <- sd(x) 
#   SEM <- x.sd / (sqrt(length(x))) 
#   return(x.mean + (SEM*1.96)) 
# } 
# 
# errorLower <- function(x){ 
#   x.mean <- mean(x) 
#   x.sd <- sd(x) 
#   SEM <- x.sd / (sqrt(length(x))) 
#   return(x.mean - (SEM*1.96)) 
# } 
# ggplot(C_N.data, aes(x=day, y=value.N.content, group=as.factor(stress), 
#                      colour=as.factor(stress))) + 
#   facet_wrap(~N.content,  scales = "free_y") +
#   stat_summary(fun.y=mean, geom="point") + 
#   stat_summary(fun.y=mean, geom="line")+ 
#   stat_summary(fun.ymax = errorUpper, fun.ymin = errorLower, geom = 
#                  "errorbar") + ggtitle("Gourieroux 2003 (HHH) and (LLL)") +
#   ylab("N content  (mg g-1)")
}


















