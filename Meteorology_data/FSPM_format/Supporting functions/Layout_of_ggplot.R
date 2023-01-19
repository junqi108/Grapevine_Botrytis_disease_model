##http://docs.ggplot2.org/0.9.2.1/theme.html
##http://research.stowers-institute.org/efg/R/Color/Chart/ColorChart.pdf

 l_ply(list('doBy', 'gridExtra', 'graphicsQC','grid'), require, character.only = T)

# plot.larvae <- function(data, title, legend=FALSE) 
lib.opts.layout  <- theme(
  axis.ticks = element_line(colour = "black"),
  axis.text.y = element_text(colour = "black", size=14),
  axis.text.x = element_text(colour = "black", angle = 0, size=14),
  axis.title.x = element_text(colour = "black",  size = 14, vjust=-1),
  axis.title.y = element_text(colour = "black",  size = 14, angle=90, vjust=0.2),
  plot.margin = unit(c(1, 1, 1, 1), "lines"),
  # panel.background = element_rect(colour = "black", fill = 'white'),
  # to overcome the strange proble that border become white
  # panel.border=element_rect(fill=NA),
  panel.border = element_rect(colour = "black"),
  panel.grid.major = element_blank(),
  # panel.grid.major = element_line(colour = "black"),
  panel.grid.minor = element_blank(),
  panel.background = element_blank(),
    # title for the facet
  # strip.text.x = element_blank(),
  strip.text.x = element_blank(),
  strip.background = element_blank()
  
)

lib.opts.legend    <- theme(
    legend.title= element_text(size=0), 
    legend.justification = 0,
    legend.background = element_rect(fill = "#ffffff"),
    legend.text = element_text(size=14),
    legend.key.size = unit(0.67, "cm"),
    legend.key = element_rect(fill="white", size=0, colour="white"),
    #legend.position= c(0.0,0.85), #'bottom', #
    legend.position= 'bottom', #
    #legend.position= c(0.6,0.75), #'bottom', # 
    #legend.position= c(0.53,0.75), #'bottom', # 
    legend.box = "vertical"  # 'horizontal'   
    
    
    
)
              

lib.opts.nolegend  <- theme( legend.position="none")




