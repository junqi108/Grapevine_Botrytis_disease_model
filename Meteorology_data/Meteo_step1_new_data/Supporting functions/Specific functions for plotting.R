# use parse in geom_text
# custom_label <- data.frame(cyl = c(4,6,8),
#                            wt = c(5,5,5),
#                            lab = c('X^2','X^2','X^2'),
#                            lab2 =   lab2 = c("chi^2*':'~18.5*', p =3e-04'",
#                                              "chi^2*':'~4.7*', p =0.1909'",
#                                              "chi^2*':'~15.3*', p =0.0016'"))
# p + facet_grid(. ~ cyl) +
#   geom_text(data = custom_label, aes(x=30, y=wt, label = lab2), size = 3, parse = T)

##*********************************************************##
## dry weight
##*********************************************************##
plot.berry.dw <- function(data.1, data.2,  # can import many datasets if there are
                     scale.xmin,scale.xmax, scale.xby,
                     scale.ymin,scale.ymax, scale.yby, 
                     xaxis.title,
                     yaxis.title, 
                     label.text,
                     legend=FALSE, title=FALSE) {
  p <- ggplot(data = data.1,aes(x = daf))+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
     facet_wrap(~treat, nrow = 1, ncol = 2)+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
 
    geom_point(aes(y = dw_mean), size = 3, shape = 1) +
    geom_errorbar(aes(ymin = dw_mean-dw_se, ymax = dw_mean + dw_se ), width = 0.15 *scale.xby) +
    # geom_smooth(method = "lm", formula = y~x, linetype = 2, color = 'black') + #95% of the confidence interval
    # scale_shape_manual(name ="", values=c(1,2)) +
    geom_line(data = data.2, aes(y = dry.weight)) +
    geom_text(data = ann_text,aes(x = x, y = y, label = label.text),hjust=0, size = 5 ) +
    # scale_colour_hue(h=c(200, 360), l=70, c=100)+
    # geom_abline(slope = 1, intercept = 0, col = 'black') +
    # control the shape of the line
    # can suppress the legend display by  guide = "none"
    #scale_linetype_manual(name = "",values=c(1,2,3,4,5,6,1))+
    # errorbar
    #
    #scale_size_manual(name = "",values=c(1.5,1,1,1,1,1,1))+   # size of each line
    # add arrows
    #     geom_segment(data = ann_arrows,aes(x = x, y = y, xend = xend, yend = yend), 
    #                  arrow = arrow(length = unit(0.2, "cm"))) +
  #     #scale_colour_manual("", values=c("red", "blue")) + 
  
  theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    #scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.95, scale.xmax*1.05),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.97, scale.xmax*1.02),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=scale.ymin, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
     lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}

##*********************************************************##
## fresh weight
##*********************************************************##
plot.berry.fw <- function(data.1, data.2,  # can import many datasets if there are
                          scale.xmin,scale.xmax, scale.xby,
                          scale.ymin,scale.ymax, scale.yby, 
                          xaxis.title,
                          yaxis.title, 
                          label.text,
                          legend=FALSE, title=FALSE) {
  p <- ggplot(data = data.1,aes(x = daf))+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    facet_wrap(~treat, nrow = 1, ncol = 2)+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    
    geom_point(aes(y = fw_mean), size = 3, shape = 1) +
    geom_errorbar(aes(ymin = fw_mean - fw_se, ymax = fw_mean + fw_se ), width = 0.15 *scale.xby) +
    # geom_smooth(method = "lm", formula = y~x, linetype = 2, color = 'black') + #95% of the confidence interval
    # scale_shape_manual(name ="", values=c(1,2)) +
    geom_line(data = data.2, aes(y = fresh.weight)) +
    geom_text(data = ann_text,aes(x = x, y = y, label = label.text),hjust=0, size = 5 ) +
    # scale_colour_hue(h=c(200, 360), l=70, c=100)+
    # geom_abline(slope = 1, intercept = 0, col = 'black') +
    # control the shape of the line
    # can suppress the legend display by  guide = "none"
    #scale_linetype_manual(name = "",values=c(1,2,3,4,5,6,1))+
    # errorbar
    #
    #scale_size_manual(name = "",values=c(1.5,1,1,1,1,1,1))+   # size of each line
    # add arrows
    #     geom_segment(data = ann_arrows,aes(x = x, y = y, xend = xend, yend = yend), 
    #                  arrow = arrow(length = unit(0.2, "cm"))) +
  #     #scale_colour_manual("", values=c("red", "blue")) + 
  
  theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    #scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.95, scale.xmax*1.05),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.97, scale.xmax*1.02),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=scale.ymin, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}


##*********************************************************##
## sugar concentration  ########
##*********************************************************##

plot.berry.sugarConcentration <- function(data.1, data.2,  # can import many datasets if there are
                          scale.xmin,scale.xmax, scale.xby,
                          scale.ymin,scale.ymax, scale.yby, 
                          xaxis.title,
                          yaxis.title, 
                          label.text,
                          legend=FALSE, title=FALSE) {
  p <- ggplot(data = data.1,aes(x = daf))+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    facet_wrap(~treat, nrow = 1, ncol = 2)+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    
    geom_point(aes(y = Hexose.g.gh20_mean), size = 3, shape = 1) +
    geom_errorbar(aes(ymin = Hexose.g.gh20_mean - Hexose.g.gh20_se, 
                      ymax = Hexose.g.gh20_mean + Hexose.g.gh20_se), width = 0.15 *scale.xby) +
    # geom_smooth(method = "lm", formula = y~x, linetype = 2, color = 'black') + #95% of the confidence interval
    # scale_shape_manual(name ="", values=c(1,2)) +
    geom_line(data = data.2, aes(y = berry.sugarConcentration_fruit)) +
    geom_text(data = ann_text,aes(x = x, y = y, label = label.text),hjust=0, size = 5 ) +
    # scale_colour_hue(h=c(200, 360), l=70, c=100)+
    # geom_abline(slope = 1, intercept = 0, col = 'black') +
    # control the shape of the line
    # can suppress the legend display by  guide = "none"
    #scale_linetype_manual(name = "",values=c(1,2,3,4,5,6,1))+
    # errorbar
    #
    #scale_size_manual(name = "",values=c(1.5,1,1,1,1,1,1))+   # size of each line
    # add arrows
    #     geom_segment(data = ann_arrows,aes(x = x, y = y, xend = xend, yend = yend), 
    #                  arrow = arrow(length = unit(0.2, "cm"))) +
  #     #scale_colour_manual("", values=c("red", "blue")) + 
  
  theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    #scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.95, scale.xmax*1.05),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.97, scale.xmax*1.02),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=scale.ymin, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}




##*********************************************************##
## dry weight
##*********************************************************##
plot.berry.dw.pub <- function(data.1, data.2,  # can import many datasets if there are
                          scale.xmin,scale.xmax, scale.xby,
                          scale.ymin,scale.ymax, scale.yby, 
                          xaxis.title,
                          yaxis.title, 
                          label.text,
                          legend=FALSE, title=FALSE) {
  p <- ggplot(data = data.1,aes(x = daf))+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    # facet_wrap(~treat, nrow = 1, ncol = 2)+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    geom_point(aes(y = dw_mean, shape = as.factor(treat)), size = 3) +
    scale_shape_manual(name ="", values=c(1,2)) +
    geom_errorbar(aes(ymin = dw_mean-dw_se, ymax = dw_mean + dw_se ), width = 0.15 *scale.xby) +
    # geom_smooth(method = "lm", formula = y~x, linetype = 2, color = 'black') + #95% of the confidence interval
    geom_line(data = data.2, aes(x = daf, y = dry.weight, linetype = as.factor(treat))) +
    # geom_text(data = ann_text,aes(x = x, y = y, label = label.text),hjust=0, size = 5 ) +
    # scale_colour_hue(h=c(200, 360), l=70, c=100)+
    # geom_abline(slope = 1, intercept = 0, col = 'black') +
    # control the shape of the line
    # can suppress the legend display by  guide = "none"
    scale_linetype_manual(name = "",values=c(1,2))+
    # errorbar
    #
    #scale_size_manual(name = "",values=c(1.5,1,1,1,1,1,1))+   # size of each line
    # add arrows
    #     geom_segment(data = ann_arrows,aes(x = x, y = y, xend = xend, yend = yend), 
    #                  arrow = arrow(length = unit(0.2, "cm"))) +
  #     #scale_colour_manual("", values=c("red", "blue")) + 
  
  theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    #scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.95, scale.xmax*1.05),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.97, scale.xmax*1.02),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=scale.ymin, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}

##*********************************************************##
## fresh weight
##*********************************************************##
plot.berry.fw.pub <- function(data.1, data.2,  # can import many datasets if there are
                          scale.xmin,scale.xmax, scale.xby,
                          scale.ymin,scale.ymax, scale.yby, 
                          xaxis.title,
                          yaxis.title, 
                          label.text,
                          legend=FALSE, title=FALSE) {
  p <- ggplot(data = data.1,aes(x = daf))+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    # facet_wrap(~treat, nrow = 1, ncol = 2)+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    
    geom_point(aes(y = fw_mean, shape = as.factor(treat)), size = 3) +
    geom_errorbar(aes(ymin = fw_mean - fw_se, ymax = fw_mean + fw_se ), width = 0.15 *scale.xby) +
    # geom_smooth(method = "lm", formula = y~x, linetype = 2, color = 'black') + #95% of the confidence interval
    scale_shape_manual(name ="", values=c(1,2)) +
    geom_line(data = data.2, aes(x = daf, y = fresh.weight, linetype = treat)) +
    # geom_text(data = ann_text,aes(x = x, y = y, label = label.text),hjust=0, size = 5 ) +
    # scale_colour_hue(h=c(200, 360), l=70, c=100)+
    # geom_abline(slope = 1, intercept = 0, col = 'black') +
    # control the shape of the line
    # can suppress the legend display by  guide = "none"
    scale_linetype_manual(name = "",values=c(1,2))+
    # errorbar
    #
    #scale_size_manual(name = "",values=c(1.5,1,1,1,1,1,1))+   # size of each line
    # add arrows
    #     geom_segment(data = ann_arrows,aes(x = x, y = y, xend = xend, yend = yend), 
    #                  arrow = arrow(length = unit(0.2, "cm"))) +
  #     #scale_colour_manual("", values=c("red", "blue")) + 
  
  theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    #scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.95, scale.xmax*1.05),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.97, scale.xmax*1.02),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=scale.ymin, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}


##*********************************************************##
## sugar concentration  ########
##*********************************************************##

plot.berry.sugarConcentration.pub <- function(data.1, data.2,  # can import many datasets if there are
                                          scale.xmin,scale.xmax, scale.xby,
                                          scale.ymin,scale.ymax, scale.yby, 
                                          xaxis.title,
                                          yaxis.title, 
                                          label.text,
                                          legend=FALSE, title=FALSE) {
  p <- ggplot(data = data.1,aes(x = daf))+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    # facet_wrap(~treat, nrow = 1, ncol = 2)+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    
    geom_point(aes(y = Hexose.g.gh20_mean, shape = as.factor(treat)), size = 3) +
    geom_errorbar(aes(ymin = Hexose.g.gh20_mean - Hexose.g.gh20_se, 
                      ymax = Hexose.g.gh20_mean + Hexose.g.gh20_se), width = 0.15 *scale.xby) +
    # geom_smooth(method = "lm", formula = y~x, linetype = 2, color = 'black') + #95% of the confidence interval
    scale_shape_manual(name ="", values=c(1,2)) +
    geom_line(data = data.2, aes(y = berry.sugarConcentration_fruit, linetype = treat)) +
    # geom_text(data = ann_text,aes(x = x, y = y, label = label.text),hjust=0, size = 5 ) +
    # scale_colour_hue(h=c(200, 360), l=70, c=100)+
    # geom_abline(slope = 1, intercept = 0, col = 'black') +
    # control the shape of the line
    # can suppress the legend display by  guide = "none"
    scale_linetype_manual(name = "",values=c(1,2))+
    # errorbar
    #
    #scale_size_manual(name = "",values=c(1.5,1,1,1,1,1,1))+   # size of each line
    # add arrows
    #     geom_segment(data = ann_arrows,aes(x = x, y = y, xend = xend, yend = yend), 
    #                  arrow = arrow(length = unit(0.2, "cm"))) +
  #     #scale_colour_manual("", values=c("red", "blue")) + 
  
   theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    #scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.95, scale.xmax*1.05),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.97, scale.xmax*1.02),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=scale.ymin, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}



##*********************************************************##
## carbon.balance  ########
##*********************************************************##
carbon.balance <- function(data.1, # can import many datasets if there are
                          xvar,
                          scale.xmin,scale.xmax, scale.xby,
                          scale.ymin,scale.ymax, scale.yby, 
                          xaxis.title,
                          yaxis.title, 
                          label.text,
                          legend=FALSE, title=FALSE) {
  p <- ggplot() +    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    facet_wrap(~treat, nrow = 1, ncol = 2)+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    geom_point(data = data.1, aes(x = xvar, y = leafLoading), shape = 1, size = 2) +
    geom_point(data = data.1, aes(x = xvar, y = stemLoading), shape = 0, size = 1) +
    
    # geom_point(data = data.1, aes(x = xvar, y = berryUnloading), shape = 16, size = 2) +
    # geom_point(aes(y = stemUnloading), shape = 17, size = 1) +
    # geom_point(aes(y = rootUnloading), shape = 15, size = 1) +
    
    geom_line(data = data.1, aes(x = xvar, y = leafLoading), linetype = 1) +
    geom_line(data = data.1, aes(x = xvar, y = stemLoading), linetype = 2) +
    
    # geom_line(data = data.1, aes(x = xvar, y = berryUnloading),linetype = 3) +
    # geom_line(data = data.1, aes(x = xvar, y = stemUnloading), linetype = 4) +
    # geom_line(data = data.1, aes(x = xvar, y = rootUnloading), linetype = 5) +
    
    geom_rect(data=rectangles, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
              fill='gray80', alpha=0.3) +
    theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.998, scale.xmax*1.001),
    # scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin, scale.xmax),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                      breaks=seq(from=0, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}

carbon.unloading <- function(data.1, # can import many datasets if there are
                           xvar,
                           scale.xmin,scale.xmax, scale.xby,
                           scale.ymin,scale.ymax, scale.yby, 
                           xaxis.title,
                           yaxis.title, 
                           label.text,
                           legend=FALSE, title=FALSE) {
  p <- ggplot(data = data.1) +    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    facet_wrap(~treat, nrow = 1, ncol = 2)+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    # geom_point(data = data.1, aes(x = xvar, y = leafLoading), shape = 1, size = 2) +
    # geom_point(data = data.1, aes(x = xvar, y = stemLoading), shape = 2, size = 1) +
    
    geom_point(data = data.1, aes(x = xvar, y = berryUnloading), shape = 1, size = 2) +
    geom_point(aes(x = xvar, y = stemUnloading), shape = 0, size = 1) +
    geom_point(aes(x = xvar, y = rootUnloading), shape = 2, size = 1) +
    
    # geom_line(data = data.1, aes(x = xvar, y = leafLoading), linetype = 1) +
    # geom_line(data = data.1, aes(x = xvar, y = stemLoading), linetype = 2) +
    
    geom_line(data = data.1, aes(x = xvar, y = berryUnloading),linetype = 1) +
    geom_line(data = data.1, aes(x = xvar, y = stemUnloading), linetype = 2) +
    geom_line(data = data.1, aes(x = xvar, y = rootUnloading), linetype = 3) +
    
    geom_rect(data=rectangles, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
              fill='gray80', alpha=0.3) +
    theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.998, scale.xmax*1.001),
                       # scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin, scale.xmax),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=0, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}


carbon.balance.2 <- function(data.1, # can import many datasets if there are
                           # xvar,
                           scale.xmin,scale.xmax, scale.xby,
                           scale.ymin,scale.ymax, scale.yby, 
                           xaxis.title,
                           yaxis.title, 
                           label.text,
                           legend=FALSE, title=FALSE) {
  p <- ggplot(data = data.1, aes(x = daf)) +    # aes to map the data by specifying what is x and what is y
    # wrap and panels
     facet_wrap(~treat, nrow = 1, ncol = 2)+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    geom_point(data = data.1, aes(y = leafLoading), shape = 1, size = 2) +
    # geom_point(aes(y = stemLoading), shape = 2, size = 1) +
    
    geom_point(data = data.1, aes(y = berryUnloading), shape = 16, size = 2) +
    # geom_point(aes(y = stemUnloading), shape = 17, size = 1) +
    # geom_point(aes(y = rootUnloading), shape = 15, size = 1) +
    
    geom_line(data = data.1, aes(y = leafLoading), linetype = 1) +
    geom_line(data = data.1, aes(y = stemLoading), linetype = 2) +
    
    geom_line(data = data.1, aes(y = berryUnloading),linetype = 3) +
    geom_line(data = data.1, aes(y = stemUnloading), linetype = 4) +
    geom_line(data = data.1, aes(y = rootUnloading), linetype = 5) +
    
    # geom_rect(data=rectangles, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
              # fill='gray80', alpha=0.3) +
    theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.998, scale.xmax*1.001),
                       # scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin, scale.xmax),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=0, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}


##*********************************************************##
## fraction of carbon loadings ########
##*********************************************************##
frac.carbon.loadings <- function(data.1, # can import many datasets if there are
                           # xvar,
                           scale.xmin,scale.xmax, scale.xby,
                           scale.ymin,scale.ymax, scale.yby, 
                           xaxis.title,
                           yaxis.title, 
                           label.text,
                           legend=FALSE, title=FALSE) {
  p <- ggplot(data = data.1,aes(x = daf))+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    facet_wrap(~treat, nrow = 1, ncol = 2)+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    geom_point(aes(y = fraction_leafLoading), shape = 1, size = 2) +
    geom_point(aes(y = fraction_stemLoading), shape = 0, size = 2) +
    
    # geom_point(aes(y = fraction_berryUnloading), shape = 16, size = 2) +
    # geom_point(aes(y = fraction_stemUnloading), shape = 17, size = 2) +
    # geom_point(aes(y = fraction_rootUnloading), shape = 15, size = 2) +
    
    geom_line(aes(y = fraction_leafLoading), linetype = 1) +
    geom_line(aes(y = fraction_stemLoading), linetype = 2) +
    
    # geom_line(aes(y = fraction_berryUnloading),linetype = 3) +
    # geom_line(aes(y = fraction_stemUnloading), linetype = 4) +
    # geom_line(aes(y = fraction_rootUnloading), linetype = 5) +
    
    theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.98, scale.xmax*1.01),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin, scale.xmax),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=0, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}

frac.carbon.unloadings <- function(data.1, # can import many datasets if there are
                                 # xvar,
                                 scale.xmin,scale.xmax, scale.xby,
                                 scale.ymin,scale.ymax, scale.yby, 
                                 xaxis.title,
                                 yaxis.title, 
                                 label.text,
                                 legend=FALSE, title=FALSE) {
  p <- ggplot(data = data.1,aes(x = daf))+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    facet_wrap(~treat, nrow = 1, ncol = 2)+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    # geom_point(aes(y = fraction_leafLoading), shape = 1, size = 2) +
    # geom_point(aes(y = fraction_stemLoading), shape = 2, size = 2) +
    
    geom_point(aes(y = fraction_berryUnloading), shape = 1, size = 2) +
    geom_point(aes(y = fraction_stemUnloading), shape = 0, size = 2) +
    geom_point(aes(y = fraction_rootUnloading), shape = 2, size = 2) +
    
    # geom_line(aes(y = fraction_leafLoading), linetype = 1) +
    # geom_line(aes(y = fraction_stemLoading), linetype = 2) +
    
    geom_line(aes(y = fraction_berryUnloading),linetype = 1) +
    geom_line(aes(y = fraction_stemUnloading), linetype = 2) +
    geom_line(aes(y = fraction_rootUnloading), linetype = 3) +
    
    theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.98, scale.xmax*1.01),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin, scale.xmax),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=0, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}



















carbon.balance.frac <- function(data.1, # can import many datasets if there are
                           scale.xmin,scale.xmax, scale.xby,
                           scale.ymin,scale.ymax, scale.yby, 
                           xaxis.title,
                           yaxis.title, 
                           label.text,
                           legend=FALSE, title=FALSE) {
  p <- ggplot(data = data.1,aes(x = day.hour))+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    facet_wrap(~treat, nrow = 2, ncol = 1, scales = "free_y")+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    geom_point(aes(y = fraction_leafLoading), shape = 1, size = 2) +
    geom_point(aes(y = fraction_stemLoading), shape = 2, size = 2) +
    
    geom_point(aes(y = fraction_berryUnloading), shape = 16, size = 2) +
    geom_point(aes(y = fraction_stemUnloading), shape = 17, size = 2) +
    geom_point(aes(y = fraction_rootUnloading), shape = 15, size = 2) +
    
    theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    #scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.95, scale.xmax*1.05),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin, scale.xmax),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    # scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
    # breaks=seq(from=0, to=scale.ymax, by=scale.yby)) +
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}

##*********************************************************##
## carbon.balance  ########
##*********************************************************##
plot.water.balance <- function(data.1, # can import many datasets if there are
                          xvar, yvar,
                           scale.xmin,scale.xmax, scale.xby,
                           scale.ymin,scale.ymax, scale.yby, 
                           xaxis.title,
                           yaxis.title, 
                           label.text,
                           legend=FALSE, title=FALSE) {
  p <- ggplot()+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    # facet_wrap(~treat, nrow = 2, ncol = 1, scales = "free_y")+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    geom_point(data = data.1, aes(x = xvar, y=yvar,shape = as.factor(treat)), size = 1)+
    # geom_line(data = data.1, aes(x = xvar, y=yvar,linetype = as.factor(treat)))+
    scale_shape_manual(name ="", values=c(1,2)) +
    # scale_linetype_manual(name ="", values=c(1,2)) +
    geom_rect(data=rectangles, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
              fill='gray80', alpha=0.3) +
    
    theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    #scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.95, scale.xmax*1.05),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin, scale.xmax),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=scale.ymin, to=scale.ymax, by=scale.yby)) +

    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}

##*********************************************************##
## berry properties  ########
##*********************************************************##
plot.berry.property <- function(data.1, # can import many datasets if there are
                               xvar, yvar,
                               scale.xmin,scale.xmax, scale.xby,
                               scale.ymin,scale.ymax, scale.yby, 
                               xaxis.title,
                               yaxis.title, 
                               label.text,
                               legend=FALSE, title=FALSE) {
  p <- ggplot()+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    # facet_wrap(~treat, nrow = 2, ncol = 1, scales = "free_y")+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    # geom_text(data = data_frame(x_var = 0, y_var = 0),
    #           aes(x = scale.xmin + 0.05*(scale.xmax -scale.xmin),
    #               y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
    #           hjust=0,  size = 4) +
    geom_line(data = data.1, aes(x = xvar, y=yvar, linetype = as.factor(treat)))+
    scale_linetype_manual(name ="", values=c(1,2)) +
    geom_rect(data=rectangles, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
              fill='gray80', alpha=0.3) +
    
    theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    #scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.95, scale.xmax*1.05),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin, scale.xmax),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin*0.98, scale.ymax*1.02),
                       breaks=seq(from=scale.ymin, to=scale.ymax, by=scale.yby)) +
    
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}



##*********************************************************##
## berry properties long term########
##*********************************************************##
plot.berry.property.long <- function(data, # can import many datasets if there are
                                xvar, yvar,
                                scale.xmin,scale.xmax, scale.xby,
                                scale.ymin,scale.ymax, scale.yby, 
                                xaxis.title,
                                yaxis.title, 
                                label.text,
                                legend=FALSE, title=FALSE) {
  p <- ggplot()+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    # facet_wrap(~treat, nrow = 2, ncol = 1, scales = "free_y")+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    geom_text(data = data_frame(x_var = 0, y_var = 0),
              aes(x = scale.xmin + 0.03*(scale.xmax -scale.xmin),
                  y = scale.ymin + 0.9*(scale.ymax -scale.ymin),  label = label.text),
              hjust=0,  size = 5) +
    geom_line(data = data, aes(x = xvar, y=yvar))+
    # geom_rect(data=rectangles, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
    #           fill='gray80', alpha=0.3) +
    # 
    theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    #scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.95, scale.xmax*1.05),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin, scale.xmax),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin, scale.ymax),
                       breaks=seq(from=scale.ymin, to=scale.ymax, by=scale.yby)) +
    
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}





plot.radiation <- function(data.1, # can import many datasets if there are
                           xvar, yvar,
                           scale.xmin,scale.xmax, scale.xby,
                           scale.ymin,scale.ymax, scale.yby, 
                           xaxis.title,
                           yaxis.title, 
                           label.text,
                           legend=FALSE, title=FALSE) {
  p <- ggplot()+    # aes to map the data by specifying what is x and what is y
    # wrap and panels
    # facet_wrap(~treat, nrow = 2, ncol = 1, scales = "free_y")+ # , scales = "free"
    # ifelse(scale.ymax >0,0.9*scale.ymax, 1.05*scale.ymax ),
    geom_text(data = data_frame(x_var = 0, y_var = 0),
              aes(x = scale.xmin + 0.03*(scale.xmax -scale.xmin),
                  y = scale.ymin + 0.95*(scale.ymax -scale.ymin),  label = label.text),
              hjust=0,  size = 5) +
    geom_line(data = data.1, aes(x = xvar, y=yvar))+
    # geom_point(data = data.2, aes(x = xvar, y=yvar), shape = 1, colour = 'red')+
    # scale_linetype_manual(name ="", values=c(1,2)) +
    # geom_rect(data=rectangles, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), 
    #           fill='gray80', alpha=0.3) +
    # 
    theme(panel.spacing = unit(0.5, "lines")) +      # control the distance between different panels
    
    # control the axis
    #      # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    # scale_y_continuous(yaxis.title)+                  # control the x, y lab and ticks
    #scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin*0.95, scale.xmax*1.05),
    scale_x_continuous(xaxis.title, expand=c(0,0), limits=c(scale.xmin, scale.xmax),
                       breaks=seq(from = scale.xmin, to=scale.xmax, by=scale.xby)) +
    scale_y_continuous(yaxis.title, expand=c(0,0), limits=c(scale.ymin*0.98, scale.ymax*1.02),
                       breaks=seq(from=scale.ymin, to=scale.ymax, by=scale.yby)) +
    
    
    # theme of the layout
    lib.opts.layout +
    panel_border(colour = "black", size = 0.5,
                 linetype = 1,remove = FALSE)
  
  if (legend) {
    p+lib.opts.legend 
  } 
  else {
    p+lib.opts.nolegend
  }
  # if (title) p+opts(title=title)
}


