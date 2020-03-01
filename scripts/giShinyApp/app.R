library(shiny)
library(tidyverse)
library(ggplot2)
library(gridExtra)
theme_set(theme_bw())

# Define the shiny UI
ui = fluidPage(
  titlePanel('Child Anxiety Risk Assessment Tool'),
  sidebarLayout(position = 'left', sidebarPanel(
    selectInput("age", label = "Age",
                choices = 5:18),
    selectInput("sex", label = "Sex",
                choices = c('Male', 'Female')),
    selectInput("nausea", label = 'Frequency of Nausea', choices= c('Never', 'Sometimes', 'Often')),
    selectInput("stomachaches", label = 'Frequency of Stomachaches', choices= c('Never', 'Sometimes', 'Often')),
    selectInput("vomit", label = 'Frequency of Vomiting', choices= c('Never', 'Sometimes', 'Often')),
    selectInput("constipation", label = 'Frequency of Constipation', choices= c('Never', 'Sometimes', 'Often'))),
  mainPanel(
    tabsetPanel(
      tabPanel('Risk', plotOutput(outputId = 'plot')),
      tabPanel('Resources', h5('Coming Soon!')),
      tabPanel('Research', h5('Coming Soon!'))
    )
  )))



# Server functions defines the plot to be shown
server = shinyServer(function(input, output){
  load('data/shinyPreds.rda')
  output$plot = renderPlot(height = 500, {
      
      
      childAge = input$age
      childSex = ifelse(input$sex == 'Male', -.5,.5)
      childNausea = case_when(input$nausea == 'Never' ~ 0, input$nausea == 'Sometimes' ~ 1, input$nausea == 'Often' ~ 2)
      childStomachaches = case_when(input$stomachaches == 'Never' ~ 0, input$stomachaches== 'Sometimes' ~ 1, input$stomachaches== 'Often' ~ 2)
      childVomiting = case_when(input$vomit == 'Never' ~ 0, input$vomit == 'Sometimes' ~ 1, input$vomit == 'Often' ~ 2)
      childConstipation = case_when(input$constipation == 'Never' ~ 0, input$constipation == 'Sometimes' ~ 1, input$constipation == 'Often' ~ 2)
      
      childGiTotal = childNausea + childStomachaches + childVomiting + childConstipation

      plotData = filter(shinyPreds, sexContrast == childSex, ageBin == childAge)
      enteredData = filter(shinyPreds, sexContrast == childSex, ageBin ==childAge, cbclGISum == childGiTotal)
      
      ggplot(plotData, aes(x=cbclGISum, y = X50.)) +
        geom_ribbon(aes(ymin = X2.5., ymax = X97.5., fill = '95%'), alpha = .2) +
        geom_ribbon(aes(ymin = X10., ymax = X90., fill = '80%'), alpha = .3) +
        geom_point() +
        geom_line(lwd = 2) +
        geom_point(data = enteredData, size = 7, color = 'blue') +
        geom_segment(aes(x=0,xend=enteredData$cbclGISum[1],
                         y=enteredData$X50.[1], yend = enteredData$X50.[1]), 
                     lty = 2, color = 'blue') +
        geom_segment(aes(y=0,yend=enteredData$X50.[1], 
                         x=enteredData$cbclGISum[1], xend = enteredData$cbclGISum[1]), 
                     lty = 2, color = 'blue') +
        ylim(0,1) +
        scale_y_continuous(limits=c(0,1), expand = c(0,0)) +
        scale_x_continuous(limits = c(0,8), expand = c(0,0)) +
        geom_text(data = enteredData, vjust = -4, hjust = -.1, check_overlap = TRUE, label = 'Your child', size = 5, face = 'bold', color= 'blue') +
        labs(x = 'Total Number of GI Symptoms Experienced', y = 'Proportion of Children Of Same Age/Sex\n Diagnosed with Anx') +
        theme(text = element_text(size = 20, face = 'bold'), 
              panel.grid.minor = element_blank(), panel.grid.major.x = element_blank()) +
        scale_fill_grey(name = 'Uncertainty\n Level') 
    }
  )
  output$placeholder <- renderText({'Sample Text'})
})
shinyApp(ui = ui, server = server, options=list(height=700))