library(shiny)
library(caret)
library(randomForest) 
library(e1071)       
library(kernlab)      
library(rpart)        
library(klaR)         
library(MASS)        
library(ggplot2)
library(dplyr)
library(tidyr)

# --- 1. LOAD MODELS DIRECTLY ---
# We load them one by one to avoid syntax errors during publishing
# Ensure the "models" folder is in the same directory as this app.R file
logit_model <- readRDS("models/logit_model.rds")
svm_model   <- readRDS("models/svm_model.rds")
tree_model  <- readRDS("models/tree_model.rds")
rf_model    <- readRDS("models/rf_model.rds")
nb_model    <- readRDS("models/nb_model.rds")

# Combine them into the list the app expects
models <- list(
  LogisticRegression = logit_model,
  SVM   = svm_model,
  Tree  = tree_model,
  RF    = rf_model,
  NB    = nb_model
)

# --- 2. USER INTERFACE (UI) ---
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      .box { border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
      .high-risk { color: #d9534f; font-weight: bold; }
      .low-risk { color: #5cb85c; font-weight: bold; }
      #predict_btn { width: 100%; font-size: 18px; font-weight: bold; margin-top: 20px; }
    "))
  ),
  
  titlePanel("Diabetes Risk Prediction: Multi-Model Consensus"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Patient Health Profile"),
      helpText("Adjust the values below to match the patient's data."),
      
      tabsetPanel(
        tabPanel("Demographics",
                 br(),
                 sliderInput("Age", "Age Category (1=18-24 ... 13=80+):", 1, 13, 8),
                 selectInput("Sex", "Sex:", choices = list("Female" = 0, "Male" = 1), selected = 0),
                 numericInput("BMI", "BMI:", 28, min = 10, max = 100),
                 selectInput("HighBP", "High Blood Pressure?", choices = list("No" = 0, "Yes" = 1), selected = 1),
                 selectInput("HighChol", "High Cholesterol?", choices = list("No" = 0, "Yes" = 1), selected = 1)
        ),
        tabPanel("Lifestyle",
                 br(),
                 selectInput("Smoker", "Smoker (Lifetime >100 cigs)?", choices = list("No" = 0, "Yes" = 1), selected = 0),
                 selectInput("PhysActivity", "Physical Activity (Past 30d)?", choices = list("No" = 0, "Yes" = 1), selected = 0),
                 selectInput("Fruits", "Daily Fruit Consumption?", choices = list("No" = 0, "Yes" = 1), selected = 1),
                 selectInput("Veggies", "Daily Veggie Consumption?", choices = list("No" = 0, "Yes" = 1), selected = 1),
                 selectInput("HvyAlcoholConsump", "Heavy Alcohol Consumption?", choices = list("No" = 0, "Yes" = 1), selected = 0)
        ),
        tabPanel("History & Health",
                 br(),
                 sliderInput("GenHlth", "General Health (1=Excellent, 5=Poor):", 1, 5, 3),
                 sliderInput("MentHlth", "Days Poor Mental Health (0-30):", 0, 30, 2),
                 sliderInput("PhysHlth", "Days Poor Physical Health (0-30):", 0, 30, 5),
                 selectInput("CholCheck", "Cholesterol Check (5y)?", choices = list("No" = 0, "Yes" = 1), selected = 1),
                 selectInput("Stroke", "History of Stroke?", choices = list("No" = 0, "Yes" = 1), selected = 0),
                 selectInput("HeartDiseaseorAttack", "Heart Disease/Attack?", choices = list("No" = 0, "Yes" = 1), selected = 0),
                 selectInput("DiffWalk", "Difficulty Walking?", choices = list("No" = 0, "Yes" = 1), selected = 0)
        )
      ),
      
      actionButton("predict_btn", "Run Analysis", class = "btn-primary")
    ),
    
    mainPanel(
      fluidRow(
        column(12, 
               div(class = "box", style = "background-color: #f9f9f9;",
                   h3(textOutput("consensus_text"), align = "center"),
                   h4(textOutput("consensus_subtext"), align = "center")
               )
        )
      ),
      fluidRow(
        column(7, 
               div(class = "box",
                   h4("Model Probability Comparison"),
                   plotOutput("prob_plot", height = "300px")
               )
        ),
        column(5,
               div(class = "box",
                   h4("Detailed Predictions"),
                   tableOutput("detail_table")
               )
        )
      ),
      fluidRow(
        column(12,
               div(class = "box",
                   h4("Interpretation Guide"),
                   p("This tool aggregates predictions from 5 different machine learning algorithms trained on BRFSS 2015 data."),
                   tags$ul(
                     tags$li(strong("Random Forest & SVM:"), " Generally the most accurate models (AUC > 0.80)."),
                     tags$li(strong("Naive Bayes:"), " Highly sensitive (good at screening) but may overpredict risk."),
                     tags$li(strong("Consensus Score:"), " If >3 models predict 'High Risk', high chance of Diabetes")
                   )
               )
        )
      )
    )
  )
)

# --- 3. SERVER LOGIC ---
server <- function(input, output) {
  
  observeEvent(input$predict_btn, {
    
    # Prepare Input Data (Must be numeric/integer to match training data)
    new_data <- data.frame(
      Age = as.numeric(input$Age),
      Sex = as.numeric(input$Sex),
      HighChol = as.numeric(input$HighChol),
      CholCheck = as.numeric(input$CholCheck),
      BMI = as.numeric(input$BMI),
      Smoker = as.numeric(input$Smoker),
      HeartDiseaseorAttack = as.numeric(input$HeartDiseaseorAttack),
      PhysActivity = as.numeric(input$PhysActivity),
      Fruits = as.numeric(input$Fruits),
      Veggies = as.numeric(input$Veggies),
      HvyAlcoholConsump = as.numeric(input$HvyAlcoholConsump),
      GenHlth = as.numeric(input$GenHlth),
      MentHlth = as.numeric(input$MentHlth),
      PhysHlth = as.numeric(input$PhysHlth),
      DiffWalk = as.numeric(input$DiffWalk),
      Stroke = as.numeric(input$Stroke),
      HighBP = as.numeric(input$HighBP)
    )
    
    results <- data.frame(Model = character(), Probability = numeric(), Prediction = character(), stringsAsFactors = FALSE)
    model_names <- names(models)
    
    for(m in model_names) {
      prob <- tryCatch({
        pred <- predict(models[[m]], new_data, type = "prob")
        pred[, "Yes"] * 100
      }, error = function(e) { NA })
      
      cls <- tryCatch({
        as.character(predict(models[[m]], new_data))
      }, error = function(e) { "Error" })
      
      results <- rbind(results, data.frame(Model = m, Probability = prob, Prediction = cls))
    }
    
    n_high_risk <- sum(results$Prediction == "Yes")
    avg_prob <- mean(results$Probability, na.rm = TRUE)
    
    output$consensus_text <- renderText({
      paste0("Consensus: ", n_high_risk, " out of 5 Models Predict High Risk")
    })
    
    output$consensus_subtext <- renderText({
      risk_label <- ifelse(avg_prob > 50, "HIGH RISK", "Low Risk")
      paste0("Average Probability: ", round(avg_prob, 1), "% (", risk_label, ")")
    })
    
    output$detail_table <- renderTable({
      results %>%
        mutate(Probability = paste0(round(Probability, 1), "%")) %>%
        dplyr::select(Model, Prediction, Probability) # <--- FIXED: Added 'dplyr::'
    }, striped = TRUE, hover = TRUE)
    
    output$prob_plot <- renderPlot({
      results$Color <- ifelse(results$Probability > 50, "#d9534f", "#5cb85c")
      ggplot(results, aes(x = reorder(Model, -Probability), y = Probability, fill = Color)) +
        geom_bar(stat = "identity") +
        scale_fill_identity() +
        geom_hline(yintercept = 50, linetype = "dashed", color = "black") +
        ylim(0, 100) +
        labs(x = "Model", y = "Probability of Diabetes (%)", 
             title = "Risk Probability by Algorithm") +
        theme_minimal() +
        theme(text = element_text(size = 14))
    })
    
  })
}

shinyApp(ui, server)