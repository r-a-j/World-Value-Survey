    
    # Load necessary libraries
    library(janitor)
    library(tidyverse)
    library(ordinal)
    
    # Step 1: Load the dataset
    raw_data <- read.csv("data/preprocessed/filtered_time_series_1981_2022.csv")
    
    # Simplify column names
    raw_data <- raw_data %>% clean_names()
    
    # Display the column names of the Raw_data data frame
    colnames(raw_data)
    
    # Step 2: Explore the structure of the dataset
    str(raw_data)
    summary(raw_data)
    
    # Step 3: Filter the dataset for US and clean NA values
    model_data <- raw_data %>%
      clean_names() %>%
      filter(country_name == "USA") %>%
      dplyr::select(survey_year, c_parliament, c_government, c_political_parties, c_armed_forces, c_courts, 
             age, highest_educational_level, scale_of_incomes) %>%
      mutate(across(starts_with("c_"), ~ na_if(., -4))) %>%
      drop_na()
    
    # Step 4: Transform the dataset to long format
    long_data <- model_data %>%
      pivot_longer(
        cols = starts_with("c_"),
        names_to = "institution",
        values_to = "trust_level"
      ) %>%
      mutate(
        institution = factor(institution, 
                             levels = c("c_parliament", "c_government", "c_political_parties", "c_armed_forces", "c_courts"),
                             labels = c("Parliament", "Government", "Political Parties", "Armed Forces", "Courts")),
        trust_level = as.factor(trust_level),
        highest_educational_level = as.factor(highest_educational_level)
      )
    
    # Step 5: Fit the Cumulative Link Mixed Model (CLMM)
    clmm_model <- clmm(
      trust_level ~ institution + age + highest_educational_level + scale_of_incomes + (1 | survey_year),
      data = long_data
    )
    
    # Step 6: Summary of the model
    summary(clmm_model)
    
    # Step 7: Extract random effects to interpret year-level variability
    ranef(clmm_model)
    
    # Step 8: Generate predicted probabilities and plots
    predicted_data <- long_data %>%
      mutate(Predicted_Trust = as.numeric(fitted(clmm_model)))
    
    aggregated_predictions <- predicted_data %>%
      group_by(Year = survey_year, Institution = institution) %>%
      summarize(Mean_Trust = mean(Predicted_Trust, na.rm = TRUE), .groups = "drop")
    
    ggplot(aggregated_predictions, aes(x = Year, y = Mean_Trust, color = Institution, group = Institution)) +
      geom_line(size = 1) +
      geom_point(size = 3) +
      labs(
        x = "Year",
        y = "Predicted Trust Level",
        color = NULL
      ) +
      theme_minimal() +
      theme(
        text = element_text(size = 14),
        legend.position = "bottom"
      )
    
    ggplot(aggregated_predictions, aes(x = as.factor(Year), y = Mean_Trust, fill = Institution)) +
      geom_bar(stat = "identity", position = "dodge", width = 0.7) +
      labs(
        x = "Year",
        y = "Predicted Trust Level",
        fill = NULL
      ) +
      theme_minimal() +
      theme(
        text = element_text(size = 14),
        legend.position = "bottom"
      )
    
    
