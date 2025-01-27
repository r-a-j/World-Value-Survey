    
    
    library(tidyverse)
    library(MASS)  # For ordinal regression
    library(ordinal)  # For ordinal logistic models
    
    # # Load the compiled fixations data
    # world_survey <- read.csv("/Users/stephenampah/Documents/Case_Study/Case_Study_2/Raj_Data/g8_filtered_data.csv")
    # 
    # # Display the column names of the Raw_data data frame
    # names(world_survey)
    # 
    # # Load the participant information data
    # Wave_7 <- read.csv("/Users/stephenampah/Documents/Case_Study/Case_Study_2/Data/WVS_Cross-National_Wave_7_csv_v6_0.csv")
    # # Display the names of the columns in the participant info dataset
    # names(Wave_7)
    # 
    # 
    # # Load the compiled fixations data
    # time_series <- read.csv("/Users/stephenampah/Documents/Case_Study/Case_Study_2/Raj_Data/filtered_time_series_1981_2022.csv")
    # 
    # # Display the column names of the Raw_data data frame
    # names(time_series)
    # 
    # 
    # # Load the compiled fixations data
    # full_time_series <- read.csv("/Users/stephenampah/Documents/Case_Study/Case_Study_2/Data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
    # names(full_time_series)
    # 
    # filtered_column <- full_time_series %>% pull(S020) 
    
    # Load the data
    time_series_2 <- read.csv("/Users/stephenampah/Documents/Case_Study/Case_Study_2/Data/filtered_time_series_1981_2022.csv")
    
    # Display the column names of the Raw_data data frame
    names(time_series_2)
    
    str(time_series_2)
    
    # List of variables of interest to convert to ordered factors
    variables_of_interest <- c(
      "C.Armed.Forces", "C.Police", "C.Parliament",
      "C.Civil.Services", "C.Television", "C.Government",
      "C.Political.Parties", "C.Courts"
    )
    
    # Convert the variables to ordered factors
    for (var in variables_of_interest) {
      time_series_2[[var]] <- factor(time_series_2[[var]], ordered = TRUE)
    }
    
    # Check the structure of the dataset to confirm changes
    str(time_series_2)
    
    freq_data <- time_series_2 %>%
      group_by(Survey.year, C.Armed.Forces) %>%
      summarize(count = n(), .groups = "drop") %>%
      mutate(proportion = count / sum(count))
    print(freq_data)
    
    
    # Fit the CLMM
    clmm_model <- clmm(
      C.Armed.Forces ~ Survey.year + (1 | Country),
      data = time_series_2,
      link = "logit"
    )
    
    # Summarize the model results
    summary(clmm_model)
    
    
    
    