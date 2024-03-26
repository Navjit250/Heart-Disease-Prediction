library(dplyr)
library(tidyr)


#---------------------------Uploading File into R------------------------------------

cvd.df <- read.csv(file.choose())
cvd.df

#--------------------------------------- Viewing Dataset-----------------------------

# Displaying Columns Names
col_names<-names(cvd.df)
col_names

#-------------------------------- Data Preprocessing---------------------------------

cvd.df <- as.data.frame(cvd.df)
cvd.df
# Drop rows with "18-24" ,"25-29" ,"30-34","35-39" in  Age_Category column
data_filtered <-  cvd.df%>%
  filter(!(Age_Category %in% c("18-24", "25-29","30-34","35-39")))
cvd.df<-data_filtered
cvd.df

# Create a new binary column for ages above 60,where if >60  is 1 and o if otherwise
cvd.df$Age_Above_60 <- ifelse(grepl("\\+", cvd.df$Age_Category) | as.integer(gsub("-.*", "", cvd.df$Age_Category)) > 60, 1, 0)

# Dropping NAs 
cvd_cleaned<-drop_na(cvd.df)

# Dropping 'Age_Category ' using dplyr
cvd_cleaned <- cvd_cleaned%>%
  select(-"Age_Category")


#-----------------------------Data Exploration ---------------------------------------------------


# viewing Column Names after removing Age_Category
col_names<-names(cvd_cleaned)
col_names
cvd_cleaned

#removing  "No, pre-diabetes or borderline diabetes &Yes, but female told only during pregnancy" from Diabetes Column
cvd_cleaned <- cvd_cleaned %>%
  filter(Diabetes != 'No, pre-diabetes or borderline diabetes' & 
           Diabetes != 'Yes, but female told only during pregnancy')

# viewing Diabetes column
cvd_cleaned$Diabetes
cvd_cleaned

# Converting the below columns into Binary
columns_to_convert <- c("Exercise", "Heart_Disease", "Skin_Cancer","Other_Cancer","Arthritis","Depression","Diabetes","Smoking_History")

# Convert 'Yes' to 1 and everything else to 0 in specified columns using mutate_at()
cvd_cleaned <- cvd_cleaned %>%
  mutate_at(vars(columns_to_convert), ~ ifelse(. == 'Yes', 1, 0))

#Convering "female" to 1 and Males to 0 in Sex column
cvd_cleaned$Sex <- ifelse(cvd_cleaned$Sex == "Female", 1, 0)

# remove 
cvd_cleaned <- cvd_cleaned %>%
  filter(Checkup != "Within the past 5 years" &
           Checkup != "Within the past 2 years" &
           Checkup != "5 or more years ago")

cvd_cleaned
cvd_cleaned$Checkup <- ifelse(cvd_cleaned$Checkup == "Within the past year", 1, 0)
cvd_cleaned

# Recreate the 'Group1' column based on 'General_Health' values
cvd_cleaned <- cvd_cleaned %>%
  mutate(Group1 = case_when(
    General_Health %in% c("Very Good", "Good", "Excellent") ~ "High",
    General_Health %in% c("Poor", "Fair") ~ "Low",
    TRUE ~ NA_character_
  )) %>%
  select(-General_Health) %>%  # Drop 'General_Health' column
  rename(General_Health = Group1) %>%  # Rename 'Group1' to 'General_Health'
  mutate(General_Health = ifelse(General_Health == "High", 1, 0))  # Assign 1 to "High" and 0 to everything else




# Removing columns using dplyr select function( optional line)
#cvd_cleaned <- select(cvd_cleaned, -c(Other_Cancer, Skin_Cancer, Height_.cm.,Fruit_Consumption,Green_Vegetables_Consumption,Checkup))



cvd_cleaned <- head(cvd_cleaned, 20000)
cvd.df <- head(cvd.df, 20000)
data_filtered <-head(data_filtered, 20000)

cvd.df <- cvd_cleaned
cvd.df



library(caret)
library(jtools)
install.packages("jtools")
install.packages("caret")
library(ipred)
install.packages("ipred")
library(caret)
library(ggplot2)
library(pROC)

#============================ logistic regression ==============================
# lm.fit <- ...: -> creates a new variable named lm.fit to store the results of the logistic regression model.
# glm(): -> run Generalized Linear Model to fit varius regression models, including logistic regression.
### -> Modeling the probability of heart disease as a function of several predictor variables
#  family = "binomial"-> indicates that you are fitting a logistic regression model, which is suitable for binary outcomes.
# Add a new column to the dataset for the composite dietary score
train.index <- sample(c(1:dim(cvd.df)[1]), dim(cvd.df)[1]*0.8)  
train.df <- cvd.df[train.index, ]
valid.df <- cvd.df[-train.index, ]

lm.fit <- glm(Heart_Disease ~ Checkup + Exercise + Skin_Cancer + Other_Cancer + Depression + Diabetes + Arthritis + Sex + BMI + Smoking_History +
                Alcohol_Consumption + Fruit_Consumption + Green_Vegetables_Consumption + FriedPotato_Consumption + Age_Above_60 + General_Health, data = train.df, 
              family = "binomial")
# remove scientific notation
options(scipen=999) 
summary(lm.fit, digits=5)

pred <- predict(lm.fit, valid.df)
valid.df$predicted_prob <- predict(lm.fit, valid.df, type = "response")


confusionMatrix(factor(ifelse(pred > 0.5, "1", "0")), as.factor(valid.df$Heart_Disease))



# Assuming lm.fit is your logistic regression model
coef_bmi <- coef(lm.fit)["BMI"]
intercept <- coef(lm.fit)["(Intercept)"]
bmi_values <- seq(min(valid.df$BMI, na.rm = TRUE), max(valid.df$BMI, na.rm = TRUE), length.out = 100)
predicted_probs <- 1 / (1 + exp(-(intercept + coef_bmi * bmi_values)))

ggplot(data.frame(BMI = bmi_values, Probability = predicted_probs), aes(x = BMI, y = Probability)) +
  geom_line(color = "blue") +
  labs(title = "Logistic Regression Curve: Probability of Heart Disease by BMI",
       x = "BMI", y = "Predicted Probability of Heart Disease") +
  theme_minimal()



# ===============================KNN Model======================================

library(class)
install.packages("caret")
library(caret)


# Partition data into 60% training set and 40% testing set
set.seed(1)  
train.index <- sample(c(1:dim(cvd_cleaned)[1]), dim(cvd_cleaned)[1]*0.6)  
train.df <- cvd_cleaned[train.index, ]
valid.df <- cvd_cleaned[-train.index, ]
  


# Create training and validation dataframes with the variable Heart_Disease as the target variable
train.cvd <- as.factor(train.df$Heart_Disease)
valid.cvd <- as.factor(valid.df$Heart_Disease)

# Run model with training and testing data; Create confusion matrix
nn3 <- knn(train.df, valid.df, cl=train.cvd, k=5)
confusionMatrix(as.factor(nn3), as.factor(valid.cvd))

# Find optimal K (from 1 to 14)
accuracy.df <- data.frame(k = seq(1, 15, 1), accuracy = 0)
View(accuracy.df)
for(i in 1:15) {
  knn.pred <- knn(train.df, valid.df, cl=train.cvd, k=i)
  accuracy.df[i, 'accuracy'] <- confusionMatrix(knn.pred, valid.cvd)$overall[1] 
}
accuracy.df





# ===============================Neural Network=================================


library(neuralnet)
library(NeuralNetTools)
library(nnet)

set.seed(1)

# split the data into training and validation sets
train.index <- sample(c(1:dim(cvd_cleaned)[1]), dim(cvd_cleaned)[1]*0.6)  
train.df <- cvd_cleaned[train.index, ]
valid.df <- cvd_cleaned[-train.index, ]

# create the neural network model
nn <- neuralnet(Heart_Disease ~ General_Health+Checkup+Exercise
                +Skin_Cancer+Other_Cancer+Depression+Diabetes+Arthritis
                +Age_Above_60+BMI+Smoking_History+Alcohol_Consumption,
                data = train.df, 
                hidden = 4,
                linear.output = TRUE)
print(nn)

# neural network results
plot(nn)
neuralweights(nn)

# neural network performance
nn.pred <- compute(nn, valid.df)
nn.class <- ifelse(nn.pred$net.result > 0.5, 1, 0)
nn.class

# Evaluate the model
confusionMatrix(as.factor(nn.class), as.factor(valid.df$Heart_Disease))





