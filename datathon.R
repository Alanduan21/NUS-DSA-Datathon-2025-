# ==========================================================
# Load and Prepare Data
# ==========================================================

data <- read.csv("~/downloads/medical insurance.csv")

str(data)

# Encode categorical variables as factors
data$sex <- as.factor(data$sex)
data$smoker <- as.factor(data$smoker)
data$region <- as.factor(data$region)
# ==========================================================
# Medical Insurance Charge Prediction using Regression Models
# ==========================================================
# ==========================================================
# Install & Load Required Packages
# ==========================================================
packages <- c("tidyverse", "caret", "glmnet", "ggplot2", "car", "corrplot")

installed_packages <- rownames(installed.packages())
for (pkg in packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}

# Load libraries
library(tidyverse)
library(caret)
library(glmnet)
library(ggplot2)
library(car)
library(corrplot)

# Fit the model
model <- lm(charges ~ age + bmi + children + smoker + region + sex, data = data)

# VIF: Variance Inflation Factor
vif_df <- as.data.frame(vif(model)) %>%
  rownames_to_column("Predictor") %>%
  rename(VIF = 2) %>%
  mutate(
    Collinearity_Level = case_when(
      VIF < 5 ~ "Low",
      VIF >= 5 & VIF < 10 ~ "Moderate",
      VIF >= 10 ~ "High",
      TRUE ~ "Unknown"
    )
  )

print(vif_df)

# Correlation Matrix (for numeric predictors only)
numeric_vars <- data %>%
  select(where(is.numeric))

cor_matrix <- cor(numeric_vars, use = "complete.obs")

# Visualize correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8, addCoef.col = "black")

# ANOVA: F-values for each predictor
anova_df <- anova(model) %>%
  as.data.frame() %>%
  rownames_to_column("Predictor") %>%
  filter(Predictor %in% vif_df$Predictor) %>%
  select(Predictor, `F value`)

# Combine VIF and ANOVA
collinearity_summary <- left_join(vif_df, anova_df, by = "Predictor")

print(collinearity_summary)

# ==========================================================
# Split Data into Train/Test Sets
# ==========================================================
set.seed(123)
trainIndex <- createDataPartition(data$charges, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Define cross-validation
ctrl <- trainControl(method = "cv", number = 10)

# ==========================================================
# Multiple Linear Regression
# ==========================================================
multi_linear_model <- train(charges ~ age + bmi  + smoker,
                            data = trainData,
                            method = "lm",
                            trControl = ctrl)

summary(multi_linear_model$finalModel)

# ==========================================================
# Ridge Regression
# ==========================================================
# Ridge requires numeric predictors, so we create dummy variables
x <- model.matrix(charges ~ ., data = trainData)[, -1]
y <- trainData$charges

ridge_model <- train(
  x = x, y = y,
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = expand.grid(alpha = 0, lambda = seq(0.001, 1, length = 20))
)

plot(ridge_model)
ridge_model$bestTune

# ==========================================================
# Model Evaluation
# ==========================================================

# Predict on test data
pred_multi <- predict(multi_linear_model, newdata = testData)
pred_ridge <- predict(ridge_model, newdata = model.matrix(charges ~ ., data = testData)[, -1])

# Continuous metrics
rmse <- function(pred, actual) sqrt(mean((pred - actual)^2))
r2 <- function(pred, actual) cor(pred, actual)^2

data.frame(
  Model = c("Multiple Linear", "Ridge"),
  RMSE = c(rmse(pred_multi, testData$charges),
           rmse(pred_ridge, testData$charges)),
  R2 = c(r2(pred_multi, testData$charges),
         r2(pred_ridge, testData$charges))
) %>%
  arrange(RMSE)

# ==========================================================
# Predicted vs Actual visualisation
# ==========================================================
pred_df <- data.frame(
  Actual = testData$charges,
  Predicted = pred_multi
)

ggplot(pred_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6) +
  geom_abline(color = "red", linetype = "dashed") +
  labs(title = "Predicted vs Actual Charges (Multiple Linear Regression)",
       x = "Actual Charges", y = "Predicted Charges") +
  theme_minimal()

# ==========================================================
# Ridge Regression Predicted vs Actual Plot
# ==========================================================

library(glmnet)
library(ggplot2)

# Assuming your ridge model is called 'ridge_model' and you have:
# x_train, y_train, x_test, y_test (or use model.matrix like before)

# If you used caret for ridge_model:
pred_ridge <- predict(ridge_model, newdata = model.matrix(charges ~ ., data = testData)[, -1])

# Combine into a dataframe
ridge_pred_df <- data.frame(
  Actual = testData$charges,
  Predicted = as.numeric(pred_ridge)
)

# Plot
ggplot(ridge_pred_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "black") +
  geom_abline(color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Charges (Ridge Regression)",
    x = "Actual Charges",
    y = "Predicted Charges"
  ) +
  theme_minimal()
