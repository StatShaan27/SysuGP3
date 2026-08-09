# Environment Setup & Data Loading
library(data.table)
library(tidyverse)
library(rsample)
library(recipes)
library(glmnet)
library(arm)
library(ranger)
library(xgboost)
library(car)
library(progress)
library(beepr)
library(tableone)


# Create data folder if missing
if(!dir.exists("data")) dir.create("data")


# Read raw BRFSS data
raw <- fread("diabetes_binary_health_indicators_BRFSS2015.csv") %>% as_tibble()
cat("Diabetes prevalence:", mean(raw$Diabetes_binary), "\n")


# Rename target and convert to factor
raw <- raw %>%
  rename(Diabetes = Diabetes_binary) %>%
  mutate(Diabetes = factor(Diabetes, levels = c(0, 1), labels = c("No", "Yes")))


# Recode binary variables
raw <- raw %>%
  mutate(
    HighBP     = factor(HighBP,     levels = c(0, 1), labels = c("No", "Yes")),
    HighChol   = factor(HighChol,   levels = c(0, 1), labels = c("No", "Yes")),
    CholCheck  = factor(CholCheck,  levels = c(0, 1), labels = c("No", "Yes")),
    Smoker     = factor(Smoker,     levels = c(0, 1), labels = c("No", "Yes")),
    Stroke     = factor(Stroke,     levels = c(0, 1), labels = c("No", "Yes")),
    HeartDiseaseorAttack = factor(HeartDiseaseorAttack, levels = c(0, 1), labels = c("No", "Yes")),
    PhysActivity = factor(PhysActivity, levels = c(0, 1), labels = c("No", "Yes")),
    Fruits     = factor(Fruits,     levels = c(0, 1), labels = c("No", "Yes")),
    Veggies    = factor(Veggies,    levels = c(0, 1), labels = c("No", "Yes")),
    HvyAlcoholConsump = factor(HvyAlcoholConsump, levels = c(0, 1), labels = c("No", "Yes")),
    AnyHealthcare = factor(AnyHealthcare, levels = c(0, 1), labels = c("No", "Yes")),
    NoDocbcCost = factor(NoDocbcCost, levels = c(0, 1), labels = c("No", "Yes")),
    DiffWalk   = factor(DiffWalk,   levels = c(0, 1), labels = c("No", "Yes")),
    Sex        = factor(Sex,        levels = c(0, 1), labels = c("Female", "Male"))
  )


# Recode ordinal variables
raw <- raw %>%
  mutate(
    GenHlth   = factor(GenHlth, levels = 1:5,
                       labels = c("Excellent", "Very Good", "Good", "Fair", "Poor"),
                       ordered = TRUE),
    Education = factor(Education, levels = 1:6,
                       labels = c("Never attended", "Elementary", "Some high school",
                                  "High school graduate", "Some college", "College graduate"),
                       ordered = TRUE),
    Income    = factor(Income, levels = 1:8,
                       labels = c("<$10k", "$10k-$15k", "$15k-$20k", "$20k-$25k",
                                  "$25k-$35k", "$35k-$50k", "$50k-$75k", "$75k+"),
                       ordered = TRUE),
    Age       = factor(Age, levels = 1:13,
                       labels = c("18-24", "25-29", "30-34", "35-39", "40-44",
                                  "45-49", "50-54", "55-59", "60-64", "65-69",
                                  "70-74", "75-79", "80+"),
                       ordered = TRUE)
  )


# Feature engineering
age_mid <- c(21,27,32,37,42,47,52,57,62,67,72,77,85)
raw <- raw %>%
  mutate(
    Age_numeric = age_mid[as.integer(Age)],
    risk_count = (HighBP == "Yes") + (HighChol == "Yes") + (BMI >= 30) +
      (Smoker == "Yes") + (PhysActivity == "No") + (HvyAlcoholConsump == "Yes"),
    GenHlth_num = as.integer(GenHlth)
    
  )


# Train/test split (stratified)
set.seed(2024)
split <- initial_split(raw, prop = 0.8, strata = Diabetes)
train <- training(split)
test  <- testing(split)

cat("\nTraining rows:", nrow(train), "| Test rows:", nrow(test), "\n")
cat("Train diabetes %:", round(mean(train$Diabetes == "Yes") * 100, 2), "%\n")
cat("Test diabetes %: ", round(mean(test$Diabetes == "Yes") * 100, 2), "%\n")


# One-hot encoding
train <- train %>%
  dplyr::select(-Age, -GenHlth)

test <- test %>%
  dplyr::select(-Age, -GenHlth)

mod_recipe <- recipe(Diabetes ~ ., data = train) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  prep(retain = TRUE)

X_train <- bake(mod_recipe, new_data = train, all_predictors()) %>%
  mutate(across(everything(), as.numeric)) %>%
  as.matrix()

X_test <- bake(mod_recipe, new_data = test, all_predictors()) %>%
  mutate(across(everything(), as.numeric)) %>%
  as.matrix()

storage.mode(X_train) <- "double"
storage.mode(X_test)  <- "double"

Y_train <- as.integer(train$Diabetes == "Yes")
Y_test  <- as.integer(test$Diabetes == "Yes")

stopifnot(
  is.matrix(X_train),
  is.matrix(X_test),
  is.numeric(X_train),
  is.numeric(X_test),
  !anyNA(X_train),
  !anyNA(X_test)
)

cat("X_train dimensions:", nrow(X_train), "x", ncol(X_train), "\n")
cat("X_test dimensions :", nrow(X_test), "x", ncol(X_test), "\n")


# 5-fold Cross-Validation
set.seed(2024)
folds <- vfold_cv(train, v = 5, strata = Diabetes)

train_oof <- data.frame(
  row_id   = seq_len(nrow(train)),
  Diabetes = train$Diabetes,
  p_freq   = NA_real_,
  p_bayes  = NA_real_,
  p_rf     = NA_real_,
  p_xgb    = NA_real_
)

cat("\nStarting 5-fold CV...\n")

pb <- progress_bar$new(
  format = " Fold :current/:total [:bar] :percent | ETA: :eta",
  total = length(folds$splits),
  clear = FALSE,
  width = 80
)

for(i in seq_along(folds$splits)){
  
  cat(sprintf("\n Fold %d of %d...\n", i, length(folds$splits)))
  
  train_idx <- folds$splits[[i]]$in_id
  holdout_idx <- setdiff(seq_len(nrow(train)), train_idx)
  
  x_tr <- X_train[train_idx, , drop = FALSE]
  y_tr <- Y_train[train_idx]
  
  x_ho <- X_train[holdout_idx, , drop = FALSE]
  
  ## Frequentist
  cat("   - Elastic Net...")
  cv_glm <- cv.glmnet(
    x = x_tr,
    y = y_tr,
    family = "binomial",
    alpha = 0.5,
    nfolds = 3
  )
  
  train_oof$p_freq[holdout_idx] <-
    predict(cv_glm,
            newx = x_ho,
            s = "lambda.min",
            type = "response")[,1]
  cat(" ✓\n")
  
  ## Bayesian
  cat("   - Bayesian...")
  df_tr <- as.data.frame(x_tr)
  df_tr$y <- y_tr
  
  bayes_fit <- bayesglm(
    y ~ .,
    data = df_tr,
    family = binomial(),
    prior.scale = 2.5,
    prior.df = Inf
  )
  
  df_ho <- as.data.frame(x_ho)
  
  train_oof$p_bayes[holdout_idx] <-
    predict(
      bayes_fit,
      newdata = df_ho,
      type = "response"
    )
  cat(" ✓\n")
  
  ## Random Forest
  cat("   - Random Forest...")
  rf_fit <- ranger(
    x = x_tr,
    y = factor(y_tr, levels = c(0,1)),
    probability = TRUE,
    num.trees = 500,
    mtry = floor(sqrt(ncol(x_tr))),
    seed = 2024
  )
  
  train_oof$p_rf[holdout_idx] <-
    predict(rf_fit, data = x_ho)$predictions[, "1"]
  cat(" ✓\n")
  
  ## XGBoost
  cat("   - XGBoost...")
  
  dtrain <- xgb.DMatrix(
    data = x_tr,
    label = y_tr
  )
  
  dholdout <- xgb.DMatrix(
    data = x_ho
  )
  
  xgb_fit <- xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = 0.1,
      max_depth = 3
    ),
    data = dtrain,
    nrounds = 100,
    verbose = 0
  )
  
  train_oof$p_xgb[holdout_idx] <- predict(
    xgb_fit,
    newdata = dholdout
  )
  
  cat(" ✓\n")
  
  pb$tick()
}

pb$terminate()
beepr::beep(2)

cat("\nOOF prediction means:\n")
train_oof %>%
  summarise(across(starts_with("p_"), mean)) %>%
  print()


# Refit on full training & predict test set
cat("\nRefitting final models on full training data...\n")


# Elastic Net
final_glm <- cv.glmnet(
  X_train,
  Y_train,
  family = "binomial",
  alpha = 0.5,
  nfolds = 5
)

p_test_freq <- predict(
  final_glm,
  newx = X_test,
  s = "lambda.min",
  type = "response"
)[,1]

# Bayesian Logistic Regression
df_train <- as.data.frame(X_train)
df_train$y <- Y_train

final_bayes <- bayesglm(
  y ~ .,
  data = df_train,
  family = binomial(),
  prior.scale = 2.5,
  prior.df = Inf
)

p_test_bayes <- predict(
  final_bayes,
  newdata = as.data.frame(X_test),
  type = "response"
)


# Random Forest
set.seed(2024)

final_rf <- ranger(
  x = X_train,
  y = factor(Y_train, levels = c(0,1), labels = c("No","Yes")),
  probability = TRUE,
  num.trees = 500,
  mtry = floor(sqrt(ncol(X_train))),
  seed = 2024
)

p_test_rf <- predict(final_rf, data = X_test)$predictions[, "Yes"]


# XGBoost
set.seed(2024)

dtrain_full <- xgb.DMatrix(
  data = X_train,
  label = Y_train
)

dtest <- xgb.DMatrix(
  data = X_test
)

final_xgb <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.1,
    max_depth = 3,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = dtrain_full,
  nrounds = 100,
  verbose = 0
)

p_test_xgb <- predict(final_xgb, dtest)


# Assemble test predictions
test_pred <- tibble(
  Diabetes = test$Diabetes,
  p_freq   = p_test_freq,
  p_bayes  = p_test_bayes,
  p_rf     = p_test_rf,
  p_xgb    = p_test_xgb
) %>%
  mutate(
    p_mean4 = rowMeans(across(starts_with("p_")))
  )

cat("\nTest prediction means:\n")
test_pred %>%
  summarise(across(starts_with("p_"), mean)) %>%
  print()


# Discordance (4-model SD)
train_oof$p_mean4 <- rowMeans(
  train_oof[, c("p_freq","p_bayes","p_rf","p_xgb")]
)

train_oof$discordance_sd <- matrixStats::rowSds(
  as.matrix(train_oof[, c("p_freq","p_bayes","p_rf","p_xgb")])
)

train_oof$discordance_tertile <- factor(
  dplyr::ntile(train_oof$discordance_sd, 3),
  labels = c("Low","Medium","High")
)


test_pred$p_mean4 <- rowMeans(
  test_pred[, c("p_freq","p_bayes","p_rf","p_xgb")]
)

test_pred$discordance_sd <- matrixStats::rowSds(
  as.matrix(test_pred[, c("p_freq","p_bayes","p_rf","p_xgb")])
)

test_pred$discordance_tertile <- factor(
  dplyr::ntile(test_pred$discordance_sd, 3),
  labels = c("Low","Medium","High")
)


# Merge predictions with clinical data
train_full <- cbind(
  train,
  train_oof[, c(
    "p_freq",
    "p_bayes",
    "p_rf",
    "p_xgb",
    "p_mean4",
    "discordance_sd",
    "discordance_tertile"
  )]
)

test_full <- cbind(
  test,
  test_pred[, c(
    "p_freq",
    "p_bayes",
    "p_rf",
    "p_xgb",
    "p_mean4",
    "discordance_sd",
    "discordance_tertile"
  )]
)


# Table 1
tab_disc <- CreateTableOne(
  vars = c(
    "Age_numeric","Sex","BMI","HighBP","HighChol",
    "Smoker","Stroke","HeartDiseaseorAttack",
    "GenHlth","risk_count","Diabetes"
  ),
  strata = "discordance_tertile",
  data = train_full,
  factorVars = c(
    "Sex","HighBP","HighChol","Smoker",
    "Stroke","HeartDiseaseorAttack",
    "GenHlth","Diabetes"
  ),
  test = TRUE
)

cat("\nTable 1\n")
print(tab_disc, smd = TRUE)


# Outcome models
m1 <- glm(
  Diabetes ~ p_mean4,
  data = test_full,
  family = binomial
)

m2 <- glm(
  Diabetes ~ p_mean4 + discordance_sd,
  data = test_full,
  family = binomial
)

cat("\nModel 1 AIC:", AIC(m1), "\n")
cat("Model 2 AIC:", AIC(m2), "\n")

print(anova(m1, m2, test = "Chisq"))

summary(m2)

pscl::pR2(m2)

exp(cbind(OR = coef(m2), confint(m2)))


# 05 – Validation, Calibration & Explainability


# Load additional packages
library(ggplot2)
library(patchwork)
library(pROC)
library(nricens)
library(rms)
library(dcurves)


# Helper function
ggplot(test_full,
       aes(p_mean4, discordance_sd)) +
  geom_point(alpha = 0.05) +
  geom_smooth(method = "loess",
              colour = "red",
              se = TRUE) +
  labs(
    title = "Relationship Between Consensus Risk and Model Discordance",
    x = "Consensus Predicted Probability",
    y = "Discordance (SD)"
  ) +
  theme_minimal()


# 1. Calibration plot (p_mean4, 4‑model tertiles)
ggplot(test_full, aes(x = p_mean4, y = as.numeric(Diabetes == "Yes"), 
                      color = discordance_tertile, fill = discordance_tertile)) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.15, span = 0.8) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Consensus predicted probability (p_mean4)", 
       y = "Observed proportion with diabetes",
       title = "Calibration by Discordance Tertile (Test Set, 4‑model)") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1")


# 2. Quantitative calibration metrics
cat("\n--- Calibration metrics (Overall) ---\n")
Y_test_num <- as.integer(test_full$Diabetes == "Yes")
val.prob(
  p = test_full$p_mean4,
  y = Y_test_num,
  pl = TRUE
)
cat("\n--- Calibration metrics by Discordance Tertile ---\n")
for (tert in c("Low", "Medium", "High")) {
  idx <- test_full$discordance_tertile == tert
  cat("\n", tert, "Discordance\n")
  cat(rep("-", 30), "\n", sep = "")
  val.prob(
    p = test_full$p_mean4[idx],
    y = Y_test_num[idx],
    pl = TRUE
  )
}


# 3. Discordance density plot + calibration summary
p1 <- ggplot(test_full, aes(x = discordance_sd, fill = Diabetes)) +
  geom_density(alpha = 0.5) +
  scale_fill_brewer(palette = "Set1") +
  labs(
    title = "Distribution of Model Discordance by Diabetes Status",
    x = "Model Discordance (SD)",
    y = "Density"
  ) +
  theme_minimal()
p2 <- ggplot(
  test_full,
  aes(
    x = p_mean4,
    y = as.numeric(Diabetes == "Yes"),
    color = discordance_tertile
  )
) +
  geom_smooth(method = "loess", se = FALSE) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  scale_color_brewer(palette = "Set1") +
  labs(
    title = "Calibration by Discordance Tertile",
    x = "Consensus Predicted Probability",
    y = "Observed Diabetes"
  ) +
  theme_minimal()
p1 / p2


# 4. Severity‑adjusted logistic models


# This directly tests whether discordance adds information beyond known severity markers.
cat("\n--- Severity‑adjusted models ---\n")
m_severity <- glm(Diabetes ~ p_mean4 + Age_numeric + BMI + risk_count + GenHlth_num,
                  data = test_full, family = binomial)
m_severity_disc <- glm(Diabetes ~ p_mean4 + Age_numeric + BMI + risk_count + GenHlth_num 
                       + discordance_sd, data = test_full, family = binomial)
cat("Likelihood ratio test (severity‑adjusted):\n")
print(anova(m_severity, m_severity_disc, test = "Chisq"))
cat("\nDiscordance coefficient in full model:\n")
summary(m_severity_disc)
cat("\nAdjusted Odds Ratios\n")

print(
  exp(
    cbind(
      OR = coef(m_severity_disc),
      confint(m_severity_disc)
    )
  )
)


cat("\nCorrelation between consensus risk and discordance:\n")
print(cor(test_full$p_mean4, test_full$discordance_sd))

cat("\nVariance Inflation Factors:\n")
print(vif(m_severity_disc))


# 5. SHAP explainability (XGBoost, reliable)

# Feature importance (gain‑based)
importance <- xgb.importance(
  model = final_xgb,
  feature_names = colnames(X_train)
)

print(importance[1:10,])

xgb.plot.importance(
  importance[1:15,]
)


# 6. AUC comparison
roc1 <- roc(test_full$Diabetes, test_full$p_mean4)
roc2 <- roc(test_full$Diabetes, predict(m2, type = "response"))
cat("\nAUC (consensus mean only):", auc(roc1), 
    "| AUC (+ discordance):", auc(roc2), "\n")

plot(
  roc1,
  legacy.axes = TRUE,
  print.auc = TRUE
)

plot(
  roc2,
  add = TRUE,
  col = "red",
  print.auc = TRUE,
  print.auc.y = 0.4
)

legend(
  "bottomright",
  legend = c(
    "Consensus",
    "Consensus + Discordance"
  ),
  col = c("black","red"),
  lwd = 2
)

# 7. Net Reclassification Improvement (NRI)
p_risk1 <- test_full$p_mean4
p_risk2 <- predict(m2, type = "response")
nri <- nribin(event = test_full$Diabetes == "Yes",
              p.std = p_risk1, p.new = p_risk2, cut = c(0.1, 0.2))
cat("\nNRI (thresholds 0.1, 0.2):\n")
print(nri$nri)

# 8. Decision Curve Analysis


# Compare strategies: treat all based on consensus mean vs. treat all based on model with discordance
dca_data <- data.frame(
  Diabetes = as.integer(test_full$Diabetes == "Yes"),
  model_mean = test_full$p_mean4,
  model_disc = predict(m2, type = "response")
)
dca <- dcurves::dca(Diabetes ~ model_mean + model_disc, data = dca_data,
                    thresholds = seq(0, 0.4, by = 0.01))
plot(dca) + labs(title = "Decision Curve Analysis (Test Set)")
