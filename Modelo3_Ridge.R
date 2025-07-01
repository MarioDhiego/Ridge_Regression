# Carregar pacotes
library(readxl)
library(caret)
library(glmnet)
library(car)
library(corrplot)

# Ler os dados
dados <- read_excel("dados_pinus.xlsx")

#------------------ Modelo Ridge Completo ----------------------------#
set.seed(123)
train_index <- createDataPartition(dados$VOLUME, p = 0.7, list = FALSE)
dados_treino <- dados[train_index, ]
dados_teste <- dados[-train_index, ]

x_train <- model.matrix(VOLUME ~ ., dados_treino)[, -1]  # Remove intercepto
y_train <- dados_treino$VOLUME

x_test <- model.matrix(VOLUME ~ ., dados_teste)[, -1]
y_test <- dados_teste$VOLUME

# Regressão Ridge (alpha = 0)
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, standardize = TRUE)

# Melhor lambda
best_lambda_ridge <- cv_ridge$lambda.min
cat("Melhor lambda (Ridge):", best_lambda_ridge, "\n")


modelo_ridge <- glmnet(x_train, y_train,
                       alpha = 0,
                       lambda = best_lambda_ridge,
                       standardize = TRUE)


# Coeficientes do modelo final
coef_ridge <- coef(modelo_ridge)
print(coef_ridge)

# Predição no conjunto de teste
y_pred_ridge <- predict(modelo_ridge, s = best_lambda_ridge, newx = x_test)

R2_ridge <- cor(y_test, y_pred_ridge)^2
rmse_ridge <- sqrt(mean((y_test - y_pred_ridge)^2))
mae_ridge <- mean(abs(y_test - y_pred_ridge))

cat("\nDesempenho do modelo Ridge (com todas as variáveis):\n")
cat("R²:", R2_ridge, "\nRMSE:", rmse_ridge, "\nMAE:", mae_ridge, "\n")

modelo_ridge_seq <- glmnet(x_train, y_train, alpha = 0, standardize = TRUE)
plot(modelo_ridge_seq, xvar = "lambda", label = TRUE,
     main = "Ridge Trace Plot: Coeficientes vs log(Lambda)",
     xlab = "log(Lambda)", ylab = "Coeficientes")
abline(v = log(best_lambda_ridge), col = "red", lty = 2)


#-----------------------------------------------------------------------#
# Modelo Ridge sem (Area Basal)

# Remover a variável AREA_BASAL
dados_sem_area <- dados %>% select(-AREA_BASAL)

# Dividir em treino/teste
set.seed(123)
train_index2 <- createDataPartition(dados_sem_area$VOLUME, p = 0.7, list = FALSE)
dados_treino2 <- dados_sem_area[train_index2, ]
dados_teste2 <- dados_sem_area[-train_index2, ]

# Matriz de preditores
x_train2 <- model.matrix(VOLUME ~ ., dados_treino2)[, -1]
y_train2 <- dados_treino2$VOLUME

x_test2 <- model.matrix(VOLUME ~ ., dados_teste2)[, -1]
y_test2 <- dados_teste2$VOLUME

# Ridge com CV sem Área Basal
cv_ridge2 <- cv.glmnet(x_train2, y_train2, alpha = 0, standardize = TRUE)
best_lambda_ridge2 <- cv_ridge2$lambda.min
modelo_ridge2 <- glmnet(x_train2, y_train2, alpha = 0, lambda = best_lambda_ridge2, standardize = TRUE)

# Predição e avaliação
y_pred_ridge2 <- predict(modelo_ridge2, s = best_lambda_ridge2, newx = x_test2)
R2_ridge2 <- cor(y_test2, y_pred_ridge2)^2
rmse_ridge2 <- sqrt(mean((y_test2 - y_pred_ridge2)^2))
mae_ridge2 <- mean(abs(y_test2 - y_pred_ridge2))

cat("\nRidge SEM Área Basal:\n")
cat("R²:", R2_ridge2, "\nRMSE:", rmse_ridge2, "\nMAE:", mae_ridge2, "\n")
#-----------------------------------------------------------------------#

#------------------ Modelo Ridge (sem DAP) ----------------------------#
# Remover DAP
dados_sem_dap <- dados %>% select(-DAP)

# Dividir dados em treino e teste
set.seed(123)
train_idx_dap <- createDataPartition(dados_sem_dap$VOLUME, p = 0.7, list = FALSE)
dados_treino_dap <- dados_sem_dap[train_idx_dap, ]
dados_teste_dap  <- dados_sem_dap[-train_idx_dap, ]

x_train_dap <- model.matrix(VOLUME ~ ., dados_treino_dap)[, -1]
y_train_dap <- dados_treino_dap$VOLUME

x_test_dap <- model.matrix(VOLUME ~ ., dados_teste_dap)[, -1]
y_test_dap <- dados_teste_dap$VOLUME

cv_ridge_dap <- cv.glmnet(x_train_dap, y_train_dap, alpha = 0, standardize = TRUE)
best_lambda_dap <- cv_ridge_dap$lambda.min
cat("Melhor lambda (sem DAP):", best_lambda_dap, "\n")

modelo_ridge_dap <- glmnet(x_train_dap, y_train_dap,
                           alpha = 0,
                           lambda = best_lambda_dap,
                           standardize = TRUE)
coef(modelo_ridge_dap)

# Predição
y_pred_dap <- predict(modelo_ridge_dap, s = best_lambda_dap, newx = x_test_dap)

# Métricas de desempenho
R2_dap <- cor(y_test_dap, y_pred_dap)^2
rmse_dap <- sqrt(mean((y_test_dap - y_pred_dap)^2))
mae_dap <- mean(abs(y_test_dap - y_pred_dap))

cat("\nModelo Ridge SEM DAP:\n")
cat("R²:", R2_dap, "\nRMSE:", rmse_dap, "\nMAE:", mae_dap, "\n")



#------------------ Modelo Ridge (sem DAP + Area basal) ----------------------------#
# Remover as duas variáveis
dados_sem_dap_ab <- dados %>% select(-DAP, -AREA_BASAL)

# Dividir em treino e teste
set.seed(123)
train_idx_dap_ab <- createDataPartition(dados_sem_dap_ab$VOLUME, p = 0.7, list = FALSE)
dados_treino_dap_ab <- dados_sem_dap_ab[train_idx_dap_ab, ]
dados_teste_dap_ab  <- dados_sem_dap_ab[-train_idx_dap_ab, ]


x_train_dap_ab <- model.matrix(VOLUME ~ ., dados_treino_dap_ab)[, -1]
y_train_dap_ab <- dados_treino_dap_ab$VOLUME

x_test_dap_ab <- model.matrix(VOLUME ~ ., dados_teste_dap_ab)[, -1]
y_test_dap_ab <- dados_teste_dap_ab$VOLUME

cv_ridge_dap_ab <- cv.glmnet(x_train_dap_ab, y_train_dap_ab, alpha = 0, standardize = TRUE)
best_lambda_dap_ab <- cv_ridge_dap_ab$lambda.min
cat("Melhor lambda (sem DAP e Área Basal):", best_lambda_dap_ab, "\n")

modelo_ridge_dap_ab <- glmnet(x_train_dap_ab, y_train_dap_ab,
                              alpha = 0,
                              lambda = best_lambda_dap_ab,
                              standardize = TRUE)


y_pred_dap_ab <- predict(modelo_ridge_dap_ab, s = best_lambda_dap_ab, newx = x_test_dap_ab)

R2_dap_ab <- cor(y_test_dap_ab, y_pred_dap_ab)^2
rmse_dap_ab <- sqrt(mean((y_test_dap_ab - y_pred_dap_ab)^2))
mae_dap_ab <- mean(abs(y_test_dap_ab - y_pred_dap_ab))

cat("\nModelo Ridge SEM DAP e SEM Área Basal:\n")
cat("R²:", R2_dap_ab, "\nRMSE:", rmse_dap_ab, "\nMAE:", mae_dap_ab, "\n")
#------------------------------------------------------------------------------#



#--------------- Modelo Lasso Geral --------------------------------------------#
# Preparar dados completos
set.seed(123)
train_index_lasso <- createDataPartition(dados$VOLUME, p = 0.7, list = FALSE)
dados_treino_lasso <- dados[train_index_lasso, ]
dados_teste_lasso  <- dados[-train_index_lasso, ]

x_train_lasso <- model.matrix(VOLUME ~ ., dados_treino_lasso)[, -1]
y_train_lasso <- dados_treino_lasso$VOLUME

x_test_lasso <- model.matrix(VOLUME ~ ., dados_teste_lasso)[, -1]
y_test_lasso <- dados_teste_lasso$VOLUME

# Validação cruzada para Lasso (alpha = 1)
cv_lasso <- cv.glmnet(x_train_lasso, y_train_lasso, alpha = 1, standardize = TRUE)
best_lambda_lasso <- cv_lasso$lambda.min
modelo_lasso <- glmnet(x_train_lasso, y_train_lasso, alpha = 1, lambda = best_lambda_lasso, standardize = TRUE)

# Predição no teste
y_pred_lasso <- predict(modelo_lasso, s = best_lambda_lasso, newx = x_test_lasso)
R2_lasso <- cor(y_test_lasso, y_pred_lasso)^2
rmse_lasso <- sqrt(mean((y_test_lasso - y_pred_lasso)^2))
mae_lasso <- mean(abs(y_test_lasso - y_pred_lasso))

# Variáveis selecionadas
coef_lasso <- coef(modelo_lasso)
variaveis_lasso <- rownames(coef_lasso)[coef_lasso[, 1] != 0 & rownames(coef_lasso) != "(Intercept)"]

cat("Lasso COM todas as variáveis:\n")
cat("R²:", R2_lasso, "\nRMSE:", rmse_lasso, "\nMAE:", mae_lasso, "\n")
cat("Variáveis selecionadas:", paste(variaveis_lasso, collapse = ", "), "\n")



#--------------- Modelo Lasso (Sem Aerea Basal) --------------------------------------------#
# Remover AREA_BASAL
dados_lasso_ab <- dados %>% select(-AREA_BASAL)

# Dividir treino/teste
set.seed(123)
train_index_lasso_ab <- createDataPartition(dados_lasso_ab$VOLUME, p = 0.7, list = FALSE)
dados_treino_lasso_ab <- dados_lasso_ab[train_index_lasso_ab, ]
dados_teste_lasso_ab  <- dados_lasso_ab[-train_index_lasso_ab, ]

# Matriz de preditores
x_train_lasso_ab <- model.matrix(VOLUME ~ ., dados_treino_lasso_ab)[, -1]
y_train_lasso_ab <- dados_treino_lasso_ab$VOLUME

x_test_lasso_ab <- model.matrix(VOLUME ~ ., dados_teste_lasso_ab)[, -1]
y_test_lasso_ab <- dados_teste_lasso_ab$VOLUME

# Ajuste do modelo Lasso com validação cruzada
cv_lasso_ab <- cv.glmnet(x_train_lasso_ab, y_train_lasso_ab, alpha = 1, standardize = TRUE)
best_lambda_lasso_ab <- cv_lasso_ab$lambda.min

modelo_lasso_ab <- glmnet(x_train_lasso_ab, y_train_lasso_ab,
                          alpha = 1,
                          lambda = best_lambda_lasso_ab,
                          standardize = TRUE)

# Predição e métricas
y_pred_lasso_ab <- predict(modelo_lasso_ab, s = best_lambda_lasso_ab, newx = x_test_lasso_ab)
R2_lasso_ab <- cor(y_test_lasso_ab, y_pred_lasso_ab)^2
rmse_lasso_ab <- sqrt(mean((y_test_lasso_ab - y_pred_lasso_ab)^2))
mae_lasso_ab <- mean(abs(y_test_lasso_ab - y_pred_lasso_ab))

# Variáveis selecionadas
coef_lasso_ab <- coef(modelo_lasso_ab)
variaveis_lasso_ab <- rownames(coef_lasso_ab)[coef_lasso_ab[, 1] != 0 & rownames(coef_lasso_ab) != "(Intercept)"]

# Resultado
cat("\nLasso SEM AREA_BASAL:\n")
cat("R²:", R2_lasso_ab, "\nRMSE:", rmse_lasso_ab, "\nMAE:", mae_lasso_ab, "\n")
cat("Variáveis selecionadas:", paste(variaveis_lasso_ab, collapse = ", "), "\n")


#--------------- Modelo Lasso (Sem DAP) --------------------------------------------#
# Remover DAP
dados_lasso_dap <- dados %>% select(-DAP)

# Dividir treino/teste
set.seed(123)
train_index_lasso_dap <- createDataPartition(dados_lasso_dap$VOLUME, p = 0.7, list = FALSE)
dados_treino_lasso_dap <- dados_lasso_dap[train_index_lasso_dap, ]
dados_teste_lasso_dap  <- dados_lasso_dap[-train_index_lasso_dap, ]

# Matriz de preditores
x_train_lasso_dap <- model.matrix(VOLUME ~ ., dados_treino_lasso_dap)[, -1]
y_train_lasso_dap <- dados_treino_lasso_dap$VOLUME

x_test_lasso_dap <- model.matrix(VOLUME ~ ., dados_teste_lasso_dap)[, -1]
y_test_lasso_dap <- dados_teste_lasso_dap$VOLUME

# Ajustar modelo Lasso
cv_lasso_dap <- cv.glmnet(x_train_lasso_dap, y_train_lasso_dap, alpha = 1, standardize = TRUE)
best_lambda_lasso_dap <- cv_lasso_dap$lambda.min

modelo_lasso_dap <- glmnet(x_train_lasso_dap, y_train_lasso_dap,
                           alpha = 1,
                           lambda = best_lambda_lasso_dap,
                           standardize = TRUE)

# Predição e métricas
y_pred_lasso_dap <- predict(modelo_lasso_dap, s = best_lambda_lasso_dap, newx = x_test_lasso_dap)
R2_lasso_dap <- cor(y_test_lasso_dap, y_pred_lasso_dap)^2
rmse_lasso_dap <- sqrt(mean((y_test_lasso_dap - y_pred_lasso_dap)^2))
mae_lasso_dap <- mean(abs(y_test_lasso_dap - y_pred_lasso_dap))

# Variáveis selecionadas
coef_lasso_dap <- coef(modelo_lasso_dap)
variaveis_lasso_dap <- rownames(coef_lasso_dap)[coef_lasso_dap[, 1] != 0 & rownames(coef_lasso_dap) != "(Intercept)"]

# Resultado
cat("\nLasso SEM DAP:\n")
cat("R²:", R2_lasso_dap, "\nRMSE:", rmse_lasso_dap, "\nMAE:", mae_lasso_dap, "\n")
cat("Variáveis selecionadas:", paste(variaveis_lasso_dap, collapse = ", "), "\n")



#--------------- Modelo Lasso (Sem DAP + Area basal) --------------------------------------------#
# Remover DAP e AREA_BASAL
dados_lasso_dap_ab <- dados %>% select(-DAP, -AREA_BASAL)

# Dividir treino/teste
set.seed(123)
train_index_lasso_dap_ab <- createDataPartition(dados_lasso_dap_ab$VOLUME, p = 0.7, list = FALSE)
dados_treino_lasso_dap_ab <- dados_lasso_dap_ab[train_index_lasso_dap_ab, ]
dados_teste_lasso_dap_ab  <- dados_lasso_dap_ab[-train_index_lasso_dap_ab, ]

# Matriz preditores
x_train_lasso_dap_ab <- model.matrix(VOLUME ~ ., dados_treino_lasso_dap_ab)[, -1]
y_train_lasso_dap_ab <- dados_treino_lasso_dap_ab$VOLUME

x_test_lasso_dap_ab <- model.matrix(VOLUME ~ ., dados_teste_lasso_dap_ab)[, -1]
y_test_lasso_dap_ab <- dados_teste_lasso_dap_ab$VOLUME

# Ajustar modelo Lasso com CV
cv_lasso_dap_ab <- cv.glmnet(x_train_lasso_dap_ab, y_train_lasso_dap_ab, alpha = 1, standardize = TRUE)
best_lambda_lasso_dap_ab <- cv_lasso_dap_ab$lambda.min

modelo_lasso_dap_ab <- glmnet(x_train_lasso_dap_ab, y_train_lasso_dap_ab,
                              alpha = 1,
                              lambda = best_lambda_lasso_dap_ab,
                              standardize = TRUE)

# Predição e métricas
y_pred_lasso_dap_ab <- predict(modelo_lasso_dap_ab, s = best_lambda_lasso_dap_ab, newx = x_test_lasso_dap_ab)
R2_lasso_dap_ab <- cor(y_test_lasso_dap_ab, y_pred_lasso_dap_ab)^2
rmse_lasso_dap_ab <- sqrt(mean((y_test_lasso_dap_ab - y_pred_lasso_dap_ab)^2))
mae_lasso_dap_ab <- mean(abs(y_test_lasso_dap_ab - y_pred_lasso_dap_ab))

# Variáveis selecionadas
coef_lasso_dap_ab <- coef(modelo_lasso_dap_ab)
variaveis_lasso_dap_ab <- rownames(coef_lasso_dap_ab)[coef_lasso_dap_ab[, 1] != 0 & rownames(coef_lasso_dap_ab) != "(Intercept)"]

# Resultado
cat("\nLasso SEM DAP e SEM AREA_BASAL:\n")
cat("R²:", R2_lasso_dap_ab, "\nRMSE:", rmse_lasso_dap_ab, "\nMAE:", mae_lasso_dap_ab, "\n")
cat("Variáveis selecionadas:", paste(variaveis_lasso_dap_ab, collapse = ", "), "\n")



#------------------- MODELO Elastic Net ----------------------------------------#

library(caret)
library(glmnet)
library(dplyr)

set.seed(123)

# Dividir dados em treino (70%) e teste (30%)
train_idx <- createDataPartition(dados$VOLUME, p = 0.7, list = FALSE)
dados_treino <- dados[train_idx, ]
dados_teste  <- dados[-train_idx, ]

# Criar matriz preditora e vetor resposta para treino e teste
x_train <- model.matrix(VOLUME ~ ., dados_treino)[, -1]
y_train <- dados_treino$VOLUME

x_test <- model.matrix(VOLUME ~ ., dados_teste)[, -1]
y_test <- dados_teste$VOLUME

# Definir grid para alpha (0 a 1 de 0.1 em 0.1), lambda será ajustado automaticamente
tuneGrid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),
  lambda = NA  # caret vai ajustar
)

# Controle de treino: 10-fold Cross Validation
train_control <- trainControl(method = "cv", number = 10)

# Treinar Elastic Net com caret
elastic_net_model <- train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  tuneGrid = tuneGrid,
  trControl = train_control,
  preProcess = c("center", "scale")
)

# Mostrar os melhores hiperparâmetros
best_alpha <- elastic_net_model$bestTune$alpha
best_lambda <- elastic_net_model$bestTune$lambda
cat("Melhor alpha (mix Ridge/Lasso):", best_alpha, "\n")
cat("Melhor lambda (penalização):", best_lambda, "\n")

# Predição e avaliação no conjunto teste
y_pred <- predict(elastic_net_model, newdata = x_test)
R2 <- cor(y_test, y_pred)^2
rmse <- sqrt(mean((y_test - y_pred)^2))
mae <- mean(abs(y_test - y_pred))

cat("Desempenho no teste:\n")
cat("R²:", R2, "\nRMSE:", rmse, "\nMAE:", mae, "\n")

# Coeficientes do modelo final
coef_elastic <- coef(elastic_net_model$finalModel, s = best_lambda)
variaveis_selecionadas <- rownames(coef_elastic)[coef_elastic[, 1] != 0 & rownames(coef_elastic) != "(Intercept)"]
cat("Variáveis selecionadas:", paste(variaveis_selecionadas, collapse = ", "), "\n")















# Pacotes necessários
library(glmnet)
library(caret)

# Selecionar variáveis
vars_selecionadas <- c("DAP", "ALTURA", "IDADE", "DAF")
formula <- as.formula(paste("VOLUME ~", paste(vars_selecionadas, collapse = " + ")))

# Divisão treino/teste
set.seed(123)
train_index <- createDataPartition(dados$VOLUME, p = 0.7, list = FALSE)
dados_treino <- dados[train_index, ]
dados_teste  <- dados[-train_index, ]

# Matriz para glmnet
x_treino <- model.matrix(formula, dados_treino)[, -1]
y_treino <- dados_treino$VOLUME

x_teste <- model.matrix(formula, dados_teste)[, -1]
y_teste <- dados_teste$VOLUME

### ------------------- RIDGE -------------------
cv_ridge <- cv.glmnet(x_treino, y_treino, alpha = 0, standardize = TRUE)
best_lambda_ridge <- cv_ridge$lambda.min
modelo_ridge <- glmnet(x_treino, y_treino, alpha = 0, lambda = best_lambda_ridge, standardize = TRUE)
pred_ridge <- predict(modelo_ridge, newx = x_teste, s = best_lambda_ridge)

# Métricas Ridge
r2_ridge <- cor(y_teste, pred_ridge)^2
rmse_ridge <- sqrt(mean((y_teste - pred_ridge)^2))
mae_ridge <- mean(abs(y_teste - pred_ridge))
mape_ridge <- mean(abs((y_teste - pred_ridge) / y_teste)) * 100

### ------------------- LASSO -------------------
cv_lasso <- cv.glmnet(x_treino, y_treino, alpha = 1, standardize = TRUE)
best_lambda_lasso <- cv_lasso$lambda.min
modelo_lasso <- glmnet(x_treino, y_treino, alpha = 1, lambda = best_lambda_lasso, standardize = TRUE)
pred_lasso <- predict(modelo_lasso, newx = x_teste, s = best_lambda_lasso)

# Métricas Lasso
r2_lasso <- cor(y_teste, pred_lasso)^2
rmse_lasso <- sqrt(mean((y_teste - pred_lasso)^2))
mae_lasso <- mean(abs(y_teste - pred_lasso))
mape_lasso <- mean(abs((y_teste - pred_lasso) / y_teste)) * 100

### ------------------- Resultados -------------------
cat("===== RIDGE =====\n")
cat("Lambda:", best_lambda_ridge, "\n")
cat("R²:", round(r2_ridge, 4), "\n")
cat("RMSE:", round(rmse_ridge, 4), "\n")
cat("MAE:", round(mae_ridge, 4), "\n")
cat("MAPE (%):", round(mape_ridge, 2), "\n")
cat("Coeficientes:\n")
print(coef(modelo_ridge, s = best_lambda_ridge))

cat("\n===== LASSO =====\n")
cat("Lambda:", best_lambda_lasso, "\n")
cat("R²:", round(r2_lasso, 4), "\n")
cat("RMSE:", round(rmse_lasso, 4), "\n")
cat("MAE:", round(mae_lasso, 4), "\n")
cat("MAPE (%):", round(mape_lasso, 2), "\n")
cat("Coeficientes:\n")
print(coef(modelo_lasso, s = best_lambda_lasso))










