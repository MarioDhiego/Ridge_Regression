#------------------------------------------------------------------------------#
# - Pacotes
library(readxl)
library(dplyr)
library(tidyr)
library(caret)
library(glmnet)
library(car)
library(corrplot)
library(ggcorrplot)
library(ggplot2)
library(reshape2)
library(psych)
library(GGally)
library(randomForest)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#-Leitura da Base de Dados
dados <- read_excel("dados_pinus.xlsx")
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Matriz de Correlação

#-Pacote coorplot
correl <- cor(dados[, -1])
corrplot(correl, 
         method = "color",         # circle, number, shade
         type = "upper",            # lower, full
         #order = "hclust",
         addCoef.col = "black",
         tl.srt = 90,
         diag = TRUE)

#-Pacote ggcorrplot
ggcorrplot(correl, 
           method = "square",      # circle
           hc.order = TRUE,
           type = "lower",         # upper
           lab = TRUE,
           lab_size = 4.0,
           colors= c("blue", "white", "red"),
           outline.color = "white",
           title = "Matriz de Correlação",
           insig = c("pch", "blank"),
           show.legend = TRUE,
           ggtheme = theme_gray()
)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Matriz de Dispersão 
library(psych)
library(GGally)

pairs.panels(dados,
             method = "pearson",
             density = TRUE,  
             ellipses = TRUE,
             smoother = TRUE)
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#-Regressão Linear Múltipla
#-Diagnóstico Inicial

modelo_mlm <- lm(VOLUME ~ ., data = dados)
summary(modelo_mlm)
vif(modelo_mlm)                                             # Multicolinearidade
#------------------------------------------------------------------------------#


#------------------ Modelo Ridge Completo ----------------------------#
set.seed(123)
train_index <- createDataPartition(dados$VOLUME, p = 0.9, list = FALSE)
dados_treino <- dados[train_index, ]
dados_teste <- dados[-train_index, ]

x_train <- model.matrix(VOLUME ~ ., dados_treino)[, -1]  # Remove intercepto
y_train <- dados_treino$VOLUME

x_test <- model.matrix(VOLUME ~ ., dados_teste)[, -1]
y_test <- dados_teste$VOLUME

# Regressão Ridge (Ridge:alpha = 0; LASSO:alpha = 1)
cv_ridge <- cv.glmnet(x_train, 
                      y_train, 
                      alpha = 0, 
                      standardize = TRUE)

# Melhor lambda
best_lambda_ridge <- cv_ridge$lambda.min
cat("Melhor lambda (Ridge):", best_lambda_ridge, "\n")


modelo_ridge <- glmnet(x_train, 
                       y_train,
                       alpha = 0,
                       lambda = best_lambda_ridge,
                       standardize = TRUE)


# Coeficientes do modelo final
coef_ridge <- coef(modelo_ridge)
print(coef_ridge)

# Predição no conjunto de teste
y_pred_ridge <- predict(modelo_ridge, s = best_lambda_ridge, newx = x_test)

#------------------------------------------------------------------------------#
# Medidas Performance

R2_ridge <- cor(y_test, y_pred_ridge)^2
rmse_ridge <- sqrt(mean((y_test - y_pred_ridge)^2))
mae_ridge <- mean(abs(y_test - y_pred_ridge))
mape_ridge <- mean(abs((y_test - y_pred_ridge) / y_test)) * 100


# --- NOVAS MÉTRICAS: AIC e BIC ---
# Predição nos dados de treino para obter o RSS
y_pred_train_ridge <- predict(modelo_ridge, s = best_lambda_ridge, newx = x_train)
RSS <- sum((y_train - y_pred_train_ridge)^2)

# Extrair os graus de liberdade efetivos do modelo
# O objeto glmnet já nos dá isso!
df <- modelo_ridge$df[which(modelo_ridge$lambda == best_lambda_ridge)]

# Número de observações no treino
n <- nrow(dados_treino)

# Cálculo do AIC e BIC
AIC_ridge <- n * log(RSS/n) + 2 * (df + 1) # Adiciona 1 para o intercepto
BIC_ridge <- n * log(RSS/n) + log(n) * (df + 1) # Adiciona 1 para o intercepto


cat("\n--- Performance do Modelo Ridge ---\n")
cat("R² =", R2_ridge, 
    "\nRMSE =", rmse_ridge, 
    "\nMAE =", mae_ridge, 
    "\nMAPE =" , mape_ridge,
    "\nAIC =", AIC_ridge,
    "\nBIC =", BIC_ridge
)

modelo_ridge_seq <- glmnet(x_train, 
                           y_train, 
                           alpha = 0, 
                           standardize = TRUE)

# Extrair os coeficientes do modelo Ridge
coefs <- as.matrix(modelo_ridge_seq$beta)
lambdas <- modelo_ridge_seq$lambda
df_coef <- as.data.frame(t(coefs))
df_coef$lambda <- log(lambdas)  # log(lambda)

# Transformar para formato longo (long format)
df_long <- melt(df_coef, id.vars = "lambda", 
                variable.name = "Variavel", 
                value.name = "Coeficiente")

#Plot com ggplot2
ggplot(df_long, aes(x = lambda, y = Coeficiente, color = Variavel)) +
  geom_line(size = 1.2, alpha = 0.9) +  # Linhas suaves e grossas
  geom_vline(xintercept = log(best_lambda_ridge), 
             color = "red", linetype = "dashed") +
  labs(
    title = "Ridge Trace Plot",
    subtitle = "Modelo Ridge Completo",
    x = expression(log(lambda)),
    y = "Coeficientes"
  ) +
  scale_color_brewer(palette = "Dark2") +  # Paleta de cores agradável
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 13, color = "gray40"),
    legend.title = element_blank(),
    legend.text = element_text(size = 10),
    legend.position = "right",
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "gray85")
  )
#------------------------------------------------------------------------------#




#------------------------------------------------------------------------------#
#-Modelo Ridge(-Area Basal)

#-Remover a variável AREA_BASAL
dados_sem_area <- dados %>% select(-AREA_BASAL)

modelo2_mlm <- lm(VOLUME ~ ., data = dados_sem_area)
summary(modelo2_mlm)
vif(modelo2_mlm)

# Dividir em treino/teste
set.seed(123)
train_index2 <- createDataPartition(dados_sem_area$VOLUME, p = 0.9, list = FALSE)
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
modelo_ridge2 <- glmnet(x_train2, 
                        y_train2, 
                        alpha = 0, 
                        lambda = best_lambda_ridge2, 
                        standardize = TRUE)

# Predição e avaliação
y_pred_ridge2 <- predict(modelo_ridge2, s = best_lambda_ridge2, newx = x_test2)

#------------------------------------------------------------------------------#
# Medidas Performance
R2_ridge2 <- cor(y_test2, y_pred_ridge2)^2
rmse_ridge2 <- sqrt(mean((y_test2 - y_pred_ridge2)^2))
mae_ridge2 <- mean(abs(y_test2 - y_pred_ridge2))
mape_ridge2 <- mean(abs((y_test2 - y_pred_ridge2) / y_test2)) * 100

# --- NOVAS MÉTRICAS: AIC e BIC ---
# Predição nos dados de treino para obter o RSS
y_pred_train_ridge2 <- predict(modelo_ridge2, s = best_lambda_ridge, newx = x_train2)
RSS <- sum((y_train2 - y_pred_train_ridge2)^2)

# Extrair os graus de liberdade efetivos do modelo
# O objeto glmnet já nos dá isso!
df <- modelo_ridge2$df[which(modelo_ridge2$lambda == best_lambda_ridge2)]

# Número de observações no treino
n <- nrow(dados_treino)

# Cálculo do AIC e BIC
AIC_ridge2 <- n * log(RSS/n) + 2 * (df + 1) # Adiciona 1 para o intercepto
BIC_ridge2 <- n * log(RSS/n) + log(n) * (df + 1) # Adiciona 1 para o intercepto


cat("\n--- Performance do Modelo Ridge(sem Area Basal) ---\n")
cat("R² =", R2_ridge2, 
    "\nRMSE =", rmse_ridge2, 
    "\nMAE =", mae_ridge2,
    "\nMAPE =" , mape_ridge2,
    "\nAIC =", AIC_ridge2,
    "\nBIC =", BIC_ridge2
    )
#------------------------------------------------------------------------------#
# Ajusta o modelo Ridge
modelo_ridge2 <- glmnet(x_train2, 
                        y_train2, 
                        alpha = 0, 
                        standardize = TRUE)

# Extrai os coeficientes
coefs <- as.matrix(modelo_ridge2$beta)
lambdas <- modelo_ridge2$lambda
df_coef <- as.data.frame(t(coefs))
df_coef$lambda <- log(lambdas)  # log(lambda)
df_long <- melt(df_coef, id.vars = "lambda", variable.name = "Variavel", value.name = "Coeficiente")

# Gráfico customizado com ggplot2
ggplot(df_long, aes(x = lambda, y = Coeficiente, color = Variavel)) +
  geom_line(size = 1.2, alpha = 0.9) +  # Linhas mais espessas e levemente transparentes
  geom_vline(xintercept = log(best_lambda_ridge2), 
             color = "red", linetype = "dashed") +
  labs(
    title = "Ridge Trace Plot",
    subtitle = "Modelo Ridge (Sem Área Basal)",
    x = expression(log(lambda)),
    y = "Coeficientes dos preditores"
  ) +
  scale_color_brewer(palette = "Dark2") +  # Paleta de cores suave
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 13, color = "gray40"),
    legend.title = element_blank(),
    legend.text = element_text(size = 11),
    legend.position = "right",
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "gray85")
  )



#------------------------------------------------------------------------------------------------#
# Gráfico de Validação Cruzada
#Mostra o erro médio de validação (MSE) para cada valor de lambda.

plot(cv_ridge)
plot(cv_ridge2)
#------------------------------------------------------------------------------------------------#



#------------------------------------------------------------------------------------------------#
#------------------------------ Modelo Ridge (sem DAP) ------------------------------------------#
# Remover DAP
dados_sem_dap <- dados %>% select(-DAP)

modelo3_mlm <- lm(VOLUME ~ ., data = dados_sem_dap)
summary(modelo3_mlm)
vif(modelo3_mlm)


# Dividir dados em treino e teste
set.seed(123)
train_idx_dap <- createDataPartition(dados_sem_dap$VOLUME, p = 0.9, list = FALSE)
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
y_pred_dap <- predict(modelo_ridge_dap, 
                      s = best_lambda_dap, 
                      newx = x_test_dap)

# Métricas de desempenho
R2_dap <- cor(y_test_dap, y_pred_dap)^2
rmse_dap <- sqrt(mean((y_test_dap - y_pred_dap)^2))
mae_dap <- mean(abs(y_test_dap - y_pred_dap))
mape_dap <- mean(abs((y_test_dap - y_pred_dap) / y_test_dap)) * 100


# ---- Cálculo aproximado de AIC e BIC ---- #
# n = número de observações de treino
# df = graus de liberdade efetivos (coeficientes não nulos + shrinkage)
# sigma² = variância residual
# logLik = log-verossimilhança gaussiana

n <- length(y_train_dap)
y_fitted <- predict(modelo_ridge_dap, s = best_lambda_dap, newx = x_train_dap)
rss <- sum((y_train_dap - y_fitted)^2)
sigma2 <- rss / n
logLik_ridge <- -n / 2 * (log(2 * pi) + log(sigma2) + 1)

# Graus de liberdade efetivos estimados
df_ridge <- modelo_ridge_dap$df

# AIC e BIC
aic_ridge <- -2 * logLik_ridge + 2 * df_ridge
bic_ridge <- -2 * logLik_ridge + log(n) * df_ridge


cat("\nModelo Ridge SEM DAP:\n")
cat("R² =", R2_dap, 
    "\nRMSE =", rmse_dap, 
    "\nMAE =", mae_dap,
    "\nMAPE =", mape_dap,
    "\nAIC  =", aic_ridge,
    "\nBIC  =", bic_ridge
    )


# Ajustar modelo ridge com vários lambdas
ridge_seq_dap <- glmnet(x_train_dap, y_train_dap, alpha = 0, standardize = TRUE)

# Extrair matriz de coeficientes e log(lambda)
coefs_matrix <- as.matrix(ridge_seq_dap$beta)
lambdas <- ridge_seq_dap$lambda
log_lambda <- log(lambdas)

# Transpor para que cada linha seja um valor de lambda
df_coefs <- as.data.frame(t(coefs_matrix))
df_coefs$log_lambda <- log_lambda

# Transformar para formato longo
df_long <- df_coefs %>%
  pivot_longer(
    cols = -log_lambda,
    names_to = "Variavel",
    values_to = "Coeficiente"
  )

# Plot com ggplot2
ggplot(df_long, aes(x = log_lambda, y = Coeficiente, color = Variavel)) +
  geom_line(size = 1) +
  geom_vline(xintercept = log(best_lambda_dap), linetype = "dashed", color = "red") +
  labs(
    title = "Ridge Trace Plot",
    subtitle = "Modelo Ridge Sem (DAP)",
    x = expression(log(lambda)),
    y = "Coeficiente") +
scale_color_brewer(palette = "Dark2") +  # Paleta de cores suave
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 13, color = "gray40"),
    legend.title = element_blank(),
    legend.text = element_text(size = 11),
    legend.position = "right",
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "gray85")
  )

#------------------------------------------------------------------------------------------------#



#------------------------------------------------------------------------------------------------#
#------------------ Modelo Ridge (sem DAP + Area basal) -----------------------------------------#
# Remover as Duas variáveis
dados_sem_dap_ab <- dados %>% select(-DAP, -AREA_BASAL)

# Dividir em treino e teste
set.seed(123)
train_idx_dap_ab <- createDataPartition(dados_sem_dap_ab$VOLUME, p = 0.9, list = FALSE)
dados_treino_dap_ab <- dados_sem_dap_ab[train_idx_dap_ab, ]
dados_teste_dap_ab  <- dados_sem_dap_ab[-train_idx_dap_ab, ]

x_train_dap_ab <- model.matrix(VOLUME ~ ., dados_treino_dap_ab)[, -1]
y_train_dap_ab <- dados_treino_dap_ab$VOLUME

x_test_dap_ab <- model.matrix(VOLUME ~ ., dados_teste_dap_ab)[, -1]
y_test_dap_ab <- dados_teste_dap_ab$VOLUME

# Ridge com validação cruzada
cv_ridge_dap_ab <- cv.glmnet(x_train_dap_ab, 
                             y_train_dap_ab, 
                             alpha = 0, 
                             standardize = TRUE)

best_lambda_dap_ab <- cv_ridge_dap_ab$lambda.min
cat("Melhor lambda (sem DAP e Área Basal):", best_lambda_dap_ab, "\n")

# Ajuste final do modelo
modelo_ridge_dap_ab <- glmnet(x_train_dap_ab, y_train_dap_ab,
                              alpha = 0,
                              lambda = best_lambda_dap_ab,
                              standardize = TRUE)

# Predição
y_pred_dap_ab <- predict(modelo_ridge_dap_ab, 
                         s = best_lambda_dap_ab, 
                         newx = x_test_dap_ab)

# Métricas
R2_dap_ab <- cor(y_test_dap_ab, y_pred_dap_ab)^2
rmse_dap_ab <- sqrt(mean((y_test_dap_ab - y_pred_dap_ab)^2))
mae_dap_ab <- mean(abs(y_test_dap_ab - y_pred_dap_ab))
mape_dap_ab <- mean(abs((y_test_dap_ab - y_pred_dap_ab) / y_test_dap_ab)) * 100

# ---- Cálculo Aproximado de AIC e BIC ----
n <- length(y_train_dap_ab)
y_fitted_ab <- predict(modelo_ridge_dap_ab, s = best_lambda_dap_ab, newx = x_train_dap_ab)
rss_ab <- sum((y_train_dap_ab - y_fitted_ab)^2)
sigma2_ab <- rss_ab / n
logLik_ridge_ab <- -n / 2 * (log(2 * pi) + log(sigma2_ab) + 1)

df_ridge_ab <- modelo_ridge_dap_ab$df
aic_ridge_ab <- -2 * logLik_ridge_ab + 2 * df_ridge_ab
bic_ridge_ab <- -2 * logLik_ridge_ab + log(n) * df_ridge_ab

# Exibir resultados
cat("\nModelo Ridge SEM DAP e SEM Área Basal:\n")
cat("R²:", R2_dap_ab, 
    "\nRMSE:", rmse_dap_ab, 
    "\nMAE:", mae_dap_ab,
    "\nMAPE:", mape_dap_ab,
    "\nAIC:", aic_ridge_ab,
    "\nBIC:", bic_ridge_ab
)



# Ajustar modelo Ridge com vários lambdas
ridge_seq_dap_ab <- glmnet(x_train_dap_ab, y_train_dap_ab, alpha = 0, standardize = TRUE)

# Extrair a matriz de coeficientes e os valores de lambda
coefs_matrix_ab <- as.matrix(ridge_seq_dap_ab$beta)
lambdas_ab <- ridge_seq_dap_ab$lambda
log_lambda_ab <- log(lambdas_ab)

# Transpor a matriz para que cada linha seja um valor de lambda
df_coefs_ab <- as.data.frame(t(coefs_matrix_ab))
df_coefs_ab$log_lambda <- log_lambda_ab

# Transformar para Formato longo (tidy)
df_long_ab <- df_coefs_ab %>%
  pivot_longer(
    cols = -log_lambda,
    names_to = "Variavel",
    values_to = "Coeficiente"
  )

# Plot com ggplot2
ggplot(df_long_ab, aes(x = log_lambda, y = Coeficiente, color = Variavel)) +
  geom_line(size = 1) +
  geom_vline(xintercept = log(best_lambda_dap_ab), linetype = "dashed", color = "red") +
  labs(
    title = "Ridge Trace Plot",
    subtitle = "Modelo Ridge Sem (DAP + Area Basal)",
    x = expression(log(lambda)),
    y = "Coeficiente"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 13, color = "gray40"),
    legend.title = element_blank(),
    legend.text = element_text(size = 11),
    legend.position = "right",
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "gray85")
  )


#------------------------------------------------------------------------------#


#--------------- Modelo Lasso Geral --------------------------------------------#
# Preparar Dados Completos
set.seed(123)
train_index_lasso <- createDataPartition(dados$VOLUME, p = 0.9, list = FALSE)
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
mape_lasso <- mean(abs((y_test_lasso - y_pred_lasso) / y_test_lasso)) * 100


# Variáveis selecionadas
coef_lasso <- coef(modelo_lasso)
variaveis_lasso <- rownames(coef_lasso)[coef_lasso[, 1] != 0 & rownames(coef_lasso) != "(Intercept)"]

cat("Lasso COM todas as variáveis:\n")
cat("R²:", R2_lasso, 
    "\nRMSE:", rmse_lasso, 
    "\nMAE:", mae_lasso,
    "\nMAPE =", mape_lasso
    )
cat("Variáveis selecionadas:", paste(variaveis_lasso, collapse = ", "), "\n")
















#--------------- Modelo Lasso (-Area Basal) --------------------------------------------#
# Remover AREA_BASAL
dados_lasso_ab <- dados %>% select(-AREA_BASAL)

# Dividir treino/teste
set.seed(123)
train_index_lasso_ab <- createDataPartition(dados_lasso_ab$VOLUME, p = 0.9, list = FALSE)
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
mape_lasso_ab <- mean(abs((y_test_lasso_ab - y_pred_lasso_ab) / y_test2)) * 100


# Variáveis selecionadas
coef_lasso_ab <- coef(modelo_lasso_ab)
variaveis_lasso_ab <- rownames(coef_lasso_ab)[coef_lasso_ab[, 1] != 0 & rownames(coef_lasso_ab) != "(Intercept)"]

# Resultado
cat("\nLasso SEM AREA_BASAL:\n")
cat("R²:", R2_lasso_ab, 
    "\nRMSE:", rmse_lasso_ab, 
    "\nMAE:", mae_lasso_ab,
    "\nMAPE =", mape_lasso_ab, "\n"
    )
cat("Variáveis selecionadas:", paste(variaveis_lasso_ab, collapse = ", "), "\n")


#--------------- Modelo Lasso (Sem DAP) --------------------------------------------#
# Remover DAP
dados_lasso_dap <- dados %>% select(-DAP)

# Dividir treino/teste
set.seed(123)
train_index_lasso_dap <- createDataPartition(dados_lasso_dap$VOLUME, p = 0.8, list = FALSE)
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
cat("\nLasso SEM DAP/AREA_BASAL:\n")
cat("R²:", R2_lasso_dap_ab, 
    "\nRMSE:", rmse_lasso_dap_ab, 
    "\nMAE:", mae_lasso_dap_ab, "\n")
cat("Variáveis selecionadas:", paste(variaveis_lasso_dap_ab, collapse = ", "), "\n")



#------------------- MODELO Elastic Net ----------------------------------------#
library(caret)
library(glmnet)


dados_en <- na.omit(dados) 
set.seed(123)
train_index_en <- createDataPartition(dados_en$VOLUME, p = 0.9, list = FALSE)
dados_treino_en <- dados_en[train_index_en, ]
dados_teste_en <- dados_en[-train_index_en, ]


# 3. Configurar o Controle de Treinamento (Validação Cruzada)
# Usaremos validação cruzada de 10 folds para encontrar os melhores parâmetros
train_control <- trainControl(method = "cv", number = 10)


# 4. Definir a Grade de Parâmetros para Testar
# O caret irá testar diferentes valores de 'alpha' e 'lambda'
# Vamos testar 11 valores de alpha, de 0 (Ridge) a 1 (Lasso)
# Para cada alpha, o caret testará automaticamente vários lambdas.
tune_grid <- expand.grid(
  alpha = seq(0, 1, length = 11),
  lambda = seq(0.001, 0.2, length = 20) # Um grid de lambdas para testar
)

# 5. Treinar o Modelo Elastic Net
# O caret vai fazer todo o trabalho de testar as combinações da grade
# e encontrar a melhor. Isso pode levar um minuto.
cat("Iniciando a otimização do Elastic Net...\n")
set.seed(123)
modelo_en <- train(
  VOLUME ~ .,
  data = dados_treino_en,
  method = "glmnet",         # Especifica o uso de Ridge, Lasso, Elastic Net
  trControl = train_control, # Aplica a validação cruzada
  tuneGrid = tune_grid,      # Fornece a grade de parâmetros para testar
  preProcess = c("center", "scale") # Boa prática: centralizar e escalar os dados
)

# 6. Ver os Melhores Parâmetros Encontrados
cat("\n--- Melhor Combinação Encontrada pelo Caret ---\n")
print(modelo_en$bestTune)


# 7. Avaliar o Modelo Final no Conjunto de Teste
y_pred_en <- predict(modelo_en, newdata = dados_teste_en)
R2_en <- cor(dados_teste_en$VOLUME, y_pred_en)^2
rmse_en <- sqrt(mean((dados_teste_en$VOLUME - y_pred_en)^2))
mae_en <- mean(abs(dados_teste_en$VOLUME - y_pred_en))
mape_en <- mean(abs((dados_teste_en$VOLUME - y_pred_en) / dados_teste_en$VOLUME)) * 100

# 8. Apresentar os Resultados Finais
cat("\n--- Performance do Modelo Elastic Net Otimizado ---\n")
cat("Melhor alpha:", modelo_en$bestTune$alpha, "\n")
cat("Melhor lambda:", modelo_en$bestTune$lambda, "\n")
cat("----------------------------------------\n")
cat("R² =", R2_en, "\n")
cat("RMSE =", rmse_en, "\n")
cat("MAE =", mae_en, "\n")
cat("MAPE =", mape_en, "%\n")




#------------------------------------------------------------------------------#




#------------------------------------------------------------------------------#
# Modelo Randon Forest

#-Carregar Pacotes 
library(randomForest)
library(caret)

dados_rf <- dados %>% select(-AREA_BASAL)

# 3. Divisão dos Dados em Treino e Teste (usando a mesma metodologia)
set.seed(123) # Para reprodutibilidade
train_index_rf <- createDataPartition(dados_rf$VOLUME, p = 0.9, list = FALSE)
dados_treino_rf <- dados_rf[train_index_rf, ]
dados_teste_rf <- dados_rf[-train_index_rf, ]

# Separar a variável resposta (y) e as preditoras (x) no teste
y_test_rf <- dados_teste_rf$VOLUME
x_test_rf <- dados_teste_rf %>% select(-VOLUME)


# 4. Treinamento do Modelo Random Forest
# ntree: Número de árvores na floresta (um valor entre 500-1000 é um bom começo).
# mtry: Número de variáveis testadas em cada "nó" da árvore. O padrão para regressão é (nº de variáveis / 3).
set.seed(123) # Para reprodutibilidade do modelo
modelo_rf <- randomForest(
  VOLUME ~ .,                  # Fórmula: prever VOLUME usando todas as outras variáveis
  data = dados_treino_rf,      # Usar os dados de treino
  ntree = 500,                 # Número de árvores
  importance = TRUE            # Guardar a importância das variáveis
)

# 5. Visualizar o resultado do modelo
print(modelo_rf)
# O resultado mostrará o "Mean of squared residuals" (MSE) e "% Var explained" (R²)
# calculados internamente nos dados de "out-of-bag" (uma forma de validação cruzada)


# 6. Fazer Previsões no Conjunto de Teste
y_pred_rf <- predict(modelo_rf, newdata = x_test_rf)


# 7. Avaliar a Performance do Modelo (comparando com o seu Ridge)
R2_rf <- cor(y_test_rf, y_pred_rf)^2
rmse_rf <- sqrt(mean((y_test_rf - y_pred_rf)^2))
mae_rf <- mean(abs(y_test_rf - y_pred_rf))
mape_rf <- mean(abs((y_test_rf - y_pred_rf) / y_test_rf)) * 100

# 8. Apresentar os resultados
cat("\n--- Performance do Modelo Random Forest ---\n")
cat("R² =", R2_rf, 
    "\nRMSE =", rmse_rf, 
    "\nMAE =", mae_rf, 
    "\nMAPE =", mape_rf
    )



# Ver a importância das variáveis
importancia <- importance(modelo_rf)
print(importancia)

# Criar um gráfico de importância
varImpPlot(modelo_rf, 
           main = "Importância das Variáveis - Random Forest",
           pch = 16, # Formato do ponto
           col = "blue") # Cor



# O gráfico é gerado diretamente a partir do objeto do modelo salvo
plot(modelo_rf, main = "Erro do Modelo vs. Número de Árvores")
legend("topright", 
       legend = "Erro OOB", 
       col = "blue", 
       lty = 3)



# Criar um dataframe com os valores reais e previstos
resultados_rf <- data.frame(
  Reais = y_test_rf,      # y_test_rf do seu script anterior
  Previstos = y_pred_rf   # y_pred_rf do seu script anterior
)

# Gerar o gráfico
ggplot(resultados_rf, aes(x = Reais, y = Previstos)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
  labs(
    title = "Random Forest: Valores Previstos vs. Reais",
    subtitle = "A linha vermelha representa a previsão perfeita",
    x = "Volume Real",
    y = "Volume Previsto"
  ) +
  theme_minimal() +
  coord_fixed() # Garante que a escala dos eixos seja a mesma (1:1)



# 1. Carregar Pacotes 
library(rpart)
library(rpart.plot)



# 2. Construir uma ÚNICA Árvore de Decisão
# Usamos o método "anova" para regressão (prever um número contínuo)
arvore_decisao <- rpart(
  VOLUME ~ .,
  data = dados_treino_rf,
  method = "anova"
)

# 3. Gerar o Gráfico da Árvore
# A função rpart.plot cria uma visualização muito mais informativa e bonita
rpart.plot(
  arvore_decisao,
  type = 4,                   # Estilo do gráfico (existem vários)
  extra = 101,                # Adiciona informações extras nos nós
  box.palette = "BuGn",       # Paleta de cores para os "nós"
  branch.lty = 3,             # Estilo da linha dos "galhos"
  shadow.col = "gray",        # Cor da sombra das caixas
  main = "Previsão de Volume"
)
#------------------------------------------------------------------------------#





#------------------------------------------------------------------------------#
# Modelo XGBoost(eXtreme Gradient Boosting)

#-Carregar Pacotes
library(xgboost)


dados_xgb <- dados %>% select(-AREA_BASAL)

# 2. Preparação dos Dados (usando a mesma divisão de antes)
set.seed(123)
train_index_xgb <- createDataPartition(dados_xgb$VOLUME, p = 0.9, list = FALSE)
dados_treino_xgb <- dados_xgb[train_index_xgb, ]
dados_teste_xgb <- dados_xgb[-train_index_xgb, ]

# 3. CONVERTER PARA O FORMATO XGBOOST (Passo Crucial!)
# XGBoost precisa de uma matriz numérica para as variáveis preditoras e um vetor para a resposta.

# Dados de treino
x_train_xgb <- data.matrix(dados_treino_xgb %>% select(-VOLUME))
y_train_xgb <- dados_treino_xgb$VOLUME
dtrain <- xgb.DMatrix(data = x_train_xgb, label = y_train_xgb)

# Dados de teste
x_test_xgb <- data.matrix(dados_teste_xgb %>% select(-VOLUME))
y_test_xgb <- dados_teste_xgb$VOLUME
dtest <- xgb.DMatrix(data = x_test_xgb, label = y_test_xgb)


# 4. Definir os Parâmetros do Modelo (Hyperparameters)
# Estes são pontos de partida razoáveis. Otimizá-los é a chave para a performance máxima.
params <- list(
  objective = "reg:squarederror", # Especifica que é um problema de regressão
  booster = "gbtree",             # Usa árvores como base
  eta = 0.1,                      # Taxa de aprendizado (learning rate)
  max_depth = 5,                  # Profundidade máxima de cada árvore
  colsample_bytree = 0.8,         # Porcentagem de variáveis usadas por árvore
  subsample = 0.8                 # Porcentagem de dados usados por árvore
)

# 5. Treinamento do Modelo XGBoost
set.seed(123)
modelo_xgb <- xgboost(
  params = params,
  data = dtrain,
  nrounds = 1000,                  # Número de "rodadas" de boosting (número de árvores)
  verbose = 0                     # 0 para não imprimir o progresso do treino
)

# 6. Fazer Previsões no Conjunto de Teste
y_pred_xgb <- predict(modelo_xgb, newdata = dtest)


# 7. Avaliar a Performance do Modelo
R2_xgb <- cor(y_test_xgb, y_pred_xgb)^2
rmse_xgb <- sqrt(mean((y_test_xgb - y_pred_xgb)^2))
mae_xgb <- mean(abs(y_test_xgb - y_pred_xgb))
mape_xgb <- mean(abs((y_test_xgb - y_pred_xgb) / y_test_xgb)) * 100

# 8. Apresentar os resultados
cat("\n--- Performance do Modelo XGBoost ---\n")
cat("R² =", R2_xgb,
    "\nRMSE =", rmse_xgb,
    "\nMAE =", mae_xgb, 
    "\nMAPE =", mape_xgb
    )

# Calcular a importância
importancia_xgb <- xgb.importance(model = modelo_xgb)
print(importancia_xgb)

# Gerar o gráfico de Importância
xgb.plot.importance(importance_matrix = importancia_xgb)
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#Ajuste de Hiperparâmetros

# 1. Definir a Grade de Hiperparâmetros para Testar
# Criaremos um data frame com as combinações que queremos avaliar
hyper_grid <- expand.grid(
  max_depth = c(2, 3, 4, 5, 6, 7, 8, 9, 10),      # Profundidade da árvore
  eta = c(0.01, 0.02, 0.05, 0.08, 0.1),           # Taxa de aprendizado
  min_rmse = 0,                       # Coluna para guardar o menor erro
  best_nrounds = 0                    # Coluna para guardar o nrounds ideal
)

# 2. Loop de Otimização com Validação Cruzada (Pode demorar alguns minutos)
cat("Iniciando a otimização do XGBoost...\n")
for (i in 1:nrow(hyper_grid)) {
  
  # Definir os parâmetros para esta iteração do loop
  params <- list(
    objective = "reg:squarederror",
    booster = "gbtree",
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    colsample_bytree = c(0.7, 0.8, 0.9),
    subsample = c(0.7, 0.8, 0.9)
  )
  
  # Usar xgb.cv (validação cruzada do xgboost)
  set.seed(123)
  xgb_cv_model <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 1000,                  # Testar até 1000 árvores
    nfold = 10,                      # Usar no mínimo 5 "folds" 
    verbose = 0,
    # Parada Antecipada: para o treino se o erro não melhorar em 50 rodadas
    early_stopping_rounds = 50       
  )
  
  # Guardar o melhor resultado desta combinação
  hyper_grid$min_rmse[i] <- min(xgb_cv_model$evaluation_log$test_rmse_mean)
  hyper_grid$best_nrounds[i] <- xgb_cv_model$best_iteration
  
  cat("Combinação", i, "de", nrow(hyper_grid), "concluída. RMSE:", hyper_grid$min_rmse[i], "\n")
}

# 3. Encontrar a Melhor Combinação de Parâmetros
# Ordenar a grade para ver os melhores resultados primeiro
hyper_grid <- hyper_grid[order(hyper_grid$min_rmse), ]
best_params <- hyper_grid[1, ]
cat("\n--- Melhor Combinação Encontrada ---\n")
print(best_params)


# 4. Treinar o Modelo Final e Otimizado
# Usamos os melhores parâmetros e o nrounds ideal encontrado
params_final <- list(
  objective = "reg:squarederror",
  booster = "gbtree",
  eta = best_params$eta,
  max_depth = best_params$max_depth,
  colsample_bytree = c(0.7, 0.8, 0.9),
  subsample = c(0.7, 0.8, 0.9)
)

set.seed(123)
modelo_xgb_otimizado <- xgboost(
  params = params_final,
  data = dtrain,
  nrounds = best_params$best_nrounds,
  verbose = 0
)

# 5. Avaliar o Modelo Otimizado
y_pred_xgb_otimizado <- predict(modelo_xgb_otimizado, newdata = dtest)
R2_xgb_otimizado <- cor(y_test_xgb, y_pred_xgb_otimizado)^2
rmse_xgb_otimizado <- sqrt(mean((y_test_xgb - y_pred_xgb_otimizado)^2))
mae_xgb_otimizado <- mean(abs(y_test_xgb - y_pred_xgb_otimizado))
mape_xgb_otimizado <- mean(abs((y_test_xgb - y_pred_xgb_otimizado) / y_test_xgb)) * 100

# 6. Apresentar os resultados finais
cat("\n--- Performance do Modelo XGBoost OTIMIZADO ---\n")
cat("R² =", R2_xgb_otimizado,
    "\nRMSE =", rmse_xgb_otimizado,
    "\nMAE =", mae_xgb_otimizado, 
    "\nMAPE =", mape_xgb_otimizado
)


#------------------------------------------------------------------------------#




