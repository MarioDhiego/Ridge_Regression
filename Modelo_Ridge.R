
# Carregar bibliotecas
library(readxl)
library(car)
library(corrplot)
library(glmnet)
library(caret)
library(lmtest)

# 1. Leitura dos dados
dados <- read_excel("dados_pinus.xlsx")
head(dados)

# 2. Matriz de correlação
correl <- cor(dados[, -1])
corrplot(correl, method = "number", type = "upper", tl.col = "black")

# 3. Regressão linear múltipla inicial para diagnóstico
modelo_mlm <- lm(VOLUME ~ ., data = dados)
summary(modelo_mlm)
vif(modelo_mlm)  # Diagnóstico de multicolinearidade




# 4. Preparar dados para glmnet (sem intercepto)
x <- model.matrix(VOLUME ~ ., dados)[, -1]
y <- dados$VOLUME

# 5. Ridge regression com validação cruzada
cv_ridge <- cv.glmnet(x, y, alpha = 0, standardize = TRUE)

# 6. Melhor valor de lambda (k)
best_lambda <- cv_ridge$lambda.min
cat("Melhor lambda (k):", best_lambda, "\n")

# 7. Modelo final com melhor lambda
modelo_final <- glmnet(x, y, alpha = 0, 
                       lambda = best_lambda, 
                       standardize = TRUE)
coef(modelo_final)

# 8. Ridge Trace Plot (Coeficientes vs log(lambda))
modelo_ridge_seq <- glmnet(x, y, alpha = 0, standardize = TRUE)
plot(modelo_ridge_seq, xvar = "lambda", label = TRUE,
     main = "Ridge Trace Plot: Coeficientes vs log(Lambda)",
     xlab = "log(Lambda)", ylab = "Coeficientes")
abline(v = log(best_lambda), col = "red", lty = 2)

# 9. Avaliação do modelo
y_pred <- predict(modelo_final, s = best_lambda, newx = x)
R2 <- cor(y, y_pred)^2
cat("R² do modelo Ridge:", R2, "\n")

# 10. Diagnóstico dos resíduos
residuos <- y - y_pred

par(mfrow = c(2, 2))
qqnorm(residuos); qqline(residuos, col = "blue", lwd = 2)
plot(y_pred, residuos, main = "Resíduos vs Ajustados",
     xlab = "Valores Ajustados", ylab = "Resíduos"); abline(h = 0, col = "red", lty = 2)
hist(residuos, main = "Histograma dos Resíduos", col = "lightgray")
plot(residuos, type = "l", main = "Resíduos Sequenciais", ylab = "Resíduos", xlab = "Ordem")

# 11. Testes estatísticos
cat("Teste de Shapiro-Wilk (normalidade):\n")
print(shapiro.test(residuos))

cat("\nTeste de Durbin-Watson (autocorrelação):\n")
print(dwtest(VOLUME ~ ., data = dados))



#----------------- Sem Area Basal
# Criar novo conjunto sem AREA_BASAL

modelo_sem_area <- lm(VOLUME ~ . - AREA_BASAL, data = dados)
cat("\nVIF sem Área Basal:\n")
vif(modelo_sem_area)
coef(modelo_sem_area)

dados_sem_area <- dados %>% select(-AREA_BASAL)
x2 <- model.matrix(VOLUME ~ ., dados_sem_area)[, -1]
y2 <- dados_sem_area$VOLUME

# Ridge sem AREA_BASAL
set.seed(123)
cv_ridge2 <- cv.glmnet(x2, y2, alpha = 0, standardize = TRUE)
best_lambda2 <- cv_ridge2$lambda.min
modelo_ridge2 <- glmnet(x2, y2, alpha = 0, lambda = best_lambda2, standardize = TRUE)


# Avaliação
y2_pred <- predict(modelo_ridge2, newx = x2, s = best_lambda2)
R2_2 <- cor(y2, y2_pred)^2
cat("R² sem Área Basal:", R2_2, "\n")


# R² com e sem área basal
cat("R² com Área Basal:", R2, "\n")
cat("R² sem Área Basal:", R2_2, "\n")
cat("Diferença (ΔR²):", R2 - R2_2, "\n")





