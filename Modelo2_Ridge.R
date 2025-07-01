

library(glmnet)
library(tidyverse)
library(combinat)

# Função para calcular métricas, incluindo AIC, BIC e MAPE
calc_model_metrics <- function(model, x, y, lambda) {
  y_pred <- predict(model, newx = x, s = lambda)
  resid <- y - y_pred
  
  n <- length(y)
  p <- sum(coef(model, s = lambda) != 0) - 1  # excluir intercepto
  
  sigma2 <- sum(resid^2) / n
  logLik <- -n/2 * (log(2*pi) + log(sigma2) + 1)
  
  AIC <- -2 * logLik + 2 * p
  BIC <- -2 * logLik + log(n) * p
  MAPE <- mean(abs((y - y_pred) / y)) * 100
  
  return(list(AIC = AIC, BIC = BIC, MAPE = MAPE))
}

# Função para avaliar modelo Ridge dado subconjunto de variáveis
avaliar_ridge <- function(vars, dados, alpha = 0) {
  x <- model.matrix(as.formula(paste("VOLUME ~", paste(vars, collapse = "+"))), dados)[, -1]
  y <- dados$VOLUME
  
  cv <- cv.glmnet(x, y, alpha = alpha, standardize = TRUE)
  lambda_min <- cv$lambda.min
  
  y_pred <- predict(cv, newx = x, s = lambda_min)
  
  r2 <- cor(y, y_pred)^2
  rmse <- sqrt(mean((y - y_pred)^2))
  mae <- mean(abs(y - y_pred))
  
  # Calcular AIC, BIC e MAPE
  metrics <- calc_model_metrics(cv, x, y, lambda_min)
  
  return(list(
    vars = vars,
    r2 = r2,
    lambda = lambda_min,
    rmse = rmse,
    mae = mae,
    AIC = metrics$AIC,
    BIC = metrics$BIC,
    MAPE = metrics$MAPE
  ))
}

# Variáveis candidatas (todas exceto VOLUME)
variaveis <- colnames(dados)[!colnames(dados) %in% c("VOLUME")]

# Avaliar todos os subconjuntos de 2 até todas as variáveis
resultados <- list()
i <- 1
for (k in 2:length(variaveis)) {
  subgrupos <- combn(variaveis, k, simplify = FALSE)
  for (s in subgrupos) {
    r <- avaliar_ridge(s, dados)
    resultados[[i]] <- c(
      vars = paste(s, collapse = ", "),
      r2 = round(r$r2, 4),
      lambda = round(r$lambda, 6),
      rmse = round(r$rmse, 4),
      mae = round(r$mae, 4),
      AIC = round(r$AIC, 4),
      BIC = round(r$BIC, 4),
      MAPE = round(r$MAPE, 4),
      n_vars = length(s)
    )
    i <- i + 1
  }
}

# Criar data frame e converter colunas numéricas
resumo <- do.call(rbind, resultados) %>% as.data.frame(stringsAsFactors = FALSE)
resumo$r2 <- as.numeric(resumo$r2)
resumo$lambda <- as.numeric(resumo$lambda)
resumo$rmse <- as.numeric(resumo$rmse)
resumo$mae <- as.numeric(resumo$mae)
resumo$AIC <- as.numeric(resumo$AIC)
resumo$BIC <- as.numeric(resumo$BIC)
resumo$MAPE <- as.numeric(resumo$MAPE)
resumo$n_vars <- as.numeric(resumo$n_vars)

# Mostrar top 15 modelos com maior R² e menor número de variáveis
resumo %>%
  arrange(desc(r2), n_vars) %>%
  head(15)

