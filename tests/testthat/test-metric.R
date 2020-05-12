context("custom metric")

test_that("correct grad and hess", {
  CRITERION <- c(0, 1)
  SCORES <- c(-1, 0.5, 1, NA_real_)
  LABELS <- c(1L, 2L, 3L, 4L)

  dtrain <- xgboost::xgb.DMatrix(data = as.matrix(1:4),
                                 label = LABELS)

  # actual grad and hess
  logit_cum <- function(x, t) 1 / (1 + exp(x - t))
  logit_cum_grad <- function(x, t) -logit_cum(x, t) * (1 - logit_cum(x, t))
  PREDICT <- list(grad = c(1 - logit_cum(SCORES[1], -Inf) - logit_cum(SCORES[1], 0),
                           1 - logit_cum(SCORES[2], 0) - logit_cum(SCORES[2], 1),
                           1 - logit_cum(SCORES[3], 1) - logit_cum(SCORES[3], Inf),
                           NA_real_),
                  hess = c(-logit_cum_grad(SCORES[1], -Inf) - logit_cum_grad(SCORES[1], 0),
                           -logit_cum_grad(SCORES[2], 0) - logit_cum_grad(SCORES[2], 1),
                           -logit_cum_grad(SCORES[3], 1) - logit_cum_grad(SCORES[3], Inf),
                           NA_real_))

  # create class
  cls <- xgboostOLR::createOLRlossClass(criterion = CRITERION)

  # compare grad and hess
  expect_identical(object = cls$obj_ordered_logit(preds = SCORES, dtrain = dtrain),
                   expected = PREDICT)
})
