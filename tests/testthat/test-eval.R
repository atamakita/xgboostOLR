context("evaluation")

test_that("correct logloss evaluation", {

})

test_that("correct hit ratio evaluation", {
  CRITERION <- c(-1, 1)
  NOTCHDIFF <- 1L

  # values for xgb.DMatrix
  SCORES <- c(-2, -1, 0.5, 1, 2)
  LABEL <- c(1L, 1L, 2L, 3L, 1L)

  # actual hit ratio
  HITRATIO_ACT <- sum(c(1, 1, 1, 1, 0)) / 5

  # create dtrain and class
  dtrain <- xgboost::xgb.DMatrix(data = as.matrix(seq_along(LABEL), ncol = 1),
                                 label = LABEL)
  cls <- xgboostOLR::createOLRlossClass(criterion = CRITERION,
                                        notchdiff = NOTCHDIFF)

  # predict labels
  result <- cls$eval_hitratio(preds = SCORES, dtrain = dtrain)
  expect_equal(object = result$metric,
               expected = "ratio_in1notch")
  expect_equal(object = result$value,
               expected = HITRATIO_ACT)
})
