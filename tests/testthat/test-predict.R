context("predict")

test_that("correct predict labels by comparing criterion", {
  CRITERION <- c(0, 1)
  SCORES <- c(-500, 0, 0.5, 500, NA_real_)

  # actial predict lables
  PREDICT <- c(1L, 2L, 2L, 3L, NA_integer_)

  # create class
  cls <- xgboostOLR::createOLRlossClass(criterion = CRITERION)

  # predict labels
  expect_equal(object = cls$pred_class_criterion(preds = SCORES),
               expected = PREDICT)
})


test_that("correct predict labels at max probability", {
  CRITERION <- c(0, 10)
  SCORES <- c(-500, 5, 500, NA_real_)

  # actial predict lables
  PREDICT <- c(1L, 2L, 3L, NA_integer_)

  # create class
  cls <- xgboostOLR::createOLRlossClass(criterion = CRITERION)

  # predict labels
  expect_equal(object = cls$pred_class_maxprob(preds = SCORES),
               expected = PREDICT)
})
