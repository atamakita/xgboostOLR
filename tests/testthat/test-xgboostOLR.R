context("initialize")

test_that("criterion and notch_diff set", {
  CRITERION <- c(-1, 0, 1)
  NOTCHDIFF <- 1L

  CRITERION_ACT <- c(-Inf, -1, 0, 1, Inf)
  NOTCHDIFF_ACT <- 1L

  cls <- xgboostOLR::createOLRlossClass(criterion = CRITERION,
                                        notchdiff = NOTCHDIFF)
  # criterion
  expect_identical(object = cls$return_criterion(),
                   expected = CRITERION_ACT)
  # notchdiff_eval
  expect_equal(object = cls$return_notchdiff_eval(),
               expected = NOTCHDIFF_ACT)
})
