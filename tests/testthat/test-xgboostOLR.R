context("initialize")

test_that("initialization works", {
  cls <- createOLRlossClass(criterion = c(-1, 0, 1),
                            notchdiff = 1L)
})
