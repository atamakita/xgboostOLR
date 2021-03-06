
<!-- README.md is generated from README.Rmd. Please edit that file -->

# xgboostOLR

<!-- badges: start -->

<!-- badges: end -->

`xgboostOLR` package provides custom objective / evaluation function for
XGBoost.  
Objective function is negative logloss of ordered logit model.  
Evaluation functions are negative logloss and accuracy of labels.

## Installation

`xgboostOLR` isn’t available from CRAN.  
Please wait a moment.

## Example

``` r
# necessary
library(xgboostOLR)
library(xgboost)

# for vis
library(tidyverse)
```

Visualising dataset.

``` r
data("scatter", package = "xgboostOLR")

scatter %>% 
  dplyr::mutate(cls = as.character(cls)) %>% 
  ggplot(aes(x = V1, y = V2, col = cls)) + 
  geom_point(size = 0.5) + 
  facet_wrap(~ type) + 
  labs(title = "Distribution of labels on train/test dataset")
```

<img src="man/figures/README-unnamed-chunk-3-1.png" width="100%" />

Preparing R6 class for loss function for `xgboost`.

  - Initialize by `xgboostOLR::createOLRlossClass()`.
      - Set criterion between each labels by `criterion` args.  
      - Set margin of hit ratio by `notchdiff` args if you use
        `eval_hitratio` as `eval_metric`.

<!-- end list -->

``` r
# metric class
cls <- xgboostOLR::createOLRlossClass(criterion = c(2, 4, 6, 8),
                                      notchdiff = 1L)
```

Preparing train/test dataset.

``` r
# train 
idx_col <- colnames(scatter) %in% c("V1", "V2")
idx_row <- scatter$type == "train"
dtrain <- xgb.DMatrix(data = as.matrix(scatter[idx_row, idx_col]),
                      label = scatter[["cls"]][idx_row])
# test
idx_row <- scatter$type == "test"
dtest <- xgb.DMatrix(data = as.matrix(scatter[idx_row, idx_col]),
                      label = scatter[["cls"]][idx_row])
```

Construct model.

``` r
# train model
param <- list(max_depth = 5, eta = 0.01, nthread = 2, verbosity = 1L,
              objective=cls$obj_ordered_logit,
              eval_metric=cls$eval_hitratio)
bst <- xgb.train(params = param, 
                 data = dtrain, 
                 nrounds = 100L, 
                 watchlist = list(eval = dtest, train = dtrain), 
                 verbose = 0L)
```

Comparing predicted label.

  - `pred_class_criterion` generates predicted labels comparing
    criterion and margins.

<!-- end list -->

``` r
# vis pred labels
pred_score <- c(predict(bst, dtrain), predict(bst, dtest))
pred_label <- cls$pred_class_criterion(pred_score)
scatter %>% 
  dplyr::mutate(pred_score = pred_score, 
                pred_label = as.character(pred_label)) %>% 
  ggplot(aes(x = V1, y = V2, col = pred_label)) + 
  geom_point(size = 0.5) + 
  facet_wrap(~ type) + 
  labs(title = "Distribution of predicted labels on train/test dataset")
```

<img src="man/figures/README-unnamed-chunk-7-1.png" width="100%" />

-----
