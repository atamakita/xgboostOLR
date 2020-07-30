#' Create custom Objective/Evaluate Class for XGBoost
#'
#' @param criterion A double vector, criterion for Ordered Logit Model
#' @param notchdiff An integer, notch width when evaluate hit ratio
#' @return XGBMetricForOrderedLogit class
#' @export
createOLRlossClass <- function(criterion, notchdiff = 0L) {
  cls <- XGBMetricForOrderedLogit$new()
  cls$set_criterion(criterion)$set_notchdiff_eval(notchdiff)
  return(cls)
}

#' R6 Class defining custom Objective/Evaluate for XGBoost
#'
#' @title Custom Objective/Evaluate Functions Class for XGBoost
#' @docType class
#' @export
XGBMetricForOrderedLogit <-
  R6::R6Class("XGBMetricForOrderedLogit",
          public = list(
            #' @description
            #' Set Criterions between each labels for Ordered Logit Model
            #'
            #' @param x A double vector
            #' @note Add -Inf/Inf both side of input x
            #' @note x is sorted ascending
            #'   \code{criterion[1] = -Inf},
            #'   \code{criterion[2] = x[1]},
            #'   ...
            #'   \code{criterion[nClass] = x[nClass - 1]},
            #'   \code{criterion[nClass + 1] = Inf}
            set_criterion = function(x) {
              x <- append(-Inf, sort(x))
              x <- append(x, Inf)

              if (length(x) != length(unique(x))) {
                stop("detect duplicated criterion")
              }

              private$criterion <- x
              invisible(self)
            },

            #' @description
            #' Return criterions between each labels for Ordered Logit Model
            #'
            #' @return Criterion set in this class
            return_criterion = function() {
              return(private$criterion)
            },

            #' @description
            #' Set notch width when evaluating hit ratio by eval_hitratio()
            #'
            #' @param x An integer, notch width when evaluate hit ratio
            #' @note x = 1 -> hit ratio is calculated by {|act - est| < x} / length(act)
            set_notchdiff_eval = function(x) {
              private$notchdiff_eval <- x
              invisible(self)
            },

            #' @description
            #' Return notch width
            #'
            #' @return private$notchdiff_eval
            return_notchdiff_eval = function() {
              return(private$notchdiff_eval)
            },

            #' @description
            #' Function to predict labels comparing preds and criterions (not maximum probability)
            #'
            #' @param preds A double vector, margin score from xgboost before logistic transformation
            #' @return Integer label, if \code{[-Inf, criterion[1])} label is 1, else if \code{[criterion[1], criterion[2])}, label is 2, ...
            pred_class_criterion = function(preds) {
              return(as.integer(cut(x = preds,
                                    breaks = private$criterion,
                                    right = FALSE,
                                    include.lowest = TRUE,
                                    ordered_result = TRUE)))
            },

            #' @description
            #' Function to predict labels at maximum probability
            #'
            #' @param preds A double vector, margin score from xgboost before logistic transformation
            #' @return Integer label at max probability
            pred_class_maxprob = function(preds) {
              lst_probs <- list()

              for(i in 1:(length(private$criterion) - 1)) {
                f_m1 <- private$calc_cum(x = preds,
                                         tau = private$criterion[i])
                f <- private$calc_cum(x = preds,
                                      tau = private$criterion[i + 1])

                colnm <- paste0("V", i)
                lst_probs[[colnm]] <- f - f_m1
              }

              return(max.col(as.data.frame(lst_probs)))
            },

            #' @description
            #' Custom Objective Function of Ordered Logit Model
            #' https://github.com/dmlc/xgboost/blob/master/R-package/demo/custom_objective.R
            #'
            #' @param preds A double vector, margin score at t-1
            #' @param dtrain A xgb.DMatrix created by xgboost::xgb.DMatrix
            #' @return A named list(both double vector).
            #'   $grad gradient of loss function.
            #'   $hess hessian of loss function.
            #' @note grad, hess were referred to URL below.
            #'   https://www.slideshare.net/TakujiTahara/201200229-lt-dsb2019-ordered-logit-model-for-qwk-tawara#23
            obj_ordered_logit = function(preds, dtrain) {
              labels <- xgboost::getinfo(dtrain, "label")

              f_m1 <- private$calc_cum(x = preds,
                                       tau = private$criterion[labels])
              f <- private$calc_cum(x = preds,
                                    tau = private$criterion[labels + 1])

              grad <- 1 - (f_m1 + f)
              hess <- f_m1 * (1 - f_m1) + f * (1 - f)
              return(list(grad = grad, hess = hess))
            },

            #' @description
            #' Custom Evaluate Function of Ordered Logit Model (logloss)
            #' https://github.com/dmlc/xgboost/blob/master/R-package/demo/custom_objective.R
            #'
            #' @param preds A double vector, margin score
            #' @param dtrain A xgb.DMatrix created by xgboost::xgb.DMatrix
            #' @return A named list(both double vector).
            #'   $metric name of metric.
            #'   $value value of metric.
            eval_logloss = function(preds, dtrain) {
              labels <- xgboost::getinfo(dtrain, "label")

              f_m1 <- private$calc_cum(x = preds,
                                       tau = private$criterion[labels])
              f <- private$calc_cum(x = preds,
                                    tau = private$criterion[labels + 1])

              return(list(metric = "logloss_ordered_logit",
                          value = mean(-log(f - f_m1 + 1e-15))))
            },

            #' @description
            #' Custom Evaluate Function of Ordered Logit Model (hit ratio)
            #' https://github.com/dmlc/xgboost/blob/master/R-package/demo/custom_objective.R
            #'
            #' @param preds A double vector, margin score
            #' @param dtrain A xgb.DMatrix created by xgboost::xgb.DMatrix
            #' @return A named list(both double vector).
            #'   $metric name of metric.
            #'   $value value of metric.
            #' @note self$notchdiff_eval is set by set_notchdiff_eval
            eval_hitratio = function(preds, dtrain) {
              act <- as.integer(xgboost::getinfo(dtrain, "label"))
              est <- self$pred_class_maxprob(preds)

              return(list(metric = paste0("ratio_in", private$notchdiff_eval, "notch"),
                          value = sum(abs(act - est) <= private$notchdiff_eval) / length(act)))
            }
            ),
          private = list(
            #' @field Criterion values between each labels for Ordered Logit Model
            criterion = NULL,

            #' @field notchdiff_eval width of notch when evaluating hit ratio
            notchdiff_eval = 0,

            #' @description
            #' Cumulative function of Logistic distribution
            #'
            #' @param x A double vector
            #' @param tau dA ouble vector, criterion of Ordered Logit Model
            calc_cum = function(x, tau) {
              return(1 / (1 + exp(x - tau)))
            }
            )
          )

# # data
# set.seed(123)
# Ntrain <- 10000
# Ntest <- 10000
# prob_class <- c(0.1, 0.2, 0.3, 0.4)
# label <- as.vector(seq_along(prob_class) %*% rmultinom(Ntrain+Ntest, 1, prob_class))
# data <- matrix(rnorm(length(label), mean = label, sd = 1))
# dtrain <- xgb.DMatrix(data = head(data, Ntrain),
#                       label = head(label, Ntrain))
# dtest <- xgb.DMatrix(data = tail(data, Ntest),
#                      label = tail(label, Ntest))
#
# tibble(label = as.character(label),
#        val = data[,1]) %>%
#   ggplot(aes(x = val, y = ..count.., fill = label)) +
#   geom_histogram(position = "identity", alpha = 0.7)
#
# # class
# cls <- XGBMetricForOrderedLogit$new()
# cls$set_criterion(c(2, 4, 6))$set_notchdiff_eval(1)
#
# # train
# watchlist <- list(eval = dtest, train = dtrain)
# num_round <- 100
# param <- list(max_depth=2, eta=0.1, nthread = 2, verbosity=0,
#               objective=cls$obj_ordered_logit,
#               eval_metric=cls$eval_hitratio)
# bst <- xgb.train(param, dtrain, num_round, watchlist)
#
# tibble(no = as.integer(seq_along(label)),
#        label = as.character(label),
#        act = data[,1],
#        pred = c(predict(bst, dtrain), predict(bst, dtest)),
#        type = c(rep("train", Ntrain), rep("test", Ntest))) %>%
#   tidyr::gather(key, value, -no, -label, -type) %>%
#   ggplot(aes(x = value, y = ..density.., fill = label)) +
#   geom_histogram(position = "identity", alpha = 0.7, col = "black", bins = 100) +
#   facet_wrap(~ type + key)
#
# tibble(no = as.integer(seq_along(label)),
#        label = as.character(label),
#        act = data[,1],
#        pred = c(predict(bst, dtrain), predict(bst, dtest)),
#        type = c(rep("train", Ntrain), rep("test", Ntest))) %>%
#   ggplot(aes(x = act, y = pred, fill = label)) +
#   geom_point() +
#   facet_grid(type ~ label)
