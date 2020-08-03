
getwd()

# ---------------------------------------------------------- scatter ----------
set.seed(123)
Ntrain <- 10000
Ntest <- 10000
Nfeature <- 2L

# explanatory variables
data <- matrix(runif(n = (Ntrain + Ntest) * Nfeature, min = 0, max = 1),
               nrow = Ntrain + Ntest,
               ncol = Nfeature) %>%
  tibble::as_tibble()

# response variable
label <- data %$%
  dplyr::case_when(V1 <= 0.1 ~ 1L,
                   0.1 < V1 & V2 + 4 * V1 <= 1.8 ~ 2L,
                   1.8 < V2 + 4 * V1 & V2 * V1 <= 0.3 ~ 3L,
                   0.3 < V2 * V1 & ((V2 - 0.7) ^ 2 + (V1 - 0.8) ^ 2 >= 0.2 ^ 2 ) ~ 4L,
                   TRUE ~ 5L)

# create data
data <- data %>%
  dplyr::mutate(no = dplyr::row_number(),
                cls = label,
                type = dplyr::if_else(no <= Ntrain, "train", "test")) %>%
  dplyr::select(type, cls, V1, V2)

save(data, file = "./data/scatter.rda")
