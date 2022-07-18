library(mice)
library(VIM)
library(Gmisc)
library(dplyr)

setwd("C:/Users/Jacky C/Desktop/SINGLE_TASK/Hydro")

meta_s <- read.csv("Meta_short.csv")
meta_s_1 <- meta_s %>% select(-s_no)
meta_s_1$DIS_group[meta_s_1$DIS_group == "CVA"] <- 1
meta_s_1$DIS_group[meta_s_1$DIS_group == "CTL"] <- 0
meta_s_2 <- sapply(meta_s_1, as.numeric)

meta_s_imp <- mice(meta_s_2, maxit = 20, m = 1, method = "pmm", print = TRUE, seed = 19)
meta_s_mice <- complete(meta_s_imp, 5))
write.csv(meta_s_mice, "Meta_long_MICE.csv", row.names=FALSE)


meta_l <- read.csv("Meta_long.csv")
meta_l_1 <- meta_l %>% select(-s_no, -File_Name_MP)
meta_l_1$DIS_group[meta_l_1$DIS_group == "CVA"] <- 1
meta_l_1$DIS_group[meta_l_1$DIS_group == "CTL"] <- 0
meta_l_2 <- sapply(meta_l_1, as.numeric)

meta_l_imp <- mice(meta_l_2, maxit = 1, m = 1, method = "pmm", print = TRUE, seed = 19)
meta_l_mice <- complete(meta_l_imp, 5))
write.csv(meta_l_mice, "Meta_long_MICE.csv", row.names=FALSE)