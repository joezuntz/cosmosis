#!/usr/bin/env Rscript

library(lattice)

data <- read.table("demo3_output.txt")
names(data) <- c("r", "loglike")

#convert to like from log-like and normalize
data$like <- exp(data$loglike)
data$like <- data$like/max(data$like)

#make plot
png("plots/demo3.png")
xyplot(like~r, data, xlab="Tensor ratio r", ylab="Likelihood", grid=TRUE, type="l")
invisible(dev.off())
