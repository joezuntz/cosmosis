#!/usr/bin/env Rscript

library(lattice)

data <- read.table("demo3_output.txt")
names(data) <- c("r", "loglike")

#convert to like from log-like and normalize
data$like <- exp(data$loglike)
norm <- data$like/max(data$like)
data$like <- data$like/norm

# Make the output directory if needed.
if (! file.exists("plots"))
   dir.create("plots")

#make plot
png("plots/demo3.png")
xyplot(like~r, data, xlab="Tensor ratio r", ylab="Likelihood", grid=TRUE, type="l")
invisible(dev.off())
