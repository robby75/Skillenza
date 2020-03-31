library(bsts)     # load the bsts package
dat <- read.csv("trnBayes.csv")     # bring the initial.claims data into scope

ss <- AddLocalLinearTrend(list(), dat[,2])
ss <- AddSeasonal(ss, dat[,2], nseasons = 3)
model1 <- bsts(dat[,2],
               state.specification = ss,
               niter = 100)

plot(model1)
plot(model1, "components")  # plot(model1, "comp") works too!
plot(model1, "help")

pred1 <- predict(model1, horizon = 6)
print(pred1)
plot(pred1, plot.original = 156)
