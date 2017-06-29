########################
########################
#         Q4           #
########################
########################

rm(list = ls())

# read training data
con = url("http://www.tau.ac.il/~saharon/StatsLearn2017/train_ratings_all.dat")
X.tr = read.table (con)
con = url("http://www.tau.ac.il/~saharon/StatsLearn2017/train_y_rating.dat")
y.tr = read.table (con)

X.tr.first.50.rows = X.tr[1:50,]
y.tr.first.50.rows = y.tr[1:50,]

train.first.50.rows = data.frame(X = X.tr.first.50.rows, y = y.tr.first.50.rows)

########################
########################
#     section g        #
########################
########################

########################
# for Lasso regression #
########################

library(lasso2)

norm.vals = seq (10e-5, 0.1,by=0.001)

mods_lasso = l1ce(y~.+1, data = train.first.50.rows, bound = norm.vals)
preds_lasso = sapply(mods_lasso, predict,newdata=train.first.50.rows)
resids_lasso = apply(preds_lasso, 2, "-", train.first.50.rows$y)
RSSs_lasso = apply(resids_lasso^2, 2, sum)
lambdas_lasso = as.numeric(sapply (mods_lasso, "[",5))

abs_coef_lasso <- abs(coefficients(mods_lasso))
lasso_norms <- rowSums(abs_coef_lasso) * lambdas_lasso

# plot the RSSs and the norms for each of the lambdas
par(mfrow=c(2,1))
plot(RSSs_lasso, main = "RSS plot - LASSO")
plot(lasso_norms, main = "Norms values plot - LASSO")

coefficients(mods_lasso)[60,]
########################
# for Ridge regression #
########################

library(MASS)
lambda.vals = exp(seq(-15,10,by=0.1))

mods_ridge = lm.ridge(y~.,data=train.first.50.rows, lambda=lambda.vals)
preds_ridge = as.matrix(train.first.50.rows[,1:99]) %*% t(coef(mods_ridge)[,-1]) +  rep(1,50) %o% coef(mods_ridge)[,1]
resids_ridge = matrix (data=train.first.50.rows$y, nrow=dim(preds_ridge)[1], ncol=dim(preds_ridge)[2], byrow=F)-preds_ridge
RSSs_ridge = apply (resids_ridge^2, 2, sum)

ridge_squared <- mods_ridge$coef^2
ridge_norms <- colSums(ridge_squared) * lambda.vals

# plot the RSSs and the norms for each of the lambdas
par(mfrow=c(2,1))
plot(RSSs_ridge, main = "RSS plot - Ridge")
plot(ridge_norms, main = "Norms values plot - Ridge")


########################
########################
#     section i        #
########################
########################

