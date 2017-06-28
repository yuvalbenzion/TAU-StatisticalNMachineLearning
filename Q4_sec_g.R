

# read training data
con = url("http://www.tau.ac.il/~saharon/StatsLearn2017/train_ratings_all.dat")
X.tr = read.table (con)
con = url("http://www.tau.ac.il/~saharon/StatsLearn2017/train_y_rating.dat")
y.tr = read.table (con)

X.tr.first.50.rows = X.tr[1:50,]
y.tr.first.50.rows = y.tr[1:50,]

train.first.50.rows = data.frame(X = X.tr.first.50.rows, y = y.tr.first.50.rows)

########################
# for Lasso regression #
########################

library(lasso2)

norm.vals = seq (10e-5, 1,by=0.01)

mods_lasso = l1ce(y~., data = train.first.50.rows, bound = norm.vals)
preds_lasso = sapply (mods_lasso, predict,newdata=train.first.50.rows) # example of lapply!
resids_lasso = apply (preds_lasso, 2, "-", train.first.50.rows$y)
RSSs_lasso = apply(resids_lasso^2, 2, sum)

plot(sqrt(RSSs_lasso/50), ylab="RMSE",xlab="s",ylim=c(0.8,1))
plot(RSSs_lasso)

norm_lasso = sapply(mods_lasso, FUN = function(x){sum(x$relative.bound * abs(x$coefficients))})
plot(norm_lasso)


########################
# for Ridge regression #
########################
library(MASS)
lambda.vals = exp(seq(-15,10,by=0.1))

mods_ridge = lm.ridge(y~.,data=train.first.50.rows, lambda=lambda.vals)
preds_ridge = as.matrix(train.first.50.rows[,1:99]) %*% t(coef(mods_ridge)[,-1]) +  rep(1,50) %o% coef(mods_ridge)[,1]
resids = matrix (data=train.first.50.rows$y, nrow=dim(preds)[1], ncol=dim(preds)[2], byrow=F)-preds
RSSs = apply (resids^2, 2, sum)

# continue from here!!!!!!!!!!!!!!!!!!!!!!!
norm_ridge = apply(mods_ridge$coef,MARGIN = 2, FUN = function(x){sum(x)})



############# test #~~~~~~~~~~~~~~~~
mods_ridge$lambda * sum(mods_ridge$coef^2)
apply(mods_ridge$coef)

