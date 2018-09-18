
# ================================================================
# 1. 泊松分布
# ================================================================
# density function 概率函数
dpois(x, lambda, log = FALSE) # lambda 表示均值, log = TRUE 表示输出log(f(x))
# distribution function 分布函数
ppois(q, lambda, lower.tail = TRUE, log.p = FALSE)
# quantile function 分布函数的逆函数（分位数函数）
qpois(p, lambda, lower.tail = TRUE, log.p = FALSE)
# 泊松分布的随机数 - 模拟
rpois(n, lambda = 10) # 模拟 n 个服从期望为10的泊松分布
# -----------------------------------------------------------------
# 画图- 不同lambda的泊松分布的概率函数图
# -----------------------------------------------------------------
lambda.po <- c(1, 2, 5, 10) # lambda 取值为 1,2,5,10
x0 <- seq(0, 25)            # x 取值为 0-25 的整数
f1 <- dpois(x0, lambda = lambda.po[1], log = FALSE) 
f2 <- dpois(x0, lambda = lambda.po[2], log = FALSE) 
f3 <- dpois(x0, lambda = lambda.po[3], log = FALSE) 
f4 <- dpois(x0, lambda = lambda.po[4], log = FALSE) 
par(mfrow = c(1,1)) 
matplot(x0, cbind(f1, f2, f3, f4), type = 'l', lty = 1:4, lwd = 2)

par(mfrow = c(2,2)) 
for (i in 1:4){
  fpo <- dpois(x0, lambda = lambda.po[i], log = FALSE)
  barplot(fpo, main = paste0('lambda = ', lambda.po[i]), 
          col = 'gray', beside = TRUE, names.arg = 0:25, legend.text = TRUE)
  lines(x0, fpo, col = 2, lwd = 2)
}


## 索赔次数等于3的概率
dpois(3, lambda = 2)
## 索赔次数小于等于4的概率为
ppois(4, lambda = 2)
## 索赔次数大于等于3小于等于5的概率
ppois(5, 2) - ppois(2, 2)

## 模拟20个索赔次数观察值
set.seed(111) # 设定随机种子
sim = rpois(n = 20, lambda = 2) # 模拟生成20个服从lambda=2的泊松分布的随机数量
sim
## 对模拟的索赔次数进行列表
table(sim)


# ================================================================
# 2. 负二项分布
# ================================================================
# 参数 r = size
# 参数 beta = q
dnbinom(x, size, prob, mu, log = FALSE)
pnbinom(q, size, prob, mu, lower.tail = TRUE, log.p = FALSE)
qnbinom(p, size, prob, mu, lower.tail = TRUE, log.p = FALSE)
rnbinom(n, size, prob, mu)

# # 画图 
# r0 <- c(1, 2, 3, 2, 2, 2)
# beta0 <- c(0.3, 0.3, 0.3, 0.2, 0.3, 0.4)
# x0 <- seq(0, 20)
# ylim0 <- list(c(0,0.3),
#               c(0,0.12),
#               c(0,0.1),
#               c(0,0.2),
#               c(0,0.2),
#               c(0,0.2)
#               )
# 
# par(mfrow = c(2, 3) )
# for (i in 1:length(beta0)){
#   fpo <- dnbinom(x0, size = r0[i], prob = beta0[i], log = FALSE)
#   barplot(fpo, main = paste0('r = ', r0[i], ',  ','beta = ', beta0[i]), names.arg = x0, ylim = ylim0[[i]]
#   )
# }
# -----------------------------------------------------------------
# 例： 零截断负二项分布的计算
# -----------------------------------------------------------------
##负二项分布的概率
x = 0:10
p = dnbinom(x, 4, 0.7)
round(p,3)
##零截断负二项分布的概率
p0 = p[1]   
##零点的概率
pt1 = p[2:11]/(1-p0)  
##其它点上的概率
pt = c(0, pt1)
round(pt, 3)
##绘图比较负二项和零截断负二项的概率
com = rbind(负二项 = p, 零截断负二项 = pt)
par(mfrow = c(1, 1))
barplot(com, beside = TRUE, names.arg = 0:10,legend.text = TRUE)

# -----------------------------------------------------------------
#例： 零调整负二项分布的计算
# -----------------------------------------------------------------
##负二项分布的概率
x = 0:10
p = dnbinom(x, 4, 0.7)
round(p,3)
##零调整负二项分布的概率
p0 = 0.3  
##调整零点的概率
pm = (1 - p0)*p[2:11]/(1 - p[1])  
##其它点上的概率
pm = c(p0, pm)
round(pm, 3)
##绘图
com = rbind(负二项 = p, 零调整负二项 = pm)
barplot(com, beside = TRUE, names.arg = 0:10, legend.text = TRUE)




# ================================================================
# 二项分布
# ================================================================
# 参数 m = size
# 参数 prob = q
dbinom(x, size, prob, log = FALSE)
pbinom(q, size, prob, lower.tail = TRUE, log.p = FALSE)
qbinom(p, size, prob, lower.tail = TRUE, log.p = FALSE)
rbinom(n, size, prob)



# 画图 
m0 <- c(1, 5, 10, 10, 10, 10)
q0 <- c(0.3, 0.3, 0.3, 0.1, 0.2, 0.3)
x0 <- seq(0, 10)

par(mfrow = c(2, 3) )
for (i in 1:length(m0)){
  fpo <- dbinom(x0, size = m0[i], prob = q0[i], log = FALSE)
  barplot(fpo, 
          main = paste0('m = ', m0[i], ',  ','q = ', q0[i]),
          names.arg = x0,
  )
}



# ================================================================
# 几何分布
# ================================================================
# 画图 
r0 <- 1
beta0 <- c(0.1, 0.2, 0.3)
x0 <- seq(0, 20)


par(mfrow = c(1, 3) )
for (i in 1:length(beta0)){
  fpo <- dnbinom(x0, size = r0, prob = beta0[i], log = FALSE)
  barplot(fpo, 
          main = paste0('beta = ', beta0[i]),
          names.arg = x0
  )
}


# ==========================================
# 指数分布
# ===========================================
theta <- c(0.5, 1, 2)
x0 <- seq(0.001, 10, length.out = 100)
par(mfrow = c(1, 1) )
fexp <- dexp(x0, rate = theta[1], log = FALSE)
plot(x0, fexp, type = 'l',ylim = c(0,0.5), main = '')

for (i in 1:3){
  fexp <- dexp(x0, rate = theta[i], log = FALSE)
  lines(x0, fexp, type = 'l', lty = i, ylim = c(0,0.5))
}
legend(x = 6, y = 0.4, legend = c('theta = 0.5', 'theta = 1', 'theta = 2'),
       lty = c(1,2,3), bty = "n"
)



# ==========================================
# 伽马分布
# ===========================================
par(mfrow = c(1, 2) )

# 固定形状参数
alpha <- 2                # 形状参数
theta <- c(0.5, 1, 2)     # 尺度参数/比率参数
x0 <- seq(0.001, 15, length.out = 100)

fga <- dgamma(x0,  shape = alpha, rate = theta[1])
plot(x0, fga, type = 'l',ylim = c(0,0.8), main = '')

for (i in 1:3){
  fga <- dgamma(x0,  shape = alpha, rate = theta[i])
  lines(x0, fga, type = 'l', lty = i, ylim = c(0,0.8))
}
legend(x = 6, y = 0.6, 
       legend = c('alpha = 2, theta = 0.5', 
                  'alpha = 2, theta = 1', 
                  'alpha = 2, theta = 2'),
       lty = c(1,2,3), bty = "n"
)


# 固定尺度参数
alpha <- c(1,2,3)
theta <- 0.5
x0 <- seq(0.001, 15, length.out = 100)

fga <- dgamma(x0,  shape = alpha[1], rate = theta)
plot(x0, fga, type = 'l',ylim = c(0,0.5), main = '')

for (i in 1:3){
  fga <- dgamma(x0,  shape = alpha[i], rate = theta)
  lines(x0, fga, type = 'l', lty = i, ylim = c(0,0.5))
}
legend(x = 6, y = 0.3, 
       legend = c('alpha = 1, theta = 0.5', 
                  'alpha = 2, theta = 0.5', 
                  'alpha = 3, theta = 0.5'),
       lty = c(1,2,3), bty = "n"
)

# ==========================================
# 逆高斯分布
# ===========================================
# 定义逆高斯分布的密度函数 fx
dinvguass <- function(y, alpha, theta, ... ){
  fx <- alpha/(2*pi*theta*y^3)^0.5*exp(-(alpha - theta*y)^2/(2*theta*y))
  return(fx)
}


par(mfrow = c(1, 2) )
# 固定 alpha 
alpha <- 2                
theta <- c(0.5, 1, 2)     
x0 <- seq(0.001, 15, length.out = 100)

fig <- dinvguass(x0,  alpha = alpha, theta = theta[1])
plot(x0, fig, type = 'l',ylim = c(0,1), main = '')

for (i in 1:3){
  fig <- dinvguass(x0,  alpha = alpha, theta = theta[i])
  lines(x0, fig, type = 'l', lty = i, ylim = c(0, 1))
}
legend(x = 6, y = 0.8, 
       legend = c('alpha = 2, theta = 0.5', 
                  'alpha = 2, theta = 1', 
                  'alpha = 2, theta = 2'),
       lty = c(1,2,3), bty = "n"
)


# 固定 theta
alpha <- c(1,2,3)
theta <- 0.5
x0 <- seq(0.001, 15, length.out = 100)

fig <- dinvguass(x0,  alpha = alpha[1], theta = theta)
plot(x0, fig, type = 'l',ylim = c(0,0.6), main = '')

for (i in 1:3){
  fig <- dinvguass(x0,  alpha = alpha[i], theta = theta)
  lines(x0, fig, type = 'l', lty = i, ylim = c(0,0.6))
}
legend(x = 6, y = 0.5, 
       legend = c('alpha = 1, theta = 0.5', 
                  'alpha = 2, theta = 0.5', 
                  'alpha = 3, theta = 0.5'),
       lty = c(1,2,3), bty = "n"
)

# ==========================================
# 对数正态分布
# ===========================================
# 密度函数
dlnorm

par(mfrow = c(1, 2) )
# 固定 mu 
mu <- 2                
sigma <- c(0.5, 1, 2)     
x0 <- seq(0.001, 15, length.out = 100)

f <- dlnorm(x0,  meanlog = mu, sdlog = sigma[1])
plot(x0, f, type = 'l',ylim = c(0, 0.3), main = '')

for (i in 1:3){
  f <- dlnorm(x0,  meanlog = mu, sdlog = sigma[i])
  lines(x0, f, type = 'l', lty = i, ylim = c(0, 0.3))
}
legend("topright", 
       legend = c('mu = 2, sigma = 0.5', 
                  'mu = 2, sigma = 1', 
                  'mu = 2, sigma = 2'),
       lty = c(1,2,3), bty = "n"
)


# 固定 sigma
mu <- c(1,2,3)
sigma <- 1
x0 <- seq(0.001, 15, length.out = 100)

f <-  dlnorm(x0,  meanlog = mu[1], sdlog = sigma)
plot(x0, f, type = 'l',ylim = c(0,0.3), main = '')

for (i in 1:3){
  f <- dlnorm(x0,  meanlog = mu[i], sdlog = sigma)
  lines(x0, f, type = 'l', lty = i, ylim = c(0, 0.3))
}
legend("topright", 
       legend = c('mu = 1, sigma = 1', 
                  'mu = 2, sigma = 1', 
                  'mu = 3, sigma = 1'),
       lty = c(1,2,3), bty = "n"
)


# ==========================================
# 帕累托分布
# ===========================================
dpareto <- function(y, alpha, theta){
  f <- alpha*(theta^alpha)/(y+theta)^(alpha+1)
  return(f)
}



par(mfrow = c(1, 2) )
# 固定 alpha 
alpha <- 2                
theta <- c(0.5, 1, 2)     
x0 <- seq(0.001, 15, length.out = 100)

f <- dpareto(x0,  alpha = alpha, theta = theta[1])
plot(x0, f, type = 'l',ylim = c(0, 0.3), main = '')

for (i in 1:3){
  f <- dpareto(x0,  alpha = alpha, theta = theta[i])
  lines(x0, f, type = 'l', lty = i, ylim = c(0, 0.3))
}
legend("topright", 
       legend = c('alpha = 2, theta = 0.5', 
                  'alpha = 2, theta = 1', 
                  'alpha = 2, theta = 2'),
       lty = c(1,2,3), bty = "n"
)


# 固定 theta
alpha <- c(1,2,3)
theta <- 0.5
x0 <- seq(0.001, 15, length.out = 100)

f <- dpareto(x0,  alpha = alpha[1], theta = theta)
plot(x0, f, type = 'l',ylim = c(0,0.5), main = '')

for (i in 1:3){
  f <- dpareto(x0,  alpha = alpha[i], theta = theta)
  lines(x0, f, type = 'l', lty = i, ylim = c(0,0.5))
}
legend("topright", 
       legend = c('alpha = 1, theta = 0.5', 
                  'alpha = 2, theta = 0.5', 
                  'alpha = 3, theta = 0.5'),
       lty = c(1,2,3), bty = "n"
)



# ==========================================
# 威布尔分布
# ===========================================

# 定义密度函数
dwei <- function(y, alpha, theta){
  f <- alpha*theta*y^(theta-1)*exp(-alpha*y^theta)
  return(f)
}

par(mfrow = c(1, 2) )
# 固定 alpha 
alpha <- 1                
theta <- c(0.5, 1, 2)     
x0 <- seq(0.001, 5, length.out = 100)

f <- dwei(x0,  alpha = alpha, theta = theta[1])
plot(x0, f, type = 'l',ylim = c(0, 1.5), main = '')

for (i in 1:3){
  f <- dwei(x0,  alpha = alpha, theta = theta[i])
  lines(x0, f, type = 'l', lty = i, ylim = c(0, 1.5))
}
legend("topright", 
       legend = c('alpha = 1, theta = 0.5', 
                  'alpha = 1, theta = 1', 
                  'alpha = 1, theta = 2'),
       lty = c(1,2,3), bty = "n"
)


# 固定 theta
alpha <- c(1,2,3)
theta <- 0.5
x0 <- seq(0.001, 5, length.out = 100)

f <- dwei(x0,  alpha = alpha[1], theta = theta)
plot(x0, f, type = 'l',ylim = c(0,0.8), main = '')

for (i in 1:3){
  f <- dwei(x0,  alpha = alpha[i], theta = theta)
  lines(x0, f, type = 'l', lty = i, ylim = c(0,0.8))
}
legend("topright", 
       legend = c('alpha = 1, theta = 0.5', 
                  'alpha = 2, theta = 0.5', 
                  'alpha = 3, theta = 0.5'),
       lty = c(1,2,3), bty = "n"
)


=========================================================================
  # 例~2.4
  # =========================================================================
# 假设某团体保单承保了1000个个体的风险。
# 每个个体风险的保额为1，
# 发生索赔的概率均为q=0.001
# 求保险公司的总赔款大于3.5的概率

# =======================================
# 1. 二项分布和泊松分布
# =======================================

q <- 0.001
n <- 1000
# 总赔款 S 服从二项分布(1000,0.001)，也可以近似服从lambda=1的泊松分布
# P(S>=3.5)的概率为
1 - pbinom(3.5, 1000, 0.001)
1 - ppois(3.5, 1)


# =======================================
# 2. 对数正态近似
# =======================================
# 根据矩估计得到对数正态分布的两个参数 mu 和 sigma
mu <- -0.34657
sigma <- 0.83256
1 - plnorm(3.5, mu, sigma)


# =======================================
# 3. 平移伽马近似
# =======================================
# 根据矩估计得到平移伽马分布的三个参数 x0 alpha beta
x0 <- -1
alpha <- 4 
beta <- 2
# S + 1 服从参数为 (4,2) 的伽马分布
1 - pgamma(4.5, shape = 4, rate = 2)


# =======================================
# 4. 平移伽马近似
# =======================================
# 泊松分布的均值、方差、偏度系数为 1
mu <- sigma <- gamma <- 1
1 - pnorm(2)


# ================================================
# 连续型分布离散化
# ================================================
  library(actuar)
  discretize(cdf, from, to, step = 1, method = c("upper", "lower",
                                               "rounding","unbiased"), lev, by = step, xlim = NULL)


  fu <- discretize(cdf = pexp(x, rate = 0.1), 
                      method = 'upper', 
                      from = 0, to = 50, step = 2)
  fl <- discretize(cdf = pexp(x, rate = 0.1), 
                   method = 'lower', 
                   from = 0, to = 50, step = 2)
  fr <- discretize(cdf = pexp(x, rate = 0.1), 
                   method = 'rounding', 
                   from = 0, to = 50, step = 2)

  #par(col = "blue")
  x <- seq(0, 50, 2)
  curve(pexp(x, rate = 0.1), xlim = c(0, 50),
        ylab = '指数分布密度函数')
  plot(stepfun(head(x, -1), diffinv(fu)), pch = 19, 
       col = "red",
       add = TRUE)
  plot(stepfun(x, diffinv(fl)), pch = 18,
       add = TRUE,
       col = 'blue'
       )
  plot(stepfun(head(x,-1), diffinv(fr)), pch = 17,
       add = TRUE,
       col = 'yellow'
  )
  legend(30, 0.4, 
         legend = c("Upper", "Lower", "Midpoint"),
         col = c("red", "blue", "yellow"), 
         pch = 19, lty = 1)
  
  

# =========================================================================  
# 累积损失的分布几种方法
# =========================================================================   
  aggregateDist(method = c("recursive", "convolution", "normal", "npower",
                           "simulation"), model.freq = NULL, model.sev = NULL, p0 = NULL, x.scale = 1,
                moments, nb.simul, ..., tol = 1e-06, maxit = 500, echo = FALSE)
# 1. 卷积法
# 假设：强度分布，货币单位数X 从0 到10，频率分布， N 从0 到8，货
# 币单位数设为25。通过作图观察S 的最大值是不是10 × 25 × 8 = 2000？
  par(mfrow = c(2, 2))
  fx1 = c(0, 0.15, 0.2, 0.25, 0.125, 0.075, 0.05, 0.05, 0.05, 0.025,
          0.025)
  pn1 = c(0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.06, 0.03, 0.01)
  Fs1 = aggregateDist("convolution", model.freq = pn1, model.sev = fx1,
                       x.scale = 25  # 货币单位数
                      )
  plot(Fs1)
# 2. 递推法
# 首先对Gamma 分布进行离散化得到强度分布，频数分布选用poisson 分布, 特别
# 指定poisson 分布参数lambda=10。
  fx2 = discretize(pgamma(x, 2, 1), from = 0, to = 22, step = 0.5,
                   method = "rounding")
  Fs2 = aggregateDist("recursive", model.freq = "poisson", model.sev = fx2,
                       lambda = 10, x.scale = 0.5)
  plot(Fs2)

# 3. 正态近似和正态幂近似，注意正态幂近似的有效范围。
  Fs3 = aggregateDist("normal", moments = c(200, 200))
  plot(Fs3)
  Fs4 = aggregateDist("npower", moments = c(200, 200, 0.5))
  plot(Fs4)
# 4. 模拟法
  par(mfrow = c(1, 2))
  model.freq = expression(data = rpois(100))
  model.sev = expression(data = rgamma(100, 2))
  Fs5 = aggregateDist("simulation", nb.simul = 100000, model.freq,
                       model.sev)
  summary(Fs5)
  plot(Fs5)
# 5. 模拟法的另外一种写法
  set.seed(1112)
  Nsim <- 100000 # 模拟次数
  S <- 0
  for (i in 1:Nsim){
    N <- rpois(1, lambda = 100)
    if (N == 0){
      Yi <- 0
    }else{
      Yi <- rgamma(N, 100, 2)
    }
    S[i] <- sum(Yi)
  }
  plot(ecdf(S))
  summary((S))
 
# =========================================================================
# 例~2.7 随机模拟求累积损失的分布
# =========================================================================


set.seed(321) # 设定随机种子
iter <- 10000 # 模拟次数
d <- 250; u <- 1000 # 免赔额和限额
r <- 3; beta <- 2 # 负二项分布的参数
alpha <- 100; theta <- 0.2 # 伽马分布的参数
P <- NULL # 保险人的年度累积赔款

# 开始模拟
for (i in 1:iter){
  n <- rnbinom(1, size = r, mu = r*beta)  # 模拟损失次数
  x <- rgamma(n, shape = alpha, rate = theta) # 模拟每次事故的损失额，x 是一个向量
  w <- pmin(x, d)  # 保单持有人对每次损失的自负金额
  v <- min(sum(w), u) # 保单持有人自负的总金额\
  S <- sum(x)   # 保单持有人的总损失
  P[i] <- S - v # 保单持有人的年度累积赔款
}

hist(P, breaks = 50, col = 'grey', prob = T, main = '',
     ylab = '频率', xlab = '累积赔款'
)
mean(P);quantile(P, 0.95)


# ======================================
# 例 3.4 含零赔款、非零赔款的分布
# ======================================
  x <- seq(0, 100)
  f.yl <- function(y){
    3*100^3/(120 + y)^4
  }
  f.yp <- function(y){
    3*120^3/(120 + y)^4
  }
  f.x <- function(y){
    3*100^3/(100 + y)^4
  }
  
  plot(x, f.x(x), type = 'l')
  lines(x, f.yl(x), lty = 2, col = 2)
  lines(x, f.yp(x), lty = 3, col = 3)
  legend('topright',
         legend = c('实际损失', '含零赔款', '非零赔款'),
         lty = c(1, 2, 3),
         col = c(1, 2, 3))






