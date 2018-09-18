
# ================================================================
# 1. ���ɷֲ�
# ================================================================
# density function ���ʺ���
dpois(x, lambda, log = FALSE) # lambda ��ʾ��ֵ, log = TRUE ��ʾ���log(f(x))
# distribution function �ֲ�����
ppois(q, lambda, lower.tail = TRUE, log.p = FALSE)
# quantile function �ֲ��������溯������λ��������
qpois(p, lambda, lower.tail = TRUE, log.p = FALSE)
# ���ɷֲ�������� - ģ��
rpois(n, lambda = 10) # ģ�� n ����������Ϊ10�Ĳ��ɷֲ�
# -----------------------------------------------------------------
# ��ͼ- ��ͬlambda�Ĳ��ɷֲ��ĸ��ʺ���ͼ
# -----------------------------------------------------------------
lambda.po <- c(1, 2, 5, 10) # lambda ȡֵΪ 1,2,5,10
x0 <- seq(0, 25)            # x ȡֵΪ 0-25 ������
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


## �����������3�ĸ���
dpois(3, lambda = 2)
## �������С�ڵ���4�ĸ���Ϊ
ppois(4, lambda = 2)
## ����������ڵ���3С�ڵ���5�ĸ���
ppois(5, 2) - ppois(2, 2)

## ģ��20����������۲�ֵ
set.seed(111) # �趨�������
sim = rpois(n = 20, lambda = 2) # ģ������20������lambda=2�Ĳ��ɷֲ����������
sim
## ��ģ���������������б�
table(sim)


# ================================================================
# 2. ������ֲ�
# ================================================================
# ���� r = size
# ���� beta = q
dnbinom(x, size, prob, mu, log = FALSE)
pnbinom(q, size, prob, mu, lower.tail = TRUE, log.p = FALSE)
qnbinom(p, size, prob, mu, lower.tail = TRUE, log.p = FALSE)
rnbinom(n, size, prob, mu)

# # ��ͼ 
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
# ���� ��ضϸ�����ֲ��ļ���
# -----------------------------------------------------------------
##������ֲ��ĸ���
x = 0:10
p = dnbinom(x, 4, 0.7)
round(p,3)
##��ضϸ�����ֲ��ĸ���
p0 = p[1]   
##���ĸ���
pt1 = p[2:11]/(1-p0)  
##�������ϵĸ���
pt = c(0, pt1)
round(pt, 3)
##��ͼ�Ƚϸ��������ضϸ�����ĸ���
com = rbind(������ = p, ��ضϸ����� = pt)
par(mfrow = c(1, 1))
barplot(com, beside = TRUE, names.arg = 0:10,legend.text = TRUE)

# -----------------------------------------------------------------
#���� �����������ֲ��ļ���
# -----------------------------------------------------------------
##������ֲ��ĸ���
x = 0:10
p = dnbinom(x, 4, 0.7)
round(p,3)
##�����������ֲ��ĸ���
p0 = 0.3  
##�������ĸ���
pm = (1 - p0)*p[2:11]/(1 - p[1])  
##�������ϵĸ���
pm = c(p0, pm)
round(pm, 3)
##��ͼ
com = rbind(������ = p, ����������� = pm)
barplot(com, beside = TRUE, names.arg = 0:10, legend.text = TRUE)




# ================================================================
# ����ֲ�
# ================================================================
# ���� m = size
# ���� prob = q
dbinom(x, size, prob, log = FALSE)
pbinom(q, size, prob, lower.tail = TRUE, log.p = FALSE)
qbinom(p, size, prob, lower.tail = TRUE, log.p = FALSE)
rbinom(n, size, prob)



# ��ͼ 
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
# ���ηֲ�
# ================================================================
# ��ͼ 
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
# ָ���ֲ�
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
# ٤��ֲ�
# ===========================================
par(mfrow = c(1, 2) )

# �̶���״����
alpha <- 2                # ��״����
theta <- c(0.5, 1, 2)     # �߶Ȳ���/���ʲ���
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


# �̶��߶Ȳ���
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
# ���˹�ֲ�
# ===========================================
# �������˹�ֲ����ܶȺ��� fx
dinvguass <- function(y, alpha, theta, ... ){
  fx <- alpha/(2*pi*theta*y^3)^0.5*exp(-(alpha - theta*y)^2/(2*theta*y))
  return(fx)
}


par(mfrow = c(1, 2) )
# �̶� alpha 
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


# �̶� theta
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
# ������̬�ֲ�
# ===========================================
# �ܶȺ���
dlnorm

par(mfrow = c(1, 2) )
# �̶� mu 
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


# �̶� sigma
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
# �����зֲ�
# ===========================================
dpareto <- function(y, alpha, theta){
  f <- alpha*(theta^alpha)/(y+theta)^(alpha+1)
  return(f)
}



par(mfrow = c(1, 2) )
# �̶� alpha 
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


# �̶� theta
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
# �������ֲ�
# ===========================================

# �����ܶȺ���
dwei <- function(y, alpha, theta){
  f <- alpha*theta*y^(theta-1)*exp(-alpha*y^theta)
  return(f)
}

par(mfrow = c(1, 2) )
# �̶� alpha 
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


# �̶� theta
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
  # ��~2.4
  # =========================================================================
# ����ĳ���屣���б���1000������ķ��ա�
# ÿ��������յı���Ϊ1��
# ��������ĸ��ʾ�Ϊq=0.001
# ���չ�˾����������3.5�ĸ���

# =======================================
# 1. ����ֲ��Ͳ��ɷֲ�
# =======================================

q <- 0.001
n <- 1000
# ����� S ���Ӷ���ֲ�(1000,0.001)��Ҳ���Խ��Ʒ���lambda=1�Ĳ��ɷֲ�
# P(S>=3.5)�ĸ���Ϊ
1 - pbinom(3.5, 1000, 0.001)
1 - ppois(3.5, 1)


# =======================================
# 2. ������̬����
# =======================================
# ���ݾع��Ƶõ�������̬�ֲ����������� mu �� sigma
mu <- -0.34657
sigma <- 0.83256
1 - plnorm(3.5, mu, sigma)


# =======================================
# 3. ƽ��٤�����
# =======================================
# ���ݾع��Ƶõ�ƽ��٤��ֲ����������� x0 alpha beta
x0 <- -1
alpha <- 4 
beta <- 2
# S + 1 ���Ӳ���Ϊ (4,2) ��٤��ֲ�
1 - pgamma(4.5, shape = 4, rate = 2)


# =======================================
# 4. ƽ��٤�����
# =======================================
# ���ɷֲ��ľ�ֵ�����ƫ��ϵ��Ϊ 1
mu <- sigma <- gamma <- 1
1 - pnorm(2)


# ================================================
# �����ͷֲ���ɢ��
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
        ylab = 'ָ���ֲ��ܶȺ���')
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
# �ۻ���ʧ�ķֲ����ַ���
# =========================================================================   
  aggregateDist(method = c("recursive", "convolution", "normal", "npower",
                           "simulation"), model.freq = NULL, model.sev = NULL, p0 = NULL, x.scale = 1,
                moments, nb.simul, ..., tol = 1e-06, maxit = 500, echo = FALSE)
# 1. �����
# ���裺ǿ�ȷֲ������ҵ�λ��X ��0 ��10��Ƶ�ʷֲ��� N ��0 ��8����
# �ҵ�λ����Ϊ25��ͨ����ͼ�۲�S �����ֵ�ǲ���10 �� 25 �� 8 = 2000��
  par(mfrow = c(2, 2))
  fx1 = c(0, 0.15, 0.2, 0.25, 0.125, 0.075, 0.05, 0.05, 0.05, 0.025,
          0.025)
  pn1 = c(0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.06, 0.03, 0.01)
  Fs1 = aggregateDist("convolution", model.freq = pn1, model.sev = fx1,
                       x.scale = 25  # ���ҵ�λ��
                      )
  plot(Fs1)
# 2. ���Ʒ�
# ���ȶ�Gamma �ֲ�������ɢ���õ�ǿ�ȷֲ���Ƶ���ֲ�ѡ��poisson �ֲ�, �ر�
# ָ��poisson �ֲ�����lambda=10��
  fx2 = discretize(pgamma(x, 2, 1), from = 0, to = 22, step = 0.5,
                   method = "rounding")
  Fs2 = aggregateDist("recursive", model.freq = "poisson", model.sev = fx2,
                       lambda = 10, x.scale = 0.5)
  plot(Fs2)

# 3. ��̬���ƺ���̬�ݽ��ƣ�ע����̬�ݽ��Ƶ���Ч��Χ��
  Fs3 = aggregateDist("normal", moments = c(200, 200))
  plot(Fs3)
  Fs4 = aggregateDist("npower", moments = c(200, 200, 0.5))
  plot(Fs4)
# 4. ģ�ⷨ
  par(mfrow = c(1, 2))
  model.freq = expression(data = rpois(100))
  model.sev = expression(data = rgamma(100, 2))
  Fs5 = aggregateDist("simulation", nb.simul = 100000, model.freq,
                       model.sev)
  summary(Fs5)
  plot(Fs5)
# 5. ģ�ⷨ������һ��д��
  set.seed(1112)
  Nsim <- 100000 # ģ�����
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
# ��~2.7 ���ģ�����ۻ���ʧ�ķֲ�
# =========================================================================


set.seed(321) # �趨�������
iter <- 10000 # ģ�����
d <- 250; u <- 1000 # �������޶�
r <- 3; beta <- 2 # ������ֲ��Ĳ���
alpha <- 100; theta <- 0.2 # ٤��ֲ��Ĳ���
P <- NULL # �����˵�����ۻ����

# ��ʼģ��
for (i in 1:iter){
  n <- rnbinom(1, size = r, mu = r*beta)  # ģ����ʧ����
  x <- rgamma(n, shape = alpha, rate = theta) # ģ��ÿ���¹ʵ���ʧ�x ��һ������
  w <- pmin(x, d)  # ���������˶�ÿ����ʧ���Ը����
  v <- min(sum(w), u) # �����������Ը����ܽ��\
  S <- sum(x)   # ���������˵�����ʧ
  P[i] <- S - v # ���������˵�����ۻ����
}

hist(P, breaks = 50, col = 'grey', prob = T, main = '',
     ylab = 'Ƶ��', xlab = '�ۻ����'
)
mean(P);quantile(P, 0.95)


# ======================================
# �� 3.4 �������������ķֲ�
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
         legend = c('ʵ����ʧ', '�������', '�������'),
         lty = c(1, 2, 3),
         col = c(1, 2, 3))






