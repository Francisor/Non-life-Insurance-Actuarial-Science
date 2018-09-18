
# ==========================================
# ָ���ֲ�
# ===========================================
theta <- c(0.5, 1, 2)
x0 <- seq(0.001, 10, length.out = 100)
par(mfrow = c(1, 1) )
f1 <- dexp(x0, rate = theta[1], log = FALSE)
f2 <- dexp(x0, rate = theta[2], log = FALSE)
f3 <- dexp(x0, rate = theta[3], log = FALSE)

matplot(x0, cbind(f1, f2, f3), type = 'l', lty = 1:3, lwd = 2, ylim = c(0, 1), ylab = '�ܶȺ���')
legend('topright', legend = c('theta = 0.5', 'theta = 1', 'theta = 2'),
       lty = c(1,2,3), bty = "n", lwd = 2,  col = 1:3)

# ==========================================
# ٤��ֲ�
# ===========================================
par(mfrow = c(1, 2) )
# �̶���״����
alpha <- 2                # ��״����
theta <- c(0.5, 1, 2)     # ���ʲ������߶Ȳ���Ϊ 1/theta
x0 <- seq(0.001, 15, length.out = 100)
f1 <- dgamma(x0,  shape = alpha, rate = theta[1]) # 
f2 <- dgamma(x0,  shape = alpha, rate = theta[2])
f3 <- dgamma(x0,  shape = alpha, rate = theta[3])
matplot(x0, cbind(f1, f2, f3),ylim = c(0,0.8), main = '',  type = 'l', lty = 1:3, lwd = 2, ylab = '�ܶȺ���')
legend('topright', legend = c('alpha = 2, theta = 0.5', 
                              'alpha = 2, theta = 1', 
                              'alpha = 2, theta = 2'),
       lty = c(1,2,3), bty = "n", lwd = 2,  col = 1:3)

# �̶����ʲ���
alpha <- c(1,2,3)
theta <- 0.5
x0 <- seq(0.001, 15, length.out = 100)
f1 <- dgamma(x0,  shape = alpha[1], rate = theta)
f2 <- dgamma(x0,  shape = alpha[2], rate = theta)
f3 <- dgamma(x0,  shape = alpha[3], rate = theta)

matplot(x0, cbind(f1, f2, f3),ylim = c(0,0.8), main = '',  type = 'l', lty = 1:3, lwd = 2, ylab = '�ܶȺ���')
legend('topright',legend = c('alpha = 1, theta = 0.5', 
                                     'alpha = 2, theta = 0.5', 
                                     'alpha = 3, theta = 0.5'),
       lty = c(1,2,3), bty = "n", lwd = 2,  col = 1:3)


# ==========================================
# ���˹�ֲ�
# ===========================================
par(mfrow = c(1, 1))
x = seq(0, 4, 0.01)
f1 = sqrt(0.5/(2*pi*x^3))*exp(-0.5*(x-1)^2/(2*1^2*x))
f2 = sqrt(1/(2*pi*x^3))*exp(-1*(x-1)^2/(2*1^2*x))
f3 = sqrt(5/(2*pi*x^3))*exp(-5*(x-1)^2/(2*1^2*x))
f4 = sqrt(10/(2*pi*x^3))*exp(-10*(x-1)^2/(2*1^2*x))
matplot(x, cbind(f1, f2, f3, f4), type = 'l', lty = 1:4, lwd = 2)
legend(2, 1, c('IG(1, 0.5)', 'IG(1, 1)', 'IG(1, 5)', 'IG(1, 10)'), lty=1:4, col=1:4, lwd=c(3, 3, 3, 3))

# ����һ�ַ���
# �������˹�ֲ����ܶȺ��� fig
dig <- function(y, alpha, theta, ... ){
  fx <- alpha/(2*pi*theta*y^3)^0.5*exp(-(alpha - theta*y)^2/(2*theta*y))
  return(fx)
}

par(mfrow = c(1, 2) )
# �̶� alpha 
alpha <- 2                
theta <- c(0.5, 1, 2)     
x0 <- seq(0.001, 15, length.out = 100)
f1 <- dig(x0,  alpha = alpha, theta = theta[1])
f2 <- dig(x0,  alpha = alpha, theta = theta[2])
f3 <- dig(x0,  alpha = alpha, theta = theta[3])
matplot(x0, cbind(f1, f2, f3), main = '',  type = 'l', lty = 1:3, lwd = 2, ylab = '�ܶȺ���')
legend('topright',legend = c('alpha = 2, theta = 0.5', 
                              'alpha = 2, theta = 1', 
                              'alpha = 2, theta = 2'),
       lty = c(1,2,3), bty = "n", lwd = 2,  col = 1:3)

# �̶� theta
alpha <- c(1,2,3)
theta <- 0.5
x0 <- seq(0.001, 15, length.out = 100)
f1 <- dig(x0,  alpha = alpha[1], theta = theta)
f2 <- dig(x0,  alpha = alpha[2], theta = theta)
f3 <- dig(x0,  alpha = alpha[3], theta = theta)
matplot(x0, cbind(f1, f2, f3), main = '',  type = 'l', lty = 1:3, lwd = 2, ylab = '�ܶȺ���')
legend('topright',       legend = c('alpha = 1, theta = 0.5', 
                                    'alpha = 2, theta = 0.5', 
                                    'alpha = 3, theta = 0.5'),
       lty = c(1,2,3), bty = "n", lwd = 2,  col = 1:3)

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
f1 <- dlnorm(x0,  meanlog = mu, sdlog = sigma[1])
f2 <- dlnorm(x0,  meanlog = mu, sdlog = sigma[2])
f3 <- dlnorm(x0,  meanlog = mu, sdlog = sigma[3])

matplot(x0, cbind(f1, f2, f3), main = '',  type = 'l', lty = 1:3, lwd = 2, ylab = '�ܶȺ���')
legend('topright',legend = c('mu = 2, sigma = 0.5', 
                             'mu = 2, sigma = 1', 
                             'mu = 2, sigma = 2'),
       lty = c(1,2,3), bty = "n", lwd = 2,  col = 1:3)

# �̶� sigma
mu <- c(1,2,3)
sigma <- 1
x0 <- seq(0.001, 15, length.out = 100)
f1 <- dlnorm(x0,  meanlog = mu[1], sdlog = sigma)
f2 <- dlnorm(x0,  meanlog = mu[2], sdlog = sigma)
f3 <- dlnorm(x0,  meanlog = mu[3], sdlog = sigma)

matplot(x0, cbind(f1, f2, f3), main = '',  type = 'l', lty = 1:3, lwd = 2, ylab = '�ܶȺ���')
legend('topright',       legend = c('mu = 1, sigma = 1', 
                                    'mu = 2, sigma = 1', 
                                    'mu = 3, sigma = 1'),
       lty = c(1,2,3), bty = "n", lwd = 2,  col = 1:3)


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
x0 <- seq(0.001, 5, length.out = 100)
f1 <- dpareto(x0,  alpha = alpha, theta = theta[1])
f2 <- dpareto(x0,  alpha = alpha, theta = theta[2])
f3 <- dpareto(x0,  alpha = alpha, theta = theta[3])
matplot(x0, cbind(f1, f2, f3), main = '',  type = 'l', lty = 1:3, lwd = 2, ylab = '�ܶȺ���', ylim = c(0,0.6))
legend('topright', legend = c('alpha = 2, theta = 0.5', 
                              'alpha = 2, theta = 1', 
                              'alpha = 2, theta = 2'),
       lty = c(1,2,3), bty = "n", lwd = 2,  col = 1:3)
# �̶� theta
alpha <- c(1,2,3)
theta <- 0.5
x0 <- seq(0.001, 5, length.out = 100)
f1 <- dpareto(x0,  alpha = alpha[1], theta = theta)
f2 <- dpareto(x0,  alpha = alpha[2], theta = theta)
f3 <- dpareto(x0,  alpha = alpha[3], theta = theta)
matplot(x0, cbind(f1, f2, f3), main = '',  type = 'l', lty = 1:3, lwd = 2, ylab = '�ܶȺ���', ylim = c(0,0.6))
legend('topright', legend = c('alpha = 1, theta = 0.5', 
                              'alpha = 2, theta = 0.5', 
                              'alpha = 3, theta = 0.5'),
       lty = c(1,2,3), bty = "n", lwd = 2,  col = 1:3)


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

f1 <- dwei(x0,  alpha = alpha, theta = theta[1])
f2 <- dwei(x0,  alpha = alpha, theta = theta[2])
f3 <- dwei(x0,  alpha = alpha, theta = theta[3])
matplot(x0, cbind(f1, f2, f3), main = '',  type = 'l', lty = 1:3, lwd = 2, ylab = '�ܶȺ���', ylim = c(0,1))
legend('topright', legend = c('alpha = 1, theta = 0.5', 
                              'alpha = 1, theta = 1', 
                              'alpha = 1, theta = 2'),
       lty = c(1,2,3), bty = "n", col = 1:3)

# �̶� theta
alpha <- c(1,2,3)
theta <- 0.5
x0 <- seq(0.001, 5, length.out = 100)
f1 <- dwei(x0,  alpha = alpha[1], theta = theta)
f2 <- dwei(x0,  alpha = alpha[2], theta = theta)
f3 <- dwei(x0,  alpha = alpha[3], theta = theta)
matplot(x0, cbind(f1, f2, f3), main = '',  type = 'l', lty = 1:3, lwd = 2, ylab = '�ܶȺ���', ylim = c(0,1))
legend('topright', legend = c('alpha = 1, theta = 0.5', 
                              'alpha = 2, theta = 0.5', 
                              'alpha = 3, theta = 0.5'),
       lty = c(1,2,3), bty = "n", col = 1:3)

# theta = 3.6
par(mfrow = c(1, 1))
x0 <- seq(0.001, 5, length.out = 100)
f0 <- dwei(x0,  alpha = 0.1, theta = 3.6)
plot(x0, f0, type = 'l', col = 2, lwd = 3)
legend('topright', legend = c('alpha = 0.1, theta = 3.6'), 
       lwd = 3, bty = "n", col = 2)
# -----------------------------------------------------------------------------------
# ����X���Ӳ���Ϊ (3,  4) ��٤��ֲ�, ��g(X)�ķֲ���
# -----------------------------------------------------------------------------------
# ٤��ֲ����ܶȺ���
par(mfrow = c(1, 1))
f = function(x)  dgamma(x,  3,  4)
### ָ���任,  Y = exp(X)
f1 = function(x)   f(log(x))/x
### �����任,   Y = log(X)
f2 = function(x)    f(exp(x)) * exp(x)
x <- seq(0, 4, 0.01)
matplot(x, cbind(f(x), f1(x), f2(x)), type='l', lty=1:3, lwd=2)
legend('topright', c('X','exp(X)','log(X)'), lty=1:3, col=1:3, lwd=2)


# -----------------------------------------------------------------------------------
# ���� ����������̬�ֲ��Ĳ����ֱ�Ϊ(1, 2)��(3, 4), �������30%��70%�ı��������ǽ��л��, ���Ϸֲ����ܶȺ�����
# -----------------------------------------------------------------------------------
p = 0.3
m1 = 1; s1 = 2
m2 = 3; s2 = 4
## ��϶�����̬�ֲ����ܶȺ���
f = function(x)  p * dlnorm(x,  m1,  s1) + (1 - p) * dlnorm(x,  m2,  s2)
curve(f,  xlim = c(0,  1),  ylim = c(0,  2),   lwd = 2,  col = 2, main = '��϶�����̬�ֲ�')
curve(dlnorm(x,  m1,  s1),  lty = 2,  add = TRUE)
curve(dlnorm(x,  m2,  s2),  lty = 3,  add = TRUE)
legend("topright",  c("mixed lnorm",  "lnorm(1, 2)",  "lnorm(3, 4)"),  lty = c(1,  2,  3),  col = c(2,  1,  1),  lwd = c(2,  1,  1))


# -----------------------------------------------------------------------------------
# ���ָ���ֲ�
# -----------------------------------------------------------------------------------
x = seq(0,  10,  0.01)
y1 = 1-pexp(x,  rate = 2)
y2 = 1-pexp(x,  rate = 3)
q = 0.7
y = q*y1 + (1 - q)* y2
matplot(x,  cbind(y1,  y2,  y),  lty=c(2,3,1),type = 'l',  col=c(1,2,4), xlim = c(0,  3), lwd=2, main = '���溯��')
legend('topright',  c('ָ����rate = 2��', 'ָ����rate = 3��',  '���ָ����q = 0.7��'),  lty=c(2,3,1), col=c(1,2,4))


# -----------------------------------------------------------------------------------
# ���ֹ��Ʒ����ıȽ�
# -----------------------------------------------------------------------------------
# ģ��٤��ֲ��������
set.seed(123)
x = rgamma(50, 2)  
# ����fitdistrplus�����
library(fitdistrplus)  
# �ü�����Ȼ�����Ʋ���
fit1 = fitdist(x,  'gamma',  method = 'mle')  
# �þع��Ʒ����Ʋ���
fit2 = fitdist(x, 'gamma', method = 'mme')  
# �÷�λ����ȷ����Ʋ���
fit3 = fitdist(x, 'gamma', method = 'qme', probs = c(1/3, 2/3))  
#����С���뷨���Ʋ���
fit4 = fitdist(x, 'gamma', method = 'mge', gof = 'CvM')  
#����������ƽ��
fit1  
plot(fit1)

# -----------------------------------------------------------------------------------
# ��������
# ------------------------------------------------------------------------------------------------
# ���ó����CASdatasets�е����ݼ�freMTPLsev, Ӧ���ʵ���ģ�����ClaimAmount�ķֲ���
# ׼����
library(CASdatasets)
#׼������
data(freMTPLsev)  
x <- freMTPLsev$ClaimAmount
summary(x)
quantile(x, 90:100/100)
x <- x[x<=100000]
hist(x, breaks = 100000, xlim = c(0, 10000))

#------------��������x�ֶ�---------------
c1 = 400; c2 = 1000; c3 = 1300; c4 = 5000
index1 <- which(x<=c1)
index2 <- which(x>c1 & x<=c2)
index3 <- which(x>c2 & x<=c3)
index4 <- which(x>c3 & x<=c4)
index5 <- which(x>c4)
#������̬�ֲ����
fit1 = fitdist(x[index1], 'lnorm')
fit2 = fitdist(x[index2], 'lnorm')
fit3 = fitdist(x[index3], 'lnorm')
fit4 = fitdist(x[index4], 'lnorm')
##��β�������зֲ����
dpareto = function(x, alpha, theta = c4) alpha*theta^alpha/x^(alpha+1) 
ppareto = function(x, alpha, theta = c4) 1-(theta/x)^alpha
fit5 = fitdist(x[index5], 'pareto', start = 5)  #�����д�c3�Ժ��ж���

hist(x[index5], freq = F)
curve(dpareto(x, fit5$estimate[1]), add = T)

#------------�õ�����ֲ��Ĺ��Ʋ���-----------
m1 <- fit1$estimate[1]
s1 <- fit1$estimate[2]
m2 <- fit2$estimate[1]
s2 <- fit2$estimate[2]
m3 <- fit3$estimate[1]
s3 <- fit3$estimate[2]
m4 <- fit4$estimate[1]
s4 <- fit4$estimate[2]
m5 <- fit5$estimate[1]
s5 <- fit5$estimate[2]

#######ʹ�÷ֶ���ϵ�Ȩ��
w1 = length(index1)/length(x)
w2 = length(index2)/length(x)
w3 = length(index3)/length(x)
w4 = length(index4)/length(x)
w5 = length(index5)/length(x)

f = function(x) {
  ifelse(x <= c1, w1*dlnorm(x, m1, s1)/(plnorm(c1, m1, s1)),
         ifelse(x > c1 & x <= c2, w2*dlnorm(x, m2, s2)/(plnorm(c2, m2, s2) - plnorm(c1, m2, s2)), 
                ifelse(x > c2 & x<= c3, w3*dlnorm(x, m3, s3)/(plnorm(c3, m3, s3) - plnorm(c2, m3, s3)),
                       ifelse(x > c3 & x<= c4, w4*dlnorm(x, m4, s4)/(plnorm(c4, m4, s4) - plnorm(c3, m4, s4)),
                              w5*dpareto(x, m5)))))
}  

hist(x, breaks = 5000, xlim = c(0, 6000), prob = TRUE,  main = "",  xlab = "�����", col='grey')
curve(f, xlim = c(0, 6000), add = T,  col=2,  lwd = 2)

# ==================================================
# ָ���ֲ��������зֲ���ƽ�������
# ==================================================
# ָ���ֲ������溯��
S <- function(x) exp(-2*x)

# ָ���ֲ���ƽ������� ex1
ex1 <- NULL
d1 <- seq(0.1, 5, 0.1) # �����
for(i in 1:length(d1)){
  ex1[i] <- integrate(S, d1[i], Inf)$value/S(d[i])
}










