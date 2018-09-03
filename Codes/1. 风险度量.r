# ==============================================================================================
# 1. ���ն���
# ==============================================================================================
# Ť������ (�Բ���Ϊ2��ָ���ֲ�����Ť��,Ť������Ϊg(t)=t^0.5)
# ==============================================================================================
x = seq(0.01,10,0.001)
a = 2
f1 = 1/a*exp(-x/a)
f2 = 1/(2*a)*exp(-x/(2*a))
plot(x, f1, pch ='', ylab = '�ܶȺ���', main = '�Բ���Ϊ2��ָ���ֲ�����Ť��,Ť������Ϊg(t)=t^0.5')
lines(x, f1, col = 1)
lines(x, f2, col = 2)
text(3.5, 0.4, expression(ָ���ֲ����ܶȺ���: f(x)==frac(1,theta)*e^(-x/theta)),col=1)
text(6.5, 0.15, expression(Ť������ܶȺ���: tilde(f)(x)==frac(1,2*theta)*e^(-x/(2*theta))),col=2)

s1 = exp(-x/a)
s2 = s1^0.5
plot(x, s1, pch = '', ylab = '���溯��', main = '�Բ���Ϊ2��ָ���ֲ�����Ť��,Ť������Ϊg(t)=t^0.5')
lines(x, s1, col = 1)
lines(x, s2, col = 2)
text(2.5, 0.2,expression(ָ���ֲ������溯��: S(x)==e^{x/theta}),col=1)
text(6, 0.6,expression(Ť��������溯��: g(S(x))==e^{x/(2*theta)}),col=2)

# ==============================================================================================
# ������ϰ
# ==============================================================================================
set.seed(111)
loss = c(rlnorm(100, 0, 1), rep(2, 40))
p = 1:length(loss)/length(loss)
plot(sort(loss), p, type = "s")
VaR = quantile(loss, 0.99)
VaR
TVaR = mean(loss[loss > VaR])
TVaR

# ==============================================================================================
# ������ϰ
# ������ʧ����ָ���ֲ�����ֵΪ10���� r = 50ʱ������PH���ն���ֵ��
# ==============================================================================================
##����һ
s1 = function(x)  (1 - pexp(x, rate = 1/10))^(1/50)
integrate(s1,  0,  Inf)$value

##������
s2 = function(x)   exp(-x/10/50)
integrate(s2,  0,  Inf)$value


# ==============================================================================================
# ������ϰ
# ������ʧ����٤��ֲ�����״����Ϊ shape = 2���߶Ȳ���Ϊ scale = 1000������ PH �任�ķ��ն���ֵ
# �� Wang �任�ķ��ն���ֵ���ŷ������ϵ���仯������
# ==============================================================================================
# ����gamma�ֲ������溯��
S <- function(x) 1 - pgamma(x, shape = 2, scale = 1000)
# PH �任
r <- seq(1, 10, by = 0.1)
PH <- NULL
for(i in 1: length(r)){
  S_PH <- function(x) S(x)^(1/r[i])
  PH[i] <- integrate(S_PH, 0, Inf)$value
}
# Wang �任
Wang <- NULL
alpha <- seq(0.5, 0.99, by = 0.01)
k <- qnorm(alpha)
for(i in 1: length(alpha)){
  S_Wang <- function(x) pnorm(qnorm(S(x)) + k[i])
  Wang[i] <- integrate(S_Wang, 0, Inf)$value
}
# ��ͼ
par(mfrow = c(1, 2))
plot(r, PH, type = 'l')
plot(alpha, Wang, type = 'l')

# ==============================================================================================
# ������ϰ
# ����״����Ϊ shape = 2���߶Ȳ���Ϊ scale = 1000 ���� 500 �������������������������������ն���ֵ������
# ==============================================================================================
# ���������
set.seed(123)
x <- rgamma(n = 500, shape = 2, scale = 1000)
PH <- Wang <- NULL
# ٤��ֲ������溯��(�������溯��)
S  <- seq(1, 1/length(x), -1/length(x))
# PH �任
r <- seq(1, 10, by = 0.1)
for(i in 1: length(r)){
  PH[i] <- sum(diff(sort(c(0, x)))*S^(1/r[i]))
}
# Wang �任
alpha <- seq(0.5, 0.99, by = 0.01)
k <- qnorm(alpha)
for(i in 1: length(alpha)){
  Wang[i] <- sum(diff(sort(c(0, x)))*pnorm(qnorm(S) + k[i]))
}
# ��ͼ
par(mfrow = c(1, 2))
plot(r, PH, type = 'l')
plot(alpha, Wang, type = 'l')


# ==============================================================================================
# Esscher principle (����ԭ��)
# ==============================================================================================
par(mfrow = c(1, 1))
h = 0.002; shape = 2; scale = 100
f = function(x) dgamma(x, shape = shape, scale = scale)
f2 = function(x) dgamma(x, shape = shape, scale = scale)*exp(h*x)
M = integrate(f,0,Inf)$value
g = function(x)  dgamma(x, shape = shape, scale = scale)*exp(h*x)/M
curve(f(x), xlim = c(0,1000), ylim = c(0,0.008))
curve(g(x), xlim = c(0,1000), col = 2, lty = 2, lwd = 2, add = T)
text(500,0.006,'����ΪEsscher�任����ܶȺ���,h=0.002,
     ʵ�߱�ʾԭ�ֲ����ܶȺ�����Ϊ٤��(shape=2,scale=100)')

