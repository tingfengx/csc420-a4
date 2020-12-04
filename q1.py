import numpy as np

# sigmoid activation
sigmoid = lambda x: 1 / (1 + np.exp(-x))
# sigmoid activation's derivative
dsigmoiddx = lambda x: sigmoid(x) * sigmoid(-x)

print("-> Initialization ... <-")
x1, x2, x3, x4 = 0.9, -1.1, -0.3, 0.8
w1, w2, w3, w4, w5, w6 = 0.75, -0.63, 0.24, -1.7, 0.8, -0.2
y = 0.5

print(f"x1 = {x1:.5f}, x2 = {x2:.5f}, x3 = {x3:.5f}, x4 = {x4:.5f}")
print(f"w1 = {w1:.5f}, w2 = {w2:.5f}, w3 = {w3:.5f}, \nw4 = {w4:.5f}, w5 = {w5:.5f}, w6 = {w6:.5f}")
print(f"target y = {y:.5f}")
print("-> Start Forward Pass <-")

sum1 = w1 * x1 + w2 * x2
sigma1 = sigmoid(sum1)
sum2 = w3 * x3 + w4 * x4
sigma2 = sigmoid(sum2)
sum3 = w5 * sigma1 + w6 * sigma2
sigma3 = sigmoid(sum3)
yhat = sigma3
L = (yhat - y) ** 2

print(f"sum1 = {sum1:.5f}, sigma1 = {sigma1:.5f}, sum2 = {sum2:.5f}")
print(f"sigma2 = {sigma2:.5f}, sum3 = {sum3:.5f}, sigma3 = {sigma3:.5f}")
print(f"yhat = {yhat:.5f}, L = {L:.5f}")

print("-> Start Back Propagation <-")
dLdL = 1
dLdyhat = dLdL * 2 * (yhat - y)
dLdsigma3 = dLdyhat * 1
dLdsum3 = dLdsigma3 * dsigmoiddx(sum3)
dLdsigma2 = dLdsum3 * w6
dLdsum2 = dLdsigma2 * dsigmoiddx(sum2)
dLdw3 = dLdsum2 * x3

print(f"dLdL = {dLdL:.5f}, dLdyhat = {dLdyhat:.5f}, dLdsigma3 = {dLdsigma3:.5f}")
print(f"dLdsum3 = {dLdsum3:.5f}, dLdsigma2 = {dLdsigma2:.5f}, dLdsum2 = {dLdsum2:.5f}")
print(f"-> End Result: dLdw3 = {dLdw3:.10f}")
