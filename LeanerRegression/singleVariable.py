import numpy as np
import matplotlib.pyplot as plt
x_array = [1.0, 2.0, 3.0, 4.0, 5.0]
# y = 2x + 3
y_array = [5.0, 7.0, 9.0, 11.0, 13.0]
loss_array = []
num = len(x_array)

learning_rate = 0.01
iter = 100

w = 2.0
b = 1.0

for i in range(iter):
    w_grad = 0.0
    b_grad = 0.0
    loss = 0.0

    for j in range(num):
        x = x_array[j]
        y = y_array[j]

        w_grad += x*(w*x+b-y)/num
        b_grad +=   (w*x+b-y)/num
        loss += pow( (w*x+b-y),2)

    loss = loss/(2*num)
    loss_array.append(loss)

    b = b - b_grad * learning_rate
    w = w - w_grad * learning_rate
plt.plot(loss_array)
print(b,w)