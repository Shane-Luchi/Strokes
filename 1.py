import numpy as np
import matplotlib.pyplot as plt

# 定义损失函数 L(w) = (w - 2)^2
def loss_function(w):
    return (w - 2) ** 2

# 定义梯度 dL/dw = 2 * (w - 2)
def gradient(w):
    return 2 * (w - 2)

# 基础梯度下降（固定方式）
def gradient_descent(w_init, learning_rate, iterations):
    w = w_init
    w_history = [w]
    for _ in range(iterations):
        grad = gradient(w)
        w = w - learning_rate * grad  # 固定步长更新
        w_history.append(w)
    return w_history

# Adam 优化器
def adam(w_init, learning_rate, iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
    w = w_init
    m = 0  # 一阶动量（梯度均值）
    v = 0  # 二阶动量（梯度方差）
    w_history = [w]
    
    for t in range(1, iterations + 1):
        grad = gradient(w)
        
        # 更新动量估计
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # 偏差校正
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # 参数更新
        w = w - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        w_history.append(w)
    
    return w_history

# 参数设置
w_init = 0.0  # 初始值
learning_rate = 0.1  # 学习率
iterations = 30  # 迭代次数

# 运行两种方法
gd_history = gradient_descent(w_init, learning_rate, iterations)
adam_history = adam(w_init, learning_rate, iterations)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(gd_history, label='Gradient Descent', marker='o')
plt.plot(adam_history, label='Adam', marker='x')
plt.axhline(y=2, color='r', linestyle='--', label='Optimal Value (w=2)')
plt.xlabel('Iteration')
plt.ylabel('Parameter w')
plt.title('Gradient Descent vs Adam Optimization')
plt.legend()
plt.grid(True)
plt.show()

# 打印最终结果
print(f"Gradient Descent final w: {gd_history[-1]:.4f}")
print(f"Adam final w: {adam_history[-1]:.4f}")