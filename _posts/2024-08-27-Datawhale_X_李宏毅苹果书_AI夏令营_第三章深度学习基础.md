---
title: "2024-08-27-Datawhale_X_李宏毅苹果书_AI夏令营_第三章深度学习基础"
author: "BlackJack0083"
date: "2024-08-27"
toc: true
tags: ["深度学习"]
comments: true
---

# 局部最小值 & 鞍点

## 临界点

在深度学习中，当优化到某个地方，参数对损失的微分为零时，梯度下降就无法更新参数了。这时候可能有两种情况：
- 鞍点
- 局部最小值点

鞍点不是局部最小值，但梯度为零。把以上两种统称为**临界点**(critical point)。

## 判断临界值种类

虽然局部最小值点无路可走(没有方向进行更新)，但是鞍点旁边还有路可以降低loss，因此我们需要判断临界点种类来帮助进一步降低loss。

使用泰勒级数进行近似， $\theta'$ 附近的损失函数 $L(\theta)$ 可近似为：

$$L(\theta) \approx L(\theta') + (\theta - \theta')^T g + \frac{1}{2}(\theta - \theta')^TH(\theta - \theta') $$

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240824154309.png)

参考原本的一元函数的泰勒展开：

$$f(x) = g(x_0)+\frac{f^1(x_0)}{1!} + \frac{f^2(x_0)}{2!}+...+ \frac{f^n(x_0)}{n!}$$

发现这里进行到二次展开。

> [!ques] 为什么只需要展开到二次？
> 可能与后面的解释有关，发现只需要展开到二次就已经有办法逃离鞍点了。于是不进行进一步计算微分(当然也有计算量太大的原因)。

前面的 $g$ 表示梯度，而 $H$ 表示Hessian矩阵。这说明损失函数与两者有关。那么当损失函数达到临界点时，此时梯度$g=0$，于是损失函数可近似为：
$$L(\theta) \approx L(\theta')  + \frac{1}{2}(\theta - \theta')^TH(\theta - \theta') $$

用 $v$ 代替 $\theta - \theta'$ ，于是得到 $v^THv$ ，发现这个就是大一线代中学习的二次型(已经快忘光了)。那么由线性代数知识，我们知道可以根据二次型情况判断临界值种类：

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240824161437.png)

对于正定矩阵，其特征值都是正的，对应的 $L(\theta')$ 为局部最小值；反之对于负定矩阵，其特征值都是负的，对应的 $L(\theta')$ 为局部最大值；而若矩阵特征值有正有负，说明对应的 $L(\theta')$ 为鞍点。

### 例子

这是一个图:

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240824161939.png)

- 中间点为鞍点：往左下和右上会降低loss，而往左上和右下会提高loss
- 其余两排为局部最小值

loss使用SSE：

$$L= (\hat y - w_1w_2x)^2 = (1-w_1w_2)^2$$

计算梯度和Hessian矩阵，根据特征值判断临界点类型：
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240824162659.png)

## 逃离鞍点的方法

### 数值计算

设 $\lambda$ 为 $H$ 的一个特征值， $u$ 为**对应的特征向量**。对于优化问题，可以令 $u = \theta - \theta'$ ，则有

$$u^THu = u^T(\lambda u) = \lambda||u||^2$$

若 $\lambda < 0$，则 $\lambda||u||^2 < 0$ 。所以 $\frac{1}{2}u^THu < 0$ ，此时，根据上面的计算式得到，$L(\theta) < L(\theta')$，loss变小了，且

$$\theta = \theta' + u$$

于是发现，如果在鞍点，可以通过找到Hessian矩阵的**负特征值**及其对应的特征向量，与 $\theta'$ 相加，即可更新 $\theta$ ，找到**损失更低的点**。

但一般没人计算Hessian矩阵，因为计算量太大，同时需要计算二次微分(Adam还是挺香的)。

### 高维空间的临界点情况

由于神经网络参数量巨大，误差表面维度很高，导致局部最小值出现的概率会非常低，大部分临界点均为鞍点。
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240824164936.png)

根据实验发现，很多情况下，局部最小值不容易出现(一方面很少有特征值均为正的情况，另一方面training loss均很小)。证明应该需要用概率论。

# 批量和动量

## 批量

关于batch size: [深度学习中的batch的大小对学习效果有何影响？ - 摘星狐狸的回答 - 知乎](https://www.zhihu.com/question/32673260/answer/3356342576)

实际上在计算梯度的时候，并不是对所有数据的损失 L 计算梯度，而是把所有的数据分成一个一个的批量（batch）
每个批量的大小是 B ，即带有 B 笔数据。每次在更新参数的时候，会去取出 B 笔数据用来计算出损失和梯度更新参数。遍历所有批量的过程称为一个回合（epoch）。事实上，在把数据分为批量的时候，我们还会进行随机打乱（shuffle）。随机打乱有很多不同的做法，一个常见的做法是在每一个回合开始之前重新划分批量，也就是说，每个回合的批量的数据都不一样。

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240826211917.png)

> [!question] 批量大小有什么影响？
> 引用：[梯度下降法的三种形式BGD、SGD、MBGD及python实现-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1604182)
> 举个例子，以一个特征的线性回归为例
>
> $$h_{\theta}\left( x^i \right) =\theta_{1}x^i+\theta_{0}$$
> 

### BGD

对于一批样本，对应的目标函数为：

$$J\left( \theta_{0},\theta_{1} \right) =\frac{1}{2m}\sum_{i=1}^{m}\left( h_{\theta}\left( x^i \right) -y^i  \right)^2 $$

**批量梯度下降法**是最原始的形式，它是指在**每一次迭代时**使用**所有样本**来进行梯度的更新。从数学上理解如下
1. 对目标函数求偏导：

$$
\frac{\Delta J\left( \theta_{0},\theta_{1} \right)}{\Delta \theta_{j}}=\frac{1}{m}\sum_{i=1}^{m} \left( h_{\theta }\left( x^i \right) -y^i \right) x_{j}^i
$$

其中， $i=1,2,\dots,m$ 表示样本数， $j=0,1$ 表示特征数，其中 $x_{0}^i=1$ (算一下就出来了)
2. 每次迭代对参数进行更新

$$
\theta j:=\theta j-\alpha\frac{1}{m}\sum_{i=1}^{m}\left( h_{\theta}\left( x^i \right) -y^i \right) x_{j}^i
$$

注意这里的求和函数，即对所有样本进行计算，随后取平均

**优点**：   
（1）一次迭代是对所有样本进行计算，此时利用矩阵进行操作，实现了**并行**。   
（2）由全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。当目标函数为**凸函数**时，BGD一定能够得到全局最优。 

**缺点**：
（1）当样本数目 $m$ 很大时，每迭代一步都需要对所有样本计算，训练过程会很慢。   
从迭代的次数上来看，BGD迭代的次数相对较少。其迭代的收敛曲线示意图可以表示如下：

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240826214855.png)

（2）容易陷入局部最小值

```python
import matplotlib.pyplot as plt
import random
##样本数据
x_train = [150,200,250,300,350,400,600]
y_train = [6450,7450,8450,9450,11450,15450,18450]
#样本个数
m = len(x_train)
#步长
alpha = 0.00001
#循环次数
cnt = 0
#假设函数为 y=theta0+theta1*x
def h(x):
    return theta0 + theta1*x
theta0 = 0
theta1 = 0
#导数
diff0=0
diff1=0
#误差
error0=0           
error1=0          
#每次迭代theta的值
retn0 = []         
retn1 = []         

#退出迭代的条件
epsilon=0.00001

#批量梯度下降
while 1:
    cnt=cnt+1
    diff0=0
    diff1=0
    #梯度下降
    for i in range(m):
        diff0+=h(x_train[i])-y_train[i]
        diff1+=(h(x_train[i])-y_train[i])*x_train[i]
    theta0=theta0-alpha/m*diff0
    theta1=theta1-alpha/m*diff1
    retn0.append(theta0)
    retn1.append(theta1)
    error1=0
    #计算迭代误差
    for i in range(len(x_train)):
        error1 += ((theta0 + theta1 * x_train[i])-y_train[i]) ** 2 / 2
    #判断是否已收敛
    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1
# 画图表现
plt.title('BGD')
plt.plot(range(len(retn0)),retn0,label='theta0')
plt.plot(range(len(retn1)),retn1,label='theta1')
plt.legend()          #显示上面的label
plt.xlabel('time')
plt.ylabel('theta')
plt.show()
plt.plot(x_train,y_train,'bo')
plt.plot(x_train,[h(x) for x in x_train],color='k',label='BGD')
plt.legend()
plt.xlabel('area')
plt.ylabel('price')
print("批量梯度下降法：theta0={},theta1={}".format(theta0,theta1))
print("批量梯度下降法循环次数：{}".format(cnt))
plt.show()
```

### SGD

**随机梯度下降法**不同于批量梯度下降，随机梯度下降是**每次迭代**使用**一个样本**来对参数进行更新。使得训练速度加快。

对于一个样本的目标函数：
$$
J^i\left( \theta_{0},\theta_{1} \right) =\frac{1}{2 }\left( h_{\theta} \left( x^i \right) -y^i\right)^2
$$

1. 对目标函数求偏导：
$$
 \frac{\Delta J^i\left( \theta_{0},\theta_{1} \right)}{\theta j}=\left( h_{\theta}\left( x^i \right) -y^i \right) x_{j}^i
$$

2. 参数更新：
$$
\theta j:=\theta j-\alpha\left( h_{\theta}\left( x^i \right) -y^i \right) x_{j}^i
$$

这里不再有求和符号。

**优点**：   
（1）由于不是在全部训练数据上的损失函数，而是在每轮迭代中，随机优化某一条训练数据上的损失函数，这样每一轮参数的更新速度大大加快。
（2）在梯度上引入了随机噪声，在非凸优化问题中更**容易逃离局部最小值**

**缺点**：   
（1）准确度下降。即使在目标函数为强凸函数的情况下，SGD仍旧无法做到线性收敛。 
（2）不易于并行实现。

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240826223120.png)

```python
import matplotlib.pyplot as plt
import random
##样本数据
x_train = [150,200,250,300,350,400,600]
y_train = [6450,7450,8450,9450,11450,15450,18450]
#样本个数
m = len(x_train)
#步长
alpha = 0.00001
#循环次数
cnt = 0
#假设函数为 y=theta0+theta1*x
def h(x):
    return theta0 + theta1*x
theta0 = 0
theta1 = 0
#导数
diff0=0
diff1=0
#误差
error0=0           
error1=0          
#每次迭代theta的值
retn0 = []         
retn1 = []         

#退出迭代的条件
epsilon=0.00001

#随机梯度下降
for i in range(1000):
    cnt=cnt+1
    diff0=0
    diff1=0
    j = random.randint(0, m - 1)
    diff0=h(x_train[j])-y_train[j]
    diff1=(h(x_train[j])-y_train[j])*x_train[j]
    theta0=theta0-alpha/m*diff0
    theta1=theta1-alpha/m*diff1
    retn0.append(theta0)
    retn1.append(theta1)
    error1=0
    #计算迭代的误差
    for i in range(len(x_train)):
        error1 += ((theta0 + theta1 * x_train[i])-y_train[i]) ** 2 / 2
    #判断是否已收敛
    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1
# 画图表现        
plt.title('SGD')
plt.plot(range(len(retn0)),retn0,label='theta0')
plt.plot(range(len(retn1)),retn1,label='theta1')
plt.legend()          #显示上面的label
plt.xlabel('time')
plt.ylabel('theta')
plt.show()
plt.plot(x_train,y_train,'bo')
plt.plot(x_train,[h(x) for x in x_train],color='k',label='SGD')
plt.legend()
plt.xlabel('area')
plt.ylabel('price')
print("随机梯度下降法：theta0={},theta1={}".format(theta0,theta1))
print("随机梯度下降法循环次数：{}".format(cnt))
plt.show()
```

### MBGD小批量梯度下降

小批量梯度下降算法是折中方案，选取训练集中一个**小批量样本**（一般是2的倍数，如32，64，128等）计算，这样可以保证训练过程更稳定，而且采用批量训练方法也可以利用[矩阵计算](https://zhida.zhihu.com/search?q=%E7%9F%A9%E9%98%B5%E8%AE%A1%E7%AE%97&zhida_source=entity&is_preview=1)的优势。这是目前最常用的梯度下降算法。

> 小批量梯度下降是批量梯度下降与随机梯度下降之间的一个折中，即经过一个小批量的训练数据更新一次参数，可以保证网络训练速度不太慢，也能使训练方向不至于偏离太多，具有一定稳定性。当使用小批量梯度下降时，通常也使用SGD这个术语。[深度学习随笔——优化算法( SGD、BGD、MBGD、Momentum、NAG、Adagrad、RMSProp、AdaDelta、Adam、NAdam) - Lu1zero9的文章 - 知乎](https://zhuanlan.zhihu.com/p/588536166)

**batcha_size的选择带来的影响：**   

（1）在合理地范围内，增大batch_size的好处：    
a. 内存利用率提高了，大矩阵乘法的并行化效率提高。    
b. 跑完一次 epoch（全数据集）所需的迭代次数减少，对于相同数据量的处理速度进一步加快。    
c. 在一定范围内，一般来说 Batch_Size 越大，其确定的下降方向越准，引起训练震荡越小。   

（2）盲目增大batch_size的坏处：    
a. 内存利用率提高了，但是内存容量可能撑不住了。    
b. 跑完一次 epoch（全数据集）所需的迭代次数减少，要想达到相同的精度，其所花费的时间大大增加了，从而对参数的修正也就显得更加缓慢。    
c. Batch_Size 增大到一定程度，其确定的下降方向已经基本不再变化。

### 时间成本

尽管批量梯度下降每次都是遍历所有数据，但是由于存在并行计算，花费的时间不一定比小批量时间长
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240826223527.png)
发现1-1000的batch大小对于更新时间是没有影响的；只有超过了一定的阈值，才会时间变长。

### 小批量的好处：更容易走到盆地

局部最小值有好最小值跟坏最小值之分，如果局部最小值在一个“峡谷”里面，它是坏的最小值；如果局部最小值在一个平原上，它是好的最小值。

训练的损失跟测试的损失函数是不一样的，这有两种可能。
- 一种可能是本来**训练跟测试的分布**就不一样；
- 另一种可能是因为训练跟测试都是从**采样的数据**算出来的，训练跟测试采样到的数据可能不一样，所以它们计算出的损失是有一点差距。

对在一个“盆地”里面的最小值，其在训练跟测试上面的结果不会差太多，只差了一点点。但对在右边在“峡谷”里面的最小值，一差就可以天差地远 。虽然它在训练集上的损失很低，但训练跟测试之间的损失函数不一样，因此测试时，损失函数一变，计算出的损失就变得很大。

大批量倾向走到峡谷中，小批量倾向走到盆地中。

## 动量

$$
\begin{gather}
\mathbf{m_{0}}=0\\
\mathbf{m_{1}}=-η\mathbf{g_{0}}\\
\mathbf{m_{2}}=-\lambdaη\mathbf{g_{0}}-η\mathbf{g_{1}}\\
\vdots
\end{gather}
$$

引入动量后，每次在移动参数的时候，不是只往梯度的反方向来移动参数，而是根据梯度的反方向加上前一步移动的方向决定移动方向。 

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240826225530.png)

一般梯度下降走到一个局部最小值或鞍点时，就被困住了。但有动量还是有办法继续走下去，因为动量不是只看梯度，还看**前一步的方向**。即使梯度方向往左走，但如果前一步的影响力比梯度要大，球还是有可能继续往右走，甚至翻过一个小丘，也许可以走到更好的局部最小值，这就是动量有可能带来的好处 。

该算法通过对梯度的一阶矩进行估计，使得梯度可以在横向上累积，在纵向上相互抵消，从而使更新震荡幅度减小，加快更新的速度。

# 自适应学习率

有时候损失不下降，不一定是卡在临界点，而是在山谷的谷壁之间来回震荡，需要多次调整才可能出去

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240827115558.png)

实际上在训练的时候，要走到鞍点或局部最小值，是一件困难的事情。一般的梯度下降，其实是做不到的。用一般的梯度下降训练，往往会在梯度还很大的时候，损失就已经降了下去，这个是需要特别方法训练的。要走到一个临界点其实是比较困难的，多数时候训练在还没有走到临界点的时候就已经停止了。

> 举个例子，我们有两个参数 $w$ 和 $b$，这两个参数值不一样的时候，损失值也不一样，得到了误差表面，该误差表面的最低点在叉号处。事实上，该误差表面是凸的形状。凸的误差表面的等高线是椭圆形的，椭圆的长轴非常长，短轴相比之下比较短，其在横轴的方向梯度非常小，坡度的变化非常小，非常平坦；其在纵轴的方向梯度变化非常大，误差表面的坡度非常陡峭。现在我们要从黑点（初始点）来做梯度下降。
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240827120801.png)

a图是 $\eta=10^{-2}$ 的误差表面，发现因为太大了所以一直在震荡；但当把 $\eta$ 设为 $10^{-7}$ 时，参数会到山谷，但因为坡度非常平坦到不了终点

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240827120825.png)

为了解决这些问题，提出了自适应学习率

##  AdaGrad

AdaGrad 可以做到梯度比较大的时候，学习率就减小，梯度比较小的时候，学习率就放大。

梯度下降更新某个参数 $θ^i_{t}$ 的过程为

$$
\mathbf{\theta_{t+1}^i \leftarrow \mathbf{\theta_{t}^i} - \frac{\eta}{\sigma_{0}^i}\mathbf{g_{t}^i}}
$$

其中 $\mathbf{\theta_{0}^i}$ 为初始化参数。而 $\sigma_{0}^i$ 的计算过程为：

$$
\sigma_{0}^i=\sqrt{ \left( \mathbf{g_{0}^i} \right) ^2 }=|\mathbf{g_{0}^i|}
$$

将初值带入得到 $\frac{\mathbf{g_{0}^i}}{\sigma_{0}^i}$ 的值是+1或-1，此时与梯度大小无关。但后面的参数更新则有变化：
第 $t+1$ 次更新参数的时候，即

$$
\mathbf{\theta_{t+1}^i}\leftarrow \mathbf{\theta_{i}^i}-\frac{\eta}{\sigma_{t}^i}\mathbf{g_{t}^i}, \quad \sigma_{t}^i=\sqrt{ \frac{1}{t+1}\sum_{i=0}^{t}\left( \mathbf{g_{t}^i} \right) ^2 }
$$

当梯度大的时候，分母 $\sigma_{t}^i$ 就会变大，此时学习率变小；反之，当梯度小的时候，学习率就会变大，从而实现根据梯度调节学习率

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240827123518.png)
## RMSProp
>RMSprop 没有论文，Geoffrey Hinton 在 Coursera 上开过深度学习的课程，他在他的课程里面讲了 RMSprop，如果要引用，需要引用对应视频的链接。

同一个参数的同个方向，学习率也是需要动态调整的。RMSProp可以通过一个参数$\alpha$控制梯度的重要性，从而更灵活调整

$$
\mathbf{\theta_{t+1}^i}\leftarrow \mathbf{\theta_{i}^i}-\frac{\eta}{\sigma_{t}^i}\mathbf{g_{t}^i}, \quad \sigma_{t}^i=\sqrt{ \alpha \left( \sigma_{t-1}^i \right) ^2+(1-\alpha)\left( \mathbf{g_{t}^i} \right) ^2 }
$$

在 RMSprop 里面，可以自己调整现在的这个梯度的重要性。如果 α 设很小趋近于 0，代表 $g_{t}^i$ 相较于之前算出来的梯度而言，比较重要；如果 α 设很大趋近于 1，代表 $g_{t}^i$ 比较不重要，之前算出来的梯度比较重要。

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240827124225.png)
## Adam

最常用也最方便，可以看作是RMSProp加上动量，使用动量作为参数更新方向，同时自适应调整学习率。

$$
\mathbf{\theta_{t+1}^i}\leftarrow \mathbf{\theta_{i}^i}-\frac{\eta}{\sigma_{t}^i}\mathbf{m_{t}^i}, \quad \sigma_{t}^i=\sqrt{ \alpha \left( \sigma_{t-1}^i \right) ^2+(1-\alpha)\left( \mathbf{g_{t}^i} \right) ^2 }
$$

在PyTorch中，调用Adam优化器非常简单。以下是使用Adam优化器的基本步骤：
1. **导入必要的库**：
   首先，你需要导入PyTorch及其子模块torch.optim。

   ```python
   import torch
   from torch.optim import Adam
   ```

2. **定义模型**：
   你需要有一个PyTorch模型，该模型的参数将由Adam优化器进行优化。

   ```python
   class MyModel(nn.Module):
       def __init__(self):
           super(MyModel, self).__init__()
           # 定义模型的层
           self.conv1 = nn.Conv2d(1, 20, 5)
           self.pool = nn.MaxPool2d(2, 2)
           # 其他层...

       def forward(self, x):
           # 定义前向传播
           x = self.pool(F.relu(self.conv1(x)))
           # 其他操作...
           return x

   model = MyModel()
   ```

3. **选择优化器**：
   使用`torch.optim.Adam`创建一个优化器实例，传入模型的参数和学习率等参数。

   ```python
   optimizer = Adam(model.parameters(), lr=0.001)
   ```

   这里`model.parameters()`返回模型的所有参数，`lr`是学习率($\alpha$通常取默认值)。另外可能还有个权重衰减率`weight decay`负责控制模型复杂度
4. **定义损失函数**：
   选择一个损失函数，例如交叉熵损失或均方误差损失。

   ```python
   criterion = nn.CrossEntropyLoss()  # 举例
   ```
   
5. **训练模型**：
   在训练循环中，使用优化器来更新模型的参数。

   ```python
   for epoch in range(num_epochs):
       for data, target in train_loader:  # 假设train_loader是数据加载器
           # 前向传播
           output = model(data)
           loss = criterion(output, target)

           # 反向传播和优化
           optimizer.zero_grad()  # 清空梯度
           loss.backward()       # 反向传播，计算梯度
           optimizer.step()      # 更新参数
   ```

6. **保存模型**：
   训练完成后，你可以保存模型的参数。

   ```python
   torch.save(model.state_dict(), 'model.pth')
   ```

## 学习率调度

防止梯度爆炸问题

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240827124738.png)

之前的学习率调整方法中 $η$ 是一个固定的值，而在学习率调度中 $η$ 跟时间有关

### 学习率衰减

随着参数的不断更新，让 $η$ 越来越小

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240827124842.png)
![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240827124922.png)

### 预热

预热的方法是让学习率先变大后变小，至于变到多大、变大的速度、变小的速度是超参数。

# 分类

回归是输入一个向量 $x$ ，输出 $\hat{y}$，我们希望 $\hat{y}$ 跟某一个标签 $y$ 越接近越好，$y$ 是要学习的目标。而分类可当作回归来看，输入 $x$ 后，输出仍然是一个标量 $\hat{y}$，要让它跟正确答案的那个类越接近越好。

通常用**独热编码**表示类

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240829114844.png)

分类实际过程是：输入 $x$ ，乘上 $W$ ，加上 $b$ ，通过激活函数 $σ$ ，乘上$W'$ ，再加上 $b'$ 得到向量 $\hat{y}$。但实际做分类的时候，往往会把 $\hat{y}$ 通过 softmax 函数得到 $\hat{y}$，才去计算 $\hat{y}$ 跟 $y'$ 之间的距离。

softmax会将**结果进行归一化**，同时**使得大的值和小的值的差距变大**

$$
y'=\frac{\exp(y_{i})}{\sum_{j}^{}\exp(y_{i})}
$$

其中，$1>y_{i}'>0, \sum_{i}^{}y_{i}'=1$

## 损失函数

对于分类问题，通常采用交叉熵作为损失函数：

$$
e=-\sum_{i}^{}y_{i}\ln y_{i}'
$$

[分类问题为什么用交叉熵损失不用 MSE 损失_为什么分类问题不能使用mse损失函数-CSDN博客](https://blog.csdn.net/wxc971231/article/details/123866413)
具体来说，有三个原因：
- 概率角度看，优化MSE损失等价于高斯分布的最大似然估计，而优化交叉熵损失等价于多项式分布的最大似然
- 梯度角度看，对于多分类问题，用MSE损失，参数梯度关于绝对误差是凹函数形式，更新强度与绝对误差值不成正比，优化效果低效；用交叉熵损失，参数梯度关于绝对误差是线性函数形式，更新强度与绝对误差值成正比，优化过程高效稳定
- 直观角度看，MSE无差别地关注全部类别上预测概率和真实概率的差；交叉熵关注正确类别的预测概率

![image.png](https://cdn.jsdelivr.net/gh/BlackJack0083/image@main/img/20240829120336.png)


**参考资料：**
- An overview of gradient descent optimization algorithms

# CNN 代码精读

```shell
git clone https://www.modelscope.cn/datasets/Datawhale/LeeDL-HW3-CNN.git
```

先看到Datawhale的介绍：
卷积神经网络是深度学习中的一个非常重要的分支，本作业提供了进行图像分类任务的基本范式。
- 准备数据
- 训练模型
- 应用模型

要完成一个深度神经网络训练模型的代码，大概需要完成下面的内容：
1. 导入所需要的库/工具包
2. 数据准备与预处理
3. 定义模型
4. 定义损失函数和优化器等其他配置
5. 训练模型
6. 评估模型
7. 进行预测
此范式不仅适用于图像分类任务，对于广泛的深度学习任务也是适用的。

### 导入必要的库

```Python
# 导入必要的库
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# “ConcatDataset” 和 “Subset” 在进行半监督学习时可能是有用的。
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# 这个是用来显示进度条的。
from tqdm.auto import tqdm
import random
```

设置随机种子，配置CUDA

```Python
# 设置随机种子以确保实验结果的可重复性
myseed = 6666

# 确保在使用CUDA时，卷积运算具有确定性，以增强实验结果的可重复性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 为numpy和pytorch设置随机种子
np.random.seed(myseed)
torch.manual_seed(myseed)

# 如果使用CUDA，为所有GPU设置随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
```

`torch.backends.cudnn.benchmark` 是 PyTorch 深度学习框架中的一个参数，它与 CUDA 神经网络库（cuDNN）相关。cuDNN 是 NVIDIA 提供的一个 GPU 加速的深度神经网络库，它允许 PyTorch 利用 NVIDIA 的 GPU 进行快速的神经网络运算。

`torch.backends.cudnn.benchmark` 参数的作用是：
- 当设置为 `True` 时，cuDNN 会尝试寻找最优的算法来执行每个操作，这可能会增加模型的初始化时间，但可以提高模型运行时的效率。
- 当设置为 `False` 时，cuDNN 会使用一个默认的算法来执行操作，这可能会减少初始化时间，但可能不会达到最优的运行效率。

通常，在训练大型模型或进行多次迭代时，使用 `torch.backends.cudnn.benchmark=True` 可能会带来性能上的提升。然而，如果模型较小或者只运行一次，开启这个参数可能不会带来明显的好处，反而可能会因为初始化时间的增加而导致总体运行时间变长。

### 数据准备与预处理

数据准备包括从指定路径加载图像数据，并对其进行预处理。作业中对图像的预处理操作包括调整大小和将图像转换为Tensor格式。

Torchvision为图像预处理、数据增强和数据加载提供了一系列的API，这些API可以方便的实现图像预处理、数据增强和数据加载。  

具体不同的详细操作可以通过[Pytorch的官方文档](https://pytorch.org/vision/stable/transforms.html)查看。

为了增强模型的鲁棒性，可以对训练集进行数据增强。相关代码如下：

```Python
# 在测试和验证阶段，通常不需要图像增强。
# 我们所需要的只是调整PIL图像的大小并将其转换为Tensor。
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 不过，在测试阶段使用图像增强也是有可能的。
# 你可以使用train_tfm生成多种图像，然后使用集成方法进行测试。
train_tfm = transforms.Compose([
    # 将图像调整为固定大小（高度和宽度均为128）
    transforms.Resize((128, 128)),
    # TODO：你可以在这里添加一些图像增强的操作。

    # ToTensor()应该是所有变换中的最后一个。
    transforms.ToTensor(),
])
```

### 数据集

数据通过名称进行标记，因此在调用'getitem'时我们同时加载图像和标签。  

定义了一个名为 FoodDataset 的类，继承自 Dataset，用于加载并预处理食品图像数据集，支持图像变换及从文件名中提取标签。

```Python
class FoodDataset(Dataset):
    """
    用于加载食品图像数据集的类。

    该类继承自Dataset，提供了对食品图像数据集的加载和预处理功能。
    它可以自动从指定路径加载所有的jpg图像，并对这些图像应用给定的变换。
    """

    def __init__(self, path, tfm=test_tfm, files=None):
        """
        初始化FoodDataset实例。

        参数:
        - path: 图像数据所在的目录路径。
        - tfm: 应用于图像的变换方法（默认为测试变换）。
        - files: 可选参数，用于直接指定图像文件的路径列表（默认为None）。
        """
        super(FoodDataset).__init__()
        self.path = path
        # 列出目录下所有jpg文件，并按顺序排序
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files is not None:
            self.files = files  # 如果提供了文件列表，则使用该列表
        self.transform = tfm  # 图像变换方法

    def __len__(self):
        """
        返回数据集中图像的数量。

        返回:
        - 数据集中的图像数量。
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        获取给定索引的图像及其标签。

        参数:
        - idx: 图像在数据集中的索引。

        返回:
        - im: 应用了变换后的图像。
        - label: 图像对应的标签（如果可用）。
        """
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)  # 应用图像变换

        # 尝试从文件名中提取标签
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # 如果无法提取标签，则设置为-1（测试数据无标签）

        return im, label
```

### 模型定义

这段代码定义了一个图像分类器类（`Classifier`），继承自PyTorch的`nn.Module`。该分类器通过一系列卷积层、批归一化层、激活函数和池化层构建卷积神经网络（CNN），用于提取图像特征。随后，这些特征被输入到全连接层进行分类，最终输出11个类别的概率，用于图像分类任务。

```Python
class Classifier(nn.Module):
    """
    定义一个图像分类器类，继承自PyTorch的nn.Module。
    该分类器包含卷积层和全连接层，用于对图像进行分类。
    """
    def __init__(self):
        """
        初始化函数，构建卷积神经网络的结构。
        包含一系列的卷积层、批归一化层、激活函数和池化层。
        """
        super(Classifier, self).__init__()
        # 定义卷积神经网络的序列结构
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # 输入通道3，输出通道64，卷积核大小3，步长1，填充1
            nn.BatchNorm2d(64),        # 批归一化，作用于64个通道
            nn.ReLU(),                 # ReLU激活函数
            nn.MaxPool2d(2, 2, 0),      # 最大池化，池化窗口大小2，步长2，填充0
            
            nn.Conv2d(64, 128, 3, 1, 1), # 输入通道64，输出通道128，卷积核大小3，步长1，填充1
            nn.BatchNorm2d(128),        # 批归一化，作用于128个通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # 最大池化，池化窗口大小2，步长2，填充0
            
            nn.Conv2d(128, 256, 3, 1, 1), # 输入通道128，输出通道256，卷积核大小3，步长1，填充1
            nn.BatchNorm2d(256),        # 批归一化，作用于256个通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # 最大池化，池化窗口大小2，步长2，填充0
            
            nn.Conv2d(256, 512, 3, 1, 1), # 输入通道256，输出通道512，卷积核大小3，步长1，填充1
            nn.BatchNorm2d(512),        # 批归一化，作用于512个通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # 最大池化，池化窗口大小2，步长2，填充0
            
            nn.Conv2d(512, 512, 3, 1, 1), # 输入通道512，输出通道512，卷积核大小3，步长1，填充1
            nn.BatchNorm2d(512),        # 批归一化，作用于512个通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # 最大池化，池化窗口大小2，步长2，填充0
        )
        # 定义全连接神经网络的序列结构
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),    # 输入大小512*4*4，输出大小1024
            nn.ReLU(),
            nn.Linear(1024, 512),        # 输入大小1024，输出大小512
            nn.ReLU(),
            nn.Linear(512, 11)           # 输入大小512，输出大小11，最终输出11个类别的概率
        )

    def forward(self, x):
        """
        前向传播函数，对输入进行处理。
        
        参数:
        x -- 输入的图像数据，形状为(batch_size, 3, 128, 128)
        
        返回:
        输出的分类结果，形状为(batch_size, 11)
        """
        out = self.cnn(x)               # 通过卷积神经网络处理输入
        out = out.view(out.size()[0], -1)  # 展平输出，以适配全连接层的输入要求
        return self.fc(out)             # 通过全连接神经网络得到最终输出
```

### 定义损失函数和优化器等其他配置

这段代码实现了图像分类模型的初始化和训练配置，目的是准备好训练环境和参数。它选择合适的设备（GPU或CPU），设置模型、批量大小、训练轮数、提前停止策略，定义了损失函数和优化器，为后续的模型训练奠定了基础。

```Python
# 根据GPU是否可用选择设备类型
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化模型，并将其放置在指定的设备上
model = Classifier().to(device)

# 定义批量大小
batch_size = 64

# 定义训练轮数
n_epochs = 8

# 如果在'patience'轮中没有改进，则提前停止
patience = 5

# 对于分类任务，我们使用交叉熵作为性能衡量标准
criterion = nn.CrossEntropyLoss()

# 初始化优化器，您可以自行调整一些超参数，如学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
```

### 加载数据

```Python
# 构建训练和验证数据集
# "loader" 参数定义了torchvision如何读取数据
train_set = FoodDataset("./hw3_data/train", tfm=train_tfm)
# 创建训练数据加载器，设置批量大小、是否打乱数据顺序、是否使用多线程加载以及是否固定内存地址
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
# 构建验证数据集
# "loader" 参数定义了torchvision如何读取数据
valid_set = FoodDataset("./hw3_data/valid", tfm=test_tfm)
# 创建验证数据加载器，设置批量大小、是否打乱数据顺序、是否使用多线程加载以及是否固定内存地址
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
```

### 训练与验证模型

这段代码实现了一个图像分类模型的训练和验证循环，目的是通过多轮训练（epochs）逐步优化模型的参数，以提高其在验证集上的性能，并保存效果最好的模型。训练阶段通过前向传播、计算损失、反向传播和参数更新来优化模型，验证阶段评估模型在未见过的数据上的表现。如果验证集的准确率超过了之前的最好成绩，保存当前模型，并在连续多轮验证性能未提升时提前停止训练。

训练完成后，需要在测试集上评估模型的性能。通过计算准确率来衡量模型在测试集上的表现。

```Python
# 初始化追踪器，这些不是参数，不应该被更改
stale = 0
best_acc = 0

for epoch in range(n_epochs):
    # ---------- 训练阶段 ----------
    # 确保模型处于训练模式
    model.train()

    # 这些用于记录训练过程中的信息
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        # 每个批次包含图像数据及其对应的标签
        imgs, labels = batch
        # imgs = imgs.half()
        # print(imgs.shape,labels.shape)

        # 前向传播数据。（确保数据和模型位于同一设备上）
        logits = model(imgs.to(device))

        # 计算交叉熵损失。
        # 在计算交叉熵之前不需要应用softmax，因为它会自动完成。
        loss = criterion(logits, labels.to(device))

        # 清除上一步中参数中存储的梯度
        optimizer.zero_grad()

        # 计算参数的梯度
        loss.backward()

        # 为了稳定训练，限制梯度范数
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # 使用计算出的梯度更新参数
        optimizer.step()

        # 计算当前批次的准确率
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # 记录损失和准确率
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # 打印信息
    print(f"[ 训练 | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

# ---------- 验证阶段 ----------
    # 确保模型处于评估模式，以便某些模块如dropout能够正常工作
    model.eval()

    # 这些用于记录验证过程中的信息
    valid_loss = []
    valid_accs = []

    # 按批次迭代验证集
    for batch in tqdm(valid_loader):
        # 每个批次包含图像数据及其对应的标签
        imgs, labels = batch
        # imgs = imgs.half()

        # 我们在验证阶段不需要梯度。
        # 使用 torch.no_grad() 加速前向传播过程。
        with torch.no_grad():
            logits = model(imgs.to(device))

        # 我们仍然可以计算损失（但不计算梯度）。
        loss = criterion(logits, labels.to(device))

        # 计算当前批次的准确率
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # 记录损失和准确率
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        # break

    # 整个验证集的平均损失和准确率是所记录值的平均
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # 打印信息
    print(f"[ 验证 | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # 更新日志
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ 验证 | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> 最佳")
    else:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ 验证 | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # 保存模型
    if valid_acc > best_acc:
        print(f"在第 {epoch} 轮找到最佳模型，正在保存模型")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")  # 只保存最佳模型以防止输出内存超出错误
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"连续 {patience} 轮没有改进，提前停止")
            break
```

### 进行预测

最后的代码构建一个测试数据集和数据加载器，以便高效地读取数据。实例化并加载预训练的分类器模型，并将其设置为评估模式。在不计算梯度的情况下，遍历测试数据，使用模型进行预测，并将预测标签存储在列表中。将预测结果与测试集的ID生成一个DataFrame，并将其保存为`submission.csv`文件。

```Python
# 构建测试数据集
# "loader"参数指定了torchvision如何读取数据
test_set = FoodDataset("./hw3_data/test", tfm=test_tfm)
# 创建测试数据加载器，批量大小为batch_size，不打乱数据顺序，不使用多线程，启用pin_memory以提高数据加载效率
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# 实例化分类器模型，并将其转移到指定的设备上
model_best = Classifier().to(device)

# 加载模型的最优状态字典
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))

# 将模型设置为评估模式
model_best.eval()

# 初始化一个空列表，用于存储所有预测标签
prediction = []

# 使用torch.no_grad()上下文管理器，禁用梯度计算
with torch.no_grad():
    # 遍历测试数据加载器
    for data, _ in tqdm(test_loader):
        # 将数据转移到指定设备上，并获得模型的预测结果
        test_pred = model_best(data.to(device))
        # 选择具有最高分数的类别作为预测标签
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        # 将预测标签添加到结果列表中
        prediction += test_label.squeeze().tolist()

# 创建测试csv文件
def pad4(i):
    """
    将输入数字i转换为长度为4的字符串，如果长度不足4，则在前面补0。
    :param i: 需要转换的数字
    :return: 补0后的字符串
    """
    return "0" * (4 - len(str(i))) + str(i)

# 创建一个空的DataFrame对象
df = pd.DataFrame()
# 使用列表推导式生成Id列，列表长度等于测试集的长度
df["Id"] = [pad4(i) for i in range(len(test_set))]
# 将预测结果赋值给Category列
df["Category"] = prediction
# 将DataFrame对象保存为submission.csv文件，不保存索引
df.to_csv("submission.csv", index=False)        
```
