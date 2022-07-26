## CNN网络
(1) CNN的基本原理，公式，代码，其中设计到矩阵的乘法和优化

① 感受野 ，权值共享
② CNN具体计算的细节，Group Conv，可分离卷积之间的原理
③ 矩阵乘法原理及其优化，caffe底层实现的源码
④ 反卷积，simple baseline中提及了
⑤ 上采样方式和下采样

(2）神经网络初期存在的问题，梯度爆炸和梯度消失的原因及其解决方法，以及原理

① 梯度不稳定，梯度消失，梯度保障->resnet->v1,v2
② 参数初始化
③ Normalization操作：BN，IN，GN，LN，SN
④ Dropout
⑤ 常见的激活函数及其原理
⑥ 正则化

(3)参数的量化操作

① 参数的计算和量化
② 常用模型压缩方法
③ 常用轻量型网络及其设计思想，以及代码
④ 知识蒸馏的原理及其思路
⑤ 常用的移动框架了解及其对比

(4) 深度学习常用的梯度函数

① 常用的优化函数
② tf和pytorch构建网络的差异
③ 常用的训练策略，warming up，step learing

(5) 常用的loss function及其原理

① 分类的loss function及其原理
② pose的常用loss function及其原理

(6) 常用的网络结构及其源码

hourglass
fpn：
cpn：
fpn：
mobilenet：
inception：
resnet

(7) 常用的数据预处理方式

① opencv的基本操作
② 常用的数据增强方式
③ mix up

(8) 常有数据后处理方法

① 极大值法 ② 翻转测试 ③ 多线程和多尺度测试

(9) python的常见问题

① 线程和进程，GIL ② 内存管理 ③ 元组与list,set的区别 ④ os与sys区别

(10) pytorch

① nn.module和functional的区别 ② pytorch 分布式原理 RingAllReduce原理

分布式使用

③ pytorch如何构建动态图原理 ④ 梯度计算，反传，更新

(11) 常见的优化器

其他整理的链接：包括了机器学习，数学

(1) https://github.com/scutan90/DeepLearning-500-questions (2) http://www.huaxiaozhuan.com/

额外补充的知识：

(1) 概率论 (2) 线性代数，神经网络的推导
