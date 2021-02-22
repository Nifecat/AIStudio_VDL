# 本文将结合paddle2.0，在中石油图像分类实战项目中介绍VisualDL的用法，以及一些小Trick

**（前记：如果对VisualDL工具使用还不熟的，请前往VDL初体验哦~扔个链接[https://aistudio.baidu.com/aistudio/projectdetail/1230514](http://)）**

## 项目背景

### （1）中石油图像分类项目要求设计一个抽油机工况数据的分类模型，可准确的识别抽油机当前的工作状态，提高抽油机井工况诊断准确率。图像共有A01~A12一共12个类别

训练集文件名称：train.zip，包含17087个样本数据，每个样本数据包含1个csv格式原始数据以及1张对应的png格式图片。训练集包含12类典型工况：


| 序号 | 类别名 | 
| -------- | -------- | 
| A01     | 工作正常     | 
| A02     | 供液不足     | 
| A03     | 气体影响     | 
| A04     | 气锁     | 
| A05     | 上碰泵     | 
| A06     | 下碰泵     | 
| A07     | 游动阀关闭迟缓     | 
| A08     | 柱塞脱出泵工作筒     | 
| A09     | 游动阀漏     | 
| A10     | 固定阀漏     | 
| A11     | 砂影响+供液不足     | 
| A12     | 惯性影响+工作正常     | 


### （2）Paddle2.0

PaddlePaddle (PArallel Distributed Deep LEarning)是一个易用、高效、灵活、可扩展的深度学习框架。

Paddle2.0引入了动态图机制，使得模型的定义、训练、评估等过程变得更加简易

### （3）VisualDL

飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、数据样本、模型结构、PR曲线、高维数据分布等。帮助用户清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型调优。

欢迎大家去VDL Github点star鸭！！

github首页：https://github.com/PaddlePaddle/VisualDL

官网：https://www.paddlepaddle.org.cn/paddle/visualdl

aistudio项目： https://aistudio.baidu.com/aistudio/projectdetail/502834

https://aistudio.baidu.com/aistudio/projectdetail/622772

aistudio论坛：https://ai.baidu.com/forum/topic/show/960053?pageNo=2


## 1 数据处理
**本节将主要分为解压数据包、轮廓填充、数据扩充与划分、以及样本处理前后对比四个部分**
### 1.1 数据处理，解压数据包



```python
!unzip -d data/data57195 data/data57195/train.zip
!unzip -d data/data57195 data/data57195/test.zip
```

### 先导入一些包


```python

import os
import zipfile
import random
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import paddle
import paddle.fluid as fluid
from visualdl import LogWriter
from paddle.fluid.dygraph import Linear,Conv2D,Pool2D
from paddle.static import InputSpec
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
```

### 1.2 数据处理，将轮廓填充，保存为原名.jpg格式


```python
root_path = 'data/data57195/train/'
class_list = os.listdir(root_path)
for ci in class_list:
    pi = root_path + ci +'/'
    s = os.listdir(pi)
    for si in s:
        if si[-1] == 'g':
            sp = pi + si
            png_image = cv2.imread(sp, cv2.IMREAD_UNCHANGED)
            gray_image = png_image[:, :, 3]
            gray_image = cv2.UMat(gray_image).get()
            indexs = np.argwhere(gray_image==255)
            polygon_points = indexs[:, ::-1]
            gray_image = cv2.fillPoly(img=gray_image, pts=[polygon_points], color=(255,255,255), lineType=cv2.LINE_4)
            cv2.imwrite(sp[:-4] + '.jpg', gray_image)

root_path = 'data/data57195/test/'
class_list = os.listdir(root_path)
for ci in class_list:
    pi = root_path + ci
    if pi[-1] == 'g':
        png_image = cv2.imread(pi, cv2.IMREAD_UNCHANGED)
        gray_image = png_image[:, :, 3]
        gray_image = cv2.UMat(gray_image).get()
        indexs = np.argwhere(gray_image==255)
        polygon_points = indexs[:, ::-1]
        gray_image = cv2.fillPoly(img=gray_image, pts=[polygon_points], color=(255,255,255), lineType=cv2.LINE_4)
        cv2.imwrite(pi[:-4] + '.jpg', gray_image)
```

### 1.3 数据处理，划分训练集和测试集，数据平衡，将所有类别数据都扩充至7500条



```python
train_root = 'data/data57195/train/'
class_list = os.listdir(train_root)
print(class_list)
train_list = []
test_list = []

class_num_list = [0 for i in range(12)]
train_class_list = [[] for i in range(12)]
all_class_list = [[] for i in range(12)]

flag = 1
for ci in class_list:
    c_path = train_root + ci + '/'
    sample_list = os.listdir(c_path)
    for si in sample_list:
        if si[-3:]=='jpg':
            item = c_path + si + ' ' + str(int(ci[1:])-1) + '\n'
            all_class_list[int(ci[1:])-1].append(item)
for i in range(12):
    all_class_list[i] = sorted(all_class_list[i])
    for j in range(int(len(all_class_list[i])*0.9)):
        train_list.append(all_class_list[i][j])    
        train_class_list[i].append(all_class_list[i][j])
        class_num_list[i] += 1
    for j in range(int(len(all_class_list[i])*0.9), len(all_class_list[i])):
        test_list.append(all_class_list[i][j])


for i in range(12):
    if class_num_list[i] < 7500:
        for j in range(class_num_list[i], 7500):
            r = random.randint(0, class_num_list[i]-1)
            train_list.append(train_class_list[i][r])

print(np.shape(train_list))
print(np.shape(test_list))
# exit(0)

random.shuffle(train_list)
with open('data/test.txt', 'w+') as f1:
    f1.writelines(test_list)
with open('data/train.txt', 'w+') as f2:
    f2.writelines(train_list)


```

    ['A10', 'A09', 'A07', 'A02', 'A05', 'A01', 'A12', 'A11', 'A03', 'A04', 'A06', 'A08']
    (90000,)
    (1715,)


### 1.4 处理前后样本可视化对比
### Trick 1 使用VisualDL的Image组件,可视化对比样本处理前后的变化，可以帮助我们更好的认识样本数据的特征
**如下图所示，右边为处理前的图片，可见只有一根线具有色彩（白色），而左边图片为处理后的图片，可见右图白色边框所围住的部分全部标为白色了。**

![](https://ai-studio-static-online.cdn.bcebos.com/ffd494b1e3e5431cb26ab11fa91ff12f935d95a49c8046aab7e09be0e5546de0)




```python
with LogWriter(logdir="./log/处理前样本") as writer:
    for i in range(12):
        for j in range(10):
            img_path, tp = all_class_list[i][j].split()
            img_path = img_path[:-3] + 'png'
            png_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            # print(np.shape(png_image[:, :, 2]))
            # png_image = np.reshape(png_image, (1, 4, 190, 400))
            # png_image=png_image.transpose(2, 0, 1)
            # png_image = np.reshape(png_image, (1, 4, 190, 400)) 
            # print(np.shape(png_image))
            # break
            # png_image = png_image[:,:,3]
            # plt.imshow(png_image)
            # plt.show()
            gray_image = png_image[:, :, 3]
            writer.add_image(tag="A" + str(int(tp)+1), img=gray_image, step=i, dataformats="HW")

with LogWriter(logdir="./log/处理后样本") as writer:
    for i in range(12):
        for j in range(10):
            img_path, tp = all_class_list[i][j].split()
            png_image = cv2.imread(img_path)
            writer.add_image(tag="A" + str(int(tp)+1), img=png_image, step=i)


```

## 2 训练准备
### 2.1 训练准备，训练参数定义


```python
train_parameters = {
    "input_size": [1, 190, 400],                        #输入图片的shape
    "class_dim": -1,                                    #分类数
    "src_path":"",                                      #原始数据集路径
    "target_path":"/home/aistudio/data/dataset",        #要解压的路径 
    "train_list_path": "data/train.txt",                #train_data.txt路径
    "eval_list_path": "data/test.txt",                  #eval_data.txt路径
    "label_dict":{},                                    #标签字典
    "readme_path": "/home/aistudio/data/readme.json",   #readme.json路径
    "num_epochs": 2,                                    #训练轮数
    "train_batch_size": 128,                            #批次的大小
    "learning_strategy": {                              #优化函数相关的配置
        "lr": 0.0001                                    #超参数学习率
    } 
}
```

### 2.2 训练准备，定义数据读取方法


```python
def data_reader(file_list):
    '''
    自定义data_reader
    '''
    def reader():
        with open(file_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                # print(line)
                img_path, lab = line.split()
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.array(img).astype('float32')
                img = img/255.0
                yield img, int(lab) 
    return reader

src_path=train_parameters['src_path']
target_path=train_parameters['target_path']
train_list_path=train_parameters['train_list_path']
eval_list_path=train_parameters['eval_list_path']
batch_size=train_parameters['train_batch_size']

dr = data_reader(train_list_path)
a = next(dr())[0]
print(np.shape(a))

train_reader = paddle.batch(data_reader(train_list_path),
                            batch_size=batch_size,
                            drop_last=True)
eval_reader = paddle.batch(data_reader(eval_list_path),
                            batch_size=batch_size,
                            drop_last=True)
```

    (190, 400)


### 2.3 训练准备，定义模型结构


```python
class MyCNN01(fluid.dygraph.Layer):
    def __init__(self):
        super(MyCNN01,self).__init__()
        self.hidden1_1 = Conv2D(1,32,3,2,1,act='relu') #通道数、卷积核个数、卷积核大小
        self.hidden1_2 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,pool_padding=1)
        self.hidden2_1 = Conv2D(32,32,3,2,1,act='relu')
        self.hidden2_2 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,pool_padding=1)
        self.hidden3 = Conv2D(32,64,3,2,1,act='relu')
        self.hidden4 = Linear(64*7*13,12,act='softmax')
    def forward(self,input):
        # print(input.shape)
        x = self.hidden1_1(input)
        # print(x.shape)
        x = self.hidden1_2(x)
        # print(x.shape)
        x = self.hidden2_1(x)
        # print(x.shape)
        x = self.hidden2_2(x)
        # print(x.shape)
        x = self.hidden3(x)
        # print(x.shape)
        x = fluid.layers.reshape(x, shape=[-1, 64*7*13])
        y = self.hidden4(x)
        return y
```

## 3 模型训练与过程可视化


### Trick 2 使用VisualDL的Scalar组件，可视化loss和accuracy动态变化过程，可以帮助我们实时观察到模型的loss和accuaracy什么时候已经达到平衡，这样我们可以提前结束训练，避免过拟合。

**注意：模型在训练的过程中，就可以观察标量的动态变化过程。而且网页会自动刷新，很方便！**

![](https://ai-studio-static-online.cdn.bcebos.com/601e449825ff4c3f9ed90f4ba2203c078b2a955f0ca743a794215a7ba5806cf7)


### Trick 3 使用VisualDL的Histogram组件，可视化参数的变化趋势，如果有急速变化，那么说明训练过程中可能出了问题，则需要再检查检查咯。
**如下图所示，可见最后w0大致分布在以0为均值的高斯分布上，说明训练正常**

![](https://ai-studio-static-online.cdn.bcebos.com/dd0737555f58412b8482614d581161fbdb649aaae45f4fa48e27a6b6b652cebc)


### Trick 4 使用VisualDL的PR Curve组件，可视化精度与召回率的权衡分析，可以帮助我们清晰直观了解模型训练效果，便于分析模型是否达到理想标准
**如下图所示，可见每个类别PR曲线下面积都比较大，说明训练的准确率还不错。**

![](https://ai-studio-static-online.cdn.bcebos.com/281d58d1bc084f328183017090c78d912c0f002d76e0429b9ff94302d937f121)



```python


with LogWriter(logdir="./log/train_CNN01") as writer:
    # 如果不是用的GPU环境的话记得把下面一行代码括号里的参数去掉哦~
    with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
        model=MyCNN01() #模型实例化
        model.train() #训练模式
        opt=fluid.optimizer.SGDOptimizer(learning_rate=train_parameters['learning_strategy']['lr'], parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
        epochs_num=train_parameters['num_epochs'] #迭代次数
        param = model.parameters()
        # print(param)
        params_name = [param.name for param in model.parameters()]
    
        step = 0
        for pass_num in range(epochs_num):
            for batch_id,data in enumerate(train_reader()):
                images=np.array([x[0].reshape(1,190,400) for x in data],np.float32)
                labels = np.array([x[1] for x in data]).astype('int64')
                labels = labels[:, np.newaxis]
                image=fluid.dygraph.to_variable(images)
                label=fluid.dygraph.to_variable(labels)

                predict=model(image) #数据传入model
                pred = predict.numpy()
            
                loss=fluid.layers.cross_entropy(predict,label)
                avg_loss=fluid.layers.mean(loss)#获取loss值
            
                acc=fluid.layers.accuracy(predict,label)#计算精度
            
                if batch_id!=0 and batch_id%50==0:
                    step += 1
                    # Scalar组件可视化loss和accuracy动态变化过程
                    writer.add_scalar(tag="MyLeNet_train/loss", step=step, value=avg_loss.numpy())
                    writer.add_scalar(tag="MyLeNet_train/acc", step=step, value=acc.numpy())
                    # histogram组件
                    for name_i in range(len(params_name)):
                        writer.add_histogram(tag=params_name[name_i], values=param[name_i].numpy(), step=step)
                    # PR Curve组件
                    labels = np.reshape(labels, (len(labels),))
                    for i in range(12):
                        label_i = np.array(labels == i, dtype='int32')
                        prediction_i = pred[:, i]
                        # print(np.shape(label_i), np.shape(prediction_i))
                        writer.add_pr_curve(tag='train/class_{}_pr_curve'.format(i),
                            labels=label_i,
                            predictions=prediction_i,
                            step=step,
                            num_thresholds=20)
                    print("train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num,batch_id,avg_loss.numpy(),acc.numpy()))
            
                avg_loss.backward()       
                opt.minimize(avg_loss)    #优化器对象的minimize方法对参数进行更新 
                model.clear_gradients()   #model.clear_gradients()来重置梯度
    
        

```

    train_pass:0,batch_id:50,train_loss:[2.3845434],train_acc:[0.1640625]
    train_pass:0,batch_id:100,train_loss:[2.116953],train_acc:[0.265625]
    train_pass:0,batch_id:150,train_loss:[1.9750732],train_acc:[0.4453125]
    train_pass:0,batch_id:200,train_loss:[1.7398998],train_acc:[0.5625]
    train_pass:0,batch_id:250,train_loss:[1.6180012],train_acc:[0.6171875]
    train_pass:0,batch_id:300,train_loss:[1.4743811],train_acc:[0.7265625]
    train_pass:0,batch_id:350,train_loss:[1.272366],train_acc:[0.765625]
    train_pass:0,batch_id:400,train_loss:[1.1991317],train_acc:[0.78125]
    train_pass:0,batch_id:450,train_loss:[1.0422437],train_acc:[0.84375]
    train_pass:0,batch_id:500,train_loss:[0.9389738],train_acc:[0.8203125]
    train_pass:0,batch_id:550,train_loss:[0.9362001],train_acc:[0.8359375]
    train_pass:0,batch_id:600,train_loss:[0.81270075],train_acc:[0.875]
    train_pass:0,batch_id:650,train_loss:[0.7238138],train_acc:[0.8671875]
    train_pass:0,batch_id:700,train_loss:[0.6938138],train_acc:[0.8984375]
    train_pass:1,batch_id:50,train_loss:[0.6137237],train_acc:[0.8671875]
    train_pass:1,batch_id:100,train_loss:[0.59991986],train_acc:[0.859375]
    train_pass:1,batch_id:150,train_loss:[0.64017385],train_acc:[0.859375]
    train_pass:1,batch_id:200,train_loss:[0.45869493],train_acc:[0.921875]
    train_pass:1,batch_id:250,train_loss:[0.47885025],train_acc:[0.9140625]
    train_pass:1,batch_id:300,train_loss:[0.4793214],train_acc:[0.8984375]
    train_pass:1,batch_id:350,train_loss:[0.3751875],train_acc:[0.9453125]
    train_pass:1,batch_id:400,train_loss:[0.42931342],train_acc:[0.90625]
    train_pass:1,batch_id:450,train_loss:[0.35743988],train_acc:[0.9453125]
    train_pass:1,batch_id:500,train_loss:[0.35655344],train_acc:[0.921875]
    train_pass:1,batch_id:550,train_loss:[0.36681372],train_acc:[0.921875]
    train_pass:1,batch_id:600,train_loss:[0.38574028],train_acc:[0.8828125]
    train_pass:1,batch_id:650,train_loss:[0.29585928],train_acc:[0.9453125]
    train_pass:1,batch_id:700,train_loss:[0.309414],train_acc:[0.953125]


## 4 动态图模型的保存


### Trick 5 使用VisualDL的Graph组件，查看模型结构、模型属性、节点信息、节点输入输出等，并进行节点搜索，协助我们快速分析模型结构与了解数据流向。
**如下图所示，可视化网络模型结构，且可以清晰看到数据流，和各参数的详细信息**

![](https://ai-studio-static-online.cdn.bcebos.com/a53c9e8261f4402b9518ef3737e93983641b0752cd294d078d2d8aea16289c2f)



```python
paddle.jit.save(layer = model,
                path='MyCNN01',
                input_spec=[InputSpec(shape=[None,1,190,400],
                dtype='float32')])
```

## 5 模型加载与评估


```python

with fluid.dygraph.guard():
    accs = []
    model2 = paddle.jit.load('MyCNN01')
    model2.eval() #测试模式
    for batch_id,data in enumerate(eval_reader()):#测试集
        images=np.array([x[0].reshape(1,190,400) for x in data],np.float32)
        labels = np.array([x[1] for x in data]).astype('int64')
        labels = labels[:, np.newaxis]
        image=fluid.dygraph.to_variable(images)
        label=fluid.dygraph.to_variable(labels)       
        predict=model2(image)
        # print(predict)
        acc=fluid.layers.accuracy(predict,label)
        accs.append(acc.numpy()[0])
        avg_acc = np.mean(accs)
    print(avg_acc)

```

    0.7487981


### Trick6 对比不同模型的训练准确率和损失值
**通过VisualDL的Scale组件，将不同模型（例如本文中的CNN01, CNN02, CNN03）标量保存到同一个目录下，分别保存为CNN01,CNN02,CNN03。注意：记录时要将tag保持一致哦，这样才能在一张图上同时展示3个模型的标量动态变化过程。如下图所示，绿色代表CNN02模型，可以看出，CNN02采用sigmoid作为激活函数，1->128->64->64，效果并不好。蓝色和紫色分别代表CNN01和CNN03，可以看出，紫色loss曲线始终处于蓝色loss曲线之下，且紫色acc曲线始终处于蓝色acc曲线之上。这说明，CNN03相对于CNN01改用了平均池化，增加卷积核对于训练是有效的。**

![](https://ai-studio-static-online.cdn.bcebos.com/dd4e70508f4e4386b5e30009735306216e47c22e825347708b5dc5f8f8f365cb)



```python
class MyCNN02(fluid.dygraph.Layer):
    def __init__(self):
        super(MyCNN02,self).__init__()
        self.hidden1_1 = Conv2D(1,128,3,2,1,act='sigmoid') #通道数、卷积核个数、卷积核大小
        self.hidden1_2 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,pool_padding=1)
        self.hidden2_1 = Conv2D(128,64,3,2,1,act='sigmoid')
        self.hidden2_2 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,pool_padding=1)
        self.hidden3 = Conv2D(64,64,3,2,1,act='relu')
        self.hidden4 = Linear(64*7*13,12,act='softmax')
    def forward(self,input):
        # print(input.shape)
        x = self.hidden1_1(input)
        # print(x.shape)
        x = self.hidden1_2(x)
        # print(x.shape)
        x = self.hidden2_1(x)
        # print(x.shape)
        x = self.hidden2_2(x)
        # print(x.shape)
        x = self.hidden3(x)
        # print(x.shape)
        x = fluid.layers.reshape(x, shape=[-1, 64*7*13])
        y = self.hidden4(x)
        return y

class MyCNN03(fluid.dygraph.Layer):
    def __init__(self):
        super(MyCNN03,self).__init__()
        self.hidden1_1 = Conv2D(1,64,3,2,1,act='relu') #通道数、卷积核个数、卷积核大小
        self.hidden1_2 = Pool2D(pool_size=2,pool_type='mean',pool_stride=2,pool_padding=1)
        self.hidden2_1 = Conv2D(64,64,3,2,1,act='relu')
        self.hidden2_2 = Pool2D(pool_size=2,pool_type='mean',pool_stride=2,pool_padding=1)
        self.hidden3 = Conv2D(64,64,3,2,1,act='relu')
        self.hidden4 = Linear(64*7*13,12,act='softmax')
    def forward(self,input):
        # print(input.shape)
        x = self.hidden1_1(input)
        # print(x.shape)
        x = self.hidden1_2(x)
        # print(x.shape)
        x = self.hidden2_1(x)
        # print(x.shape)
        x = self.hidden2_2(x)
        # print(x.shape)
        x = self.hidden3(x)
        # print(x.shape)
        x = fluid.layers.reshape(x, shape=[-1, 64*7*13])
        y = self.hidden4(x)
        return y

with LogWriter(logdir="./log/contrast_model/cnn01") as writer:
    # 如果不是用的GPU环境的话记得把下面一行代码括号里的参数去掉哦~
    with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
        model=MyCNN01() #模型实例化
        model.train() #训练模式
        opt=fluid.optimizer.SGDOptimizer(learning_rate=train_parameters['learning_strategy']['lr'], parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
        epochs_num=train_parameters['num_epochs'] #迭代次数
        param = model.parameters()
        # print(param)
        params_name = [param.name for param in model.parameters()]
        step = 0
        for pass_num in range(epochs_num):
            for batch_id,data in enumerate(train_reader()):
                images=np.array([x[0].reshape(1,190,400) for x in data],np.float32)
                labels = np.array([x[1] for x in data]).astype('int64')
                labels = labels[:, np.newaxis]
                image=fluid.dygraph.to_variable(images)
                label=fluid.dygraph.to_variable(labels)
                predict=model(image) #数据传入model
                pred = predict.numpy()
                loss=fluid.layers.cross_entropy(predict,label)
                avg_loss=fluid.layers.mean(loss)#获取loss值
                acc=fluid.layers.accuracy(predict,label)#计算精度
                if batch_id!=0 and batch_id%50==0:
                    step += 1
                    # Scalar组件可视化loss和accuracy动态变化过程
                    writer.add_scalar(tag="model/loss", step=step, value=avg_loss.numpy())
                    writer.add_scalar(tag="model/acc", step=step, value=acc.numpy())
                avg_loss.backward()       
                opt.minimize(avg_loss)    #优化器对象的minimize方法对参数进行更新 
                model.clear_gradients()   #model.clear_gradients()来重置梯度
with LogWriter(logdir="./log/contrast_model/cnn02") as writer:
    # 如果不是用的GPU环境的话记得把下面一行代码括号里的参数去掉哦~
    with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
        model=MyCNN02() #模型实例化
        model.train() #训练模式
        opt=fluid.optimizer.SGDOptimizer(learning_rate=train_parameters['learning_strategy']['lr'], parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
        epochs_num=train_parameters['num_epochs'] #迭代次数
        param = model.parameters()
        # print(param)
        params_name = [param.name for param in model.parameters()]
        step = 0
        for pass_num in range(epochs_num):
            for batch_id,data in enumerate(train_reader()):
                images=np.array([x[0].reshape(1,190,400) for x in data],np.float32)
                labels = np.array([x[1] for x in data]).astype('int64')
                labels = labels[:, np.newaxis]
                image=fluid.dygraph.to_variable(images)
                label=fluid.dygraph.to_variable(labels)
                predict=model(image) #数据传入model
                pred = predict.numpy()
                loss=fluid.layers.cross_entropy(predict,label)
                avg_loss=fluid.layers.mean(loss)#获取loss值
                acc=fluid.layers.accuracy(predict,label)#计算精度
                if batch_id!=0 and batch_id%50==0:
                    step += 1
                    # Scalar组件可视化loss和accuracy动态变化过程
                    writer.add_scalar(tag="model/loss", step=step, value=avg_loss.numpy())
                    writer.add_scalar(tag="model/acc", step=step, value=acc.numpy())
                avg_loss.backward()       
                opt.minimize(avg_loss)    #优化器对象的minimize方法对参数进行更新 
                model.clear_gradients()   #model.clear_gradients()来重置梯度
with LogWriter(logdir="./log/contrast_model/cnn03") as writer:
    # 如果不是用的GPU环境的话记得把下面一行代码括号里的参数去掉哦~
    with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
        model=MyCNN03() #模型实例化
        model.train() #训练模式
        opt=fluid.optimizer.SGDOptimizer(learning_rate=train_parameters['learning_strategy']['lr'], parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
        epochs_num=train_parameters['num_epochs'] #迭代次数
        param = model.parameters()
        # print(param)
        params_name = [param.name for param in model.parameters()]
        step = 0
        for pass_num in range(epochs_num):
            for batch_id,data in enumerate(train_reader()):
                images=np.array([x[0].reshape(1,190,400) for x in data],np.float32)
                labels = np.array([x[1] for x in data]).astype('int64')
                labels = labels[:, np.newaxis]
                image=fluid.dygraph.to_variable(images)
                label=fluid.dygraph.to_variable(labels)
                predict=model(image) #数据传入model
                pred = predict.numpy()
                loss=fluid.layers.cross_entropy(predict,label)
                avg_loss=fluid.layers.mean(loss)#获取loss值
                acc=fluid.layers.accuracy(predict,label)#计算精度
                if batch_id!=0 and batch_id%50==0:
                    step += 1
                    # Scalar组件可视化loss和accuracy动态变化过程
                    writer.add_scalar(tag="model/loss", step=step, value=avg_loss.numpy())
                    writer.add_scalar(tag="model/acc", step=step, value=acc.numpy())
                avg_loss.backward()       
                opt.minimize(avg_loss)    #优化器对象的minimize方法对参数进行更新 
                model.clear_gradients()   #model.clear_gradients()来重置梯度
```

### Trick7 对比相同模型不同参数的训练准确率和损失值

writer1=LogWriter('./log/constrast_lr/lr0.01')

writer2=LogWriter('./log/constrast_lr/lr0.001')

writer3=LogWriter('./log/constrast_lr/lr0.005')

**通过VisualDL的Scale组件，将不同参数（例如本文中的lr = 0.01, lr = 0.005, lr = 0.001）标量保存到同一个目录下，分别保存为lr0.01, lr0.005, lr0.001。注意：记录时要将tag保持一致哦，这样才能在一张图上同时展示3个模型的标量动态变化过程。如下图所示，蓝色，绿色，红色曲线分别对应着0.01，0.005，0.001。显然，acc曲线蓝色高于绿色高于红色，这说明学习率越高，acc上升的趋势越快。同理，loss曲线蓝色低于绿色低于红色，说明学习率越高，loss下降越快。不过两张图最终三条曲线都趋近相等。**

![](https://ai-studio-static-online.cdn.bcebos.com/0dadb20e29be4162a409e73fed922ef7ab5bc8b72a2c4f2bb9b651bf48c61726)





```python
writer1=LogWriter('./log/constrast_lr/lr0.01')
writer2=LogWriter('./log/constrast_lr/lr0.005')
writer3=LogWriter('./log/constrast_lr/lr0.001')

for lr in [0.01, 0.005, 0.001]:
    step = 0
    # 如果不是用的GPU环境的话记得把下面一行代码括号里的参数去掉哦~
    with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
        model=MyCNN03() #模型实例化
        model.train() #训练模式
        opt=fluid.optimizer.SGDOptimizer(learning_rate=lr, parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
        epochs_num=train_parameters['num_epochs'] #迭代次数
        param = model.parameters()
        # print(param)
        params_name = [param.name for param in model.parameters()]
        for pass_num in range(epochs_num):
            for batch_id,data in enumerate(train_reader()):
                images=np.array([x[0].reshape(1,190,400) for x in data],np.float32)
                labels = np.array([x[1] for x in data]).astype('int64')
                labels = labels[:, np.newaxis]
                image=fluid.dygraph.to_variable(images)
                label=fluid.dygraph.to_variable(labels)
                predict=model(image) #数据传入model
                pred = predict.numpy()
                loss=fluid.layers.cross_entropy(predict,label)
                avg_loss=fluid.layers.mean(loss)#获取loss值
                acc=fluid.layers.accuracy(predict,label)#计算精度
                if batch_id!=0 and batch_id%50==0:
                    step += 1
                    # Scalar组件可视化loss和accuracy动态变化过程
                    if lr == 0.01:
                        writer1.add_scalar(tag="model_lr/loss", step=step, value=avg_loss.numpy())
                        writer1.add_scalar(tag="model_lr/acc", step=step, value=acc.numpy())
                    elif lr == 0.005:
                        writer2.add_scalar(tag="model_lr/loss", step=step, value=avg_loss.numpy())
                        writer2.add_scalar(tag="model_lr/acc", step=step, value=acc.numpy())
                    else:
                        writer3.add_scalar(tag="model_lr/loss", step=step, value=avg_loss.numpy())
                        writer3.add_scalar(tag="model_lr/acc", step=step, value=acc.numpy())
                avg_loss.backward()       
                opt.minimize(avg_loss)    #优化器对象的minimize方法对参数进行更新 
                model.clear_gradients()   #model.clear_gradients()来重置梯度
```

### 再来宣传一波：
### 飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、数据样本、模型结构、PR曲线、高维数据分布等。帮助用户清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型调优。
### 欢迎大家去VDL Github点star鸭！！
### github首页：[https://github.com/PaddlePaddle/VisualDL](http://)
### 官网：[https://www.paddlepaddle.org.cn/paddle/visualdl](http://)
### aistudio项目： [https://aistudio.baidu.com/aistudio/projectdetail/502834](http://)
###                        [ https://aistudio.baidu.com/aistudio/projectdetail/622772](http://)
### aistudio论坛：[https://ai.baidu.com/forum/topic/show/960053?pageNo=2](http://)

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
