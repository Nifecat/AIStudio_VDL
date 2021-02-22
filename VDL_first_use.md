# 本文将结合paddle2.0，在MNIST项目中介绍VisualDL的用法
## 项目背景
（1）MNIST项目是对手写体数字进行分类的项目，共有0~9一共10个类别

MNIST 数据集可在 [http://yann.lecun.com/exdb/mnist/](http://) 获取, 它包含了四个部分:

* Training set images: train-images-idx3-ubyte.gz (包含 60,000 个样本)

* Training set labels: train-labels-idx1-ubyte.gz (包含 60,000 个标签)

* Test set images: t10k-images-idx3-ubyte.gz (包含 10,000 个样本)

* Test set labels: t10k-labels-idx1-ubyte.gz (10,000 个标签)

（2）Paddle2.0

PaddlePaddle (PArallel Distributed Deep LEarning)是一个易用、高效、灵活、可扩展的深度学习框架。

Paddle2.0引入了动态图机制，使得模型的定义、训练、评估等过程变得更加简易

（3）VisualDL

飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、数据样本、模型结构、PR曲线、高维数据分布等。帮助用户清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型调优。

欢迎大家去VDL Github点star鸭！！

github首页：[https://github.com/PaddlePaddle/VisualDL](http://)

官网：[https://www.paddlepaddle.org.cn/paddle/visualdl](http://)

aistudio项目： [https://aistudio.baidu.com/aistudio/projectdetail/502834](http://)

[https://aistudio.baidu.com/aistudio/projectdetail/622772](http://)

aistudio论坛：[https://ai.baidu.com/forum/topic/show/960053?pageNo=2](http://)

## 本文将按照模型训练的全过程应用VisualDL,介绍了VisualDL的用法以及意义
## 主要包括：
* 1. Image组件 -------------------------------------Step1.数据处理
* 2. Scalar组件-------------------------------|
* 3. PR Curve组件---------------------------|-------Step2.模型训练
* 4. Histogram组件--------------------------|
* 5. Graph组件--------------------------------------Step3.模型保存

----------------------------------------------------------Step4.模型评估

## 第一步，数据处理。
### 我们将直接调用paddle.vision.datasets.MNIST接口获取训练集和测试集，此阶段，VDL的Image组件可以有效帮助我们可视化输入数据，可以更有效了解我们要分类的对象。Image组件的详细用法如下所示：

### Image组件
* 作用 显示图片，可显示输入图片和处理后的结果，便于查看中间过程的变化
* 接口 add_image(tag, img, step, walltime=None, dataformats="HWC")

![](https://ai-studio-static-online.cdn.bcebos.com/d3c8caac798d4d40863e5e732d9d1a6989d4923e47304443ac681e847de48d61)




```python
import paddle
import numpy as np
import paddle.fluid as fluid
from paddle.metric import Accuracy
from visualdl import LogWriter
print(paddle.__version__)

train_dataset = paddle.vision.datasets.MNIST(mode='train')
val_dataset =  paddle.vision.datasets.MNIST(mode='test')
print(np.shape(train_dataset.images))
print(np.shape(train_dataset.labels))

train_images = train_dataset.images
train_labels = train_dataset.labels
val_images = val_dataset.images
val_labels = val_dataset.labels

# Image组件，样本图片可视化，显示图片，可显示输入图片和处理后的结果，便于查看中间过程的变化
# 可视化训练集前50张图片
with LogWriter(logdir="./log/input_image_show") as writer:
    for i in range(50):
        pic_i = train_images[i]
        pic_i = np.reshape(pic_i, (28,28,1))

        # 添加一个图片数据
        writer.add_image(tag="数字" + str(train_labels[i][0]), img=pic_i, step=i)

```

    2.0.0-rc0
    (60000, 784)
    (60000, 1)


**查看图像的界面如下图所示，我们还可以调整亮度和对比度**

![](https://ai-studio-static-online.cdn.bcebos.com/962764a02fc241489ed45e53758c9558bae5d7f8c62942d3b408d7fde0ee3f4b)


## 定义一个简单的DNN模型，summary查看网络结构


```python
import paddle
class MyDNN05(paddle.nn.Layer):
    def __init__(self):
        super(MyDNN05, self).__init__()
        self.fc1 = paddle.nn.Linear(784, 64)
        self.sig1 = paddle.nn.Sigmoid()
        self.fc2 = paddle.nn.Linear(64, 10)
        self.soft = paddle.nn.Softmax() 

    def forward(self, x):
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc1(x)
        x = self.sig1(x)
        x = self.fc2(x)
        x = self.soft(x)
        return x

my_layer = MyDNN05()
model = paddle.Model(my_layer)
model.summary((1,28,28))


```

    ---------------------------------------------------------------------------
     Layer (type)       Input Shape          Output Shape         Param #    
    ===========================================================================
       Linear-1          [[1, 784]]            [1, 64]            50,240     
       Sigmoid-1         [[1, 64]]             [1, 64]               0       
       Linear-2          [[1, 64]]             [1, 10]              650      
       Softmax-1         [[1, 10]]             [1, 10]               0       
    ===========================================================================
    Total params: 50,890
    Trainable params: 50,890
    Non-trainable params: 0
    ---------------------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.00
    Params size (MB): 0.19
    Estimated Total Size (MB): 0.20
    ---------------------------------------------------------------------------
    





    {'total_params': 50890, 'trainable_params': 50890}



## 第二步，模型训练。
### 在这一阶段我们直接采用paddle2.0的model.prepare配置模型的优化器、损失函数和评估指标。然后直接用model.fit训练模型。这里我们可以采用callback机制结合VisualDL动态展示函数值和准确率等标量数据。Scalar组件的详细用法如下所示：

## Scalar组件
* 作用 动态展示损失函数值、准确率等标量数据(训练开始时就可以打开VDL进行查看哦，不需要等待训练结束，有兴趣的小伙伴可以加一个控制按钮，看到趋势差不多了直接一键停止训练)
* 接口 add_scalar(tag, value, step, walltime=None)

![](https://ai-studio-static-online.cdn.bcebos.com/664cb2d822f34b92919b332ad39479c5d95dbe1660194f888d0459502842a817)



```python
# 配置模型
model.prepare(
    paddle.optimizer.Adam(0.0001, parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )
# callback机制结合VDL
callback = paddle.callbacks.VisualDL(log_dir='./log/train/visualdl_log_dir')
# 训练模型
model.fit(train_dataset,
        epochs=20,
        batch_size=64,
        verbose=1,
        callbacks=callback
        )
```

    Epoch 1/20
    step  70/938 [=>............................] - loss: 2.2963 - acc: 0.1665 - ETA: 4s - 5ms/ste

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return (isinstance(seq, collections.Sequence) and


    step 120/938 [==>...........................] - loss: 2.2683 - acc: 0.1880 - ETA: 3s - 5ms/stepstep 938/938 [==============================] - loss: 1.8546 - acc: 0.4925 - 4ms/step         
    Epoch 2/20
    step 938/938 [==============================] - loss: 1.7041 - acc: 0.7131 - 3ms/step         
    Epoch 3/20
    step 938/938 [==============================] - loss: 1.6571 - acc: 0.7768 - 3ms/step         
    Epoch 4/20
    step 938/938 [==============================] - loss: 1.6027 - acc: 0.8090 - 3ms/step         
    Epoch 5/20
    step 938/938 [==============================] - loss: 1.6758 - acc: 0.8172 - 3ms/step         
    Epoch 6/20
    step 938/938 [==============================] - loss: 1.6054 - acc: 0.8344 - 3ms/step         
    Epoch 7/20
    step 938/938 [==============================] - loss: 1.6557 - acc: 0.8971 - 3ms/step         
    Epoch 8/20
    step 938/938 [==============================] - loss: 1.5058 - acc: 0.9070 - 3ms/step        
    Epoch 9/20
    step 938/938 [==============================] - loss: 1.5088 - acc: 0.9136 - 3ms/step         
    Epoch 10/20
    step 938/938 [==============================] - loss: 1.5658 - acc: 0.9176 - 3ms/step         
    Epoch 11/20
    step 938/938 [==============================] - loss: 1.5918 - acc: 0.9213 - 3ms/step         
    Epoch 12/20
    step 938/938 [==============================] - loss: 1.5098 - acc: 0.9250 - 3ms/step         
    Epoch 13/20
    step 938/938 [==============================] - loss: 1.5912 - acc: 0.9260 - 3ms/step         
    Epoch 14/20
    step 938/938 [==============================] - loss: 1.4826 - acc: 0.9290 - 3ms/step         
    Epoch 15/20
    step 938/938 [==============================] - loss: 1.5093 - acc: 0.9308 - 3ms/step         
    Epoch 16/20
    step 938/938 [==============================] - loss: 1.5531 - acc: 0.9327 - 3ms/step         
    Epoch 17/20
    step 938/938 [==============================] - loss: 1.5062 - acc: 0.9356 - 3ms/step         
    Epoch 18/20
    step 938/938 [==============================] - loss: 1.4946 - acc: 0.9364 - 4ms/step         
    Epoch 19/20
    step 938/938 [==============================] - loss: 1.4917 - acc: 0.9385 - 3ms/step         
    Epoch 20/20
    step 938/938 [==============================] - loss: 1.5011 - acc: 0.9395 - 3ms/step         


**可视化结构如下图所示**
![](https://ai-studio-static-online.cdn.bcebos.com/49119a1d9d6d4733bbb7723b6ef2fc95b4297df9ddba40e7b20e487e94e3cb2e)


## 第三步,保存模型
### 对paddle2.0动态图模型的保存需要使用paddle.jit.save，如下代码所示。
### 保存之后会生成.pdmodel的文件，此时可以利用VDL的Graph组件可视化网络结构。这样可以帮助到我们更详细更直观了解到网络结构，以及tensor运算流的细节。

## Graph组件
* 用于查看模型属性、节点信息、节点输入输出等，并进行节点搜索，协助开发者们快速分析模型结构与了解数据流向。
* 支持模型格式 PaddlePaddle、ONNX、Keras、Core ML、Caffe、Caffe2、Darknet、MXNet、ncnn、TensorFlow Lite
* 实验性支持模型格式 TorchScript、PyTorch、Torch、 ArmNN、BigDL、Chainer、CNTK、Deeplearning4j、MediaPipe、ML.NET、MNN、OpenVINO、Scikit-learn、Tengine、TensorFlow.js、TensorFlow

### 动态图保存模型，后缀：.pdmodel


```python
from paddle.static import InputSpec
paddle.jit.save(layer = my_layer,
                path='MyDNN05',
                input_spec=[InputSpec(shape=[None,1,28,28],dtype='float32')])

```

**网络结构可视化如下图所示**
![](https://ai-studio-static-online.cdn.bcebos.com/f0b146d050154635b3fa3371ca786bc76a441d6fc01c49bf8b686e4bca3025e1)


## 第四步，评估模型。
### 直接采用model.evaluate即可进行模型评估


```python
model.evaluate(val_dataset, verbose=0)
```

## 另一种训练方法，可以记录更多的数据。需要重启容器，运行完第一块加载数据的代码后直接从这开始运行

## 第一步，数据处理。
### 需要准备好train_reader和eval_reader


```python
def data_reader(train_X, train_Y):
    '''
    自定义data_reader
    '''
    def reader():
        for i in range(len(train_X)):
            img = train_X[i]
            img = np.reshape(img,(1,28,28))
            lab = train_Y[i]
            yield img, int(lab) 
    return reader

train_reader = paddle.batch(data_reader(train_images, train_labels),
                            batch_size=32,
                            drop_last=True)
eval_reader = paddle.batch(data_reader(val_images, val_labels),
                            batch_size=32,
                            drop_last=True)


```

## 模型定义


```python
class MyDNN06(fluid.dygraph.Layer):
    def __init__(self):
        super(MyDNN06, self).__init__()
        self.fc1 = paddle.nn.Linear(784, 64)
        self.sig1 = paddle.nn.Sigmoid()
        self.fc2 = paddle.nn.Linear(64, 10)
        self.soft = paddle.nn.Softmax() 

    def forward(self, x):
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc1(x)
        x = self.sig1(x)
        x = self.fc2(x)
        x = self.soft(x)
        return x


```

## 第二步，模型训练
### 在模型训练的过程中，VDL的Histogram组件可以帮助我们动态展示训练过程中某个tensor的分布，这样我们可以更好的掌握权值或者梯度的变化。
### 在模型训练的过程中，VDL的PR Curve组件可以帮助我们展示每一轮训练后精度与召回率之间的关系，这样我们可以更好的观察到每一轮训练后模型的好坏
### Histogram组件和PR Curve组件的详细使用方法如下所示

## Histogram组件
* 作用 展示训练过程中权重、梯度等张量的分布
* 接口 add_histogram(tag, values, step, walltime=None, buckets=10)

![](https://ai-studio-static-online.cdn.bcebos.com/ba9dd079c66e4d0990e13bbb4ba613a98d2adca1627d4b66a49e80bb50949098)


## PR Curve组件
* 作用 权衡精度与召回率之间的平衡关系
* 接口 add_pr_curve(tag, labels, predictions, step=None, num_thresholds=10)

![](https://ai-studio-static-online.cdn.bcebos.com/9837ff698aa74575a73e631a07e5e7344902a841d01040fcbd838c11a83c99f1)



```python
from paddle.static import InputSpec
Batch = 0
with LogWriter(logdir="./log/train") as writer:
    with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
        model02=MyDNN06() #模型实例化
        model02.train() #训练模式
        opt=fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model02.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
        param = model02.parameters()
        # print(param)
        params_name = [param.name for param in model02.parameters()]

        print(params_name)
        # print(len(param), len(params_name))
        # print(param[1].numpy())

        for pass_num in range(5):
            for batch_id,data in enumerate(train_reader()):
                images=np.array([x[0] for x in data],np.float32)
                labels = np.array([x[1] for x in data]).astype('int64')
                labels = labels[:, np.newaxis]
                image=fluid.dygraph.to_variable(images)
                label=fluid.dygraph.to_variable(labels)

                predict=model02(image) #数据传入model
                pred = predict.numpy()
                # print(np.shape(pred))
                
                loss=fluid.layers.cross_entropy(predict,label)
                avg_loss=fluid.layers.mean(loss)#获取loss值
                
                acc=fluid.layers.accuracy(predict,label)#计算精度
                
                if batch_id!=0 and batch_id%50==0:
                    Batch += 1
                    # scalar组件
                    writer.add_scalar(tag="train/loss", step=Batch, value=avg_loss)
                    writer.add_scalar(tag="train/acc", step=Batch, value=acc)
                    print("train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num,batch_id,avg_loss.numpy(),acc.numpy()))
                    # histogram组件
                    for name_i in range(len(params_name)):
                        writer.add_histogram(tag=params_name[name_i], values=param[name_i].numpy(), step=Batch)
                    # PR Curve组件
                    labels = np.reshape(labels, (len(labels),))
                    for i in range(10):
                        label_i = np.array(labels == i, dtype='int32')
                        prediction_i = pred[:, i]
                        # print(np.shape(label_i), np.shape(prediction_i))
                        writer.add_pr_curve(tag='train/class_{}_pr_curve'.format(i),
                            labels=label_i,
                            predictions=prediction_i,
                            step=Batch,
                            num_thresholds=20)

                avg_loss.backward()       
                opt.minimize(avg_loss)    #优化器对象的minimize方法对参数进行更新 
                model02.clear_gradients()   #model.clear_gradients()来重置梯度

        paddle.jit.save(layer = model02,
                        path='MyDNN06',
                        input_spec=[InputSpec(shape=[None,1,28,28],dtype='float32')])

```

    ['linear_0.w_0', 'linear_0.b_0', 'linear_1.w_0', 'linear_1.b_0']
    train_pass:0,batch_id:50,train_loss:[2.5030417],train_acc:[0.15625]
    train_pass:0,batch_id:100,train_loss:[2.1248226],train_acc:[0.25]
    train_pass:0,batch_id:150,train_loss:[2.1999388],train_acc:[0.15625]
    train_pass:0,batch_id:200,train_loss:[1.9983119],train_acc:[0.34375]
    train_pass:0,batch_id:250,train_loss:[2.085506],train_acc:[0.34375]
    train_pass:0,batch_id:300,train_loss:[1.8808961],train_acc:[0.5]
    train_pass:0,batch_id:350,train_loss:[1.6846015],train_acc:[0.53125]
    train_pass:0,batch_id:400,train_loss:[1.7853789],train_acc:[0.4375]
    train_pass:0,batch_id:450,train_loss:[1.6185603],train_acc:[0.59375]
    train_pass:0,batch_id:500,train_loss:[1.9320799],train_acc:[0.3125]
    train_pass:0,batch_id:550,train_loss:[1.5889229],train_acc:[0.65625]
    train_pass:0,batch_id:600,train_loss:[1.498454],train_acc:[0.65625]
    train_pass:0,batch_id:650,train_loss:[1.5850043],train_acc:[0.59375]
    train_pass:0,batch_id:700,train_loss:[1.5123365],train_acc:[0.59375]
    train_pass:0,batch_id:750,train_loss:[1.5201966],train_acc:[0.6875]
    train_pass:0,batch_id:800,train_loss:[1.4692386],train_acc:[0.6875]
    train_pass:0,batch_id:850,train_loss:[1.5432602],train_acc:[0.65625]
    train_pass:0,batch_id:900,train_loss:[1.3355846],train_acc:[0.71875]
    train_pass:0,batch_id:950,train_loss:[1.5672383],train_acc:[0.5625]
    train_pass:0,batch_id:1000,train_loss:[1.3245198],train_acc:[0.8125]
    train_pass:0,batch_id:1050,train_loss:[1.4281952],train_acc:[0.625]
    train_pass:0,batch_id:1100,train_loss:[1.4515823],train_acc:[0.65625]
    train_pass:0,batch_id:1150,train_loss:[1.4436313],train_acc:[0.625]
    train_pass:0,batch_id:1200,train_loss:[1.331737],train_acc:[0.59375]
    train_pass:0,batch_id:1250,train_loss:[1.3248658],train_acc:[0.625]
    train_pass:0,batch_id:1300,train_loss:[1.3468654],train_acc:[0.59375]
    train_pass:0,batch_id:1350,train_loss:[1.3715186],train_acc:[0.6875]
    train_pass:0,batch_id:1400,train_loss:[1.3801335],train_acc:[0.6875]
    train_pass:0,batch_id:1450,train_loss:[1.3460943],train_acc:[0.78125]
    train_pass:0,batch_id:1500,train_loss:[1.1997194],train_acc:[0.75]
    train_pass:0,batch_id:1550,train_loss:[1.2082652],train_acc:[0.8125]
    train_pass:0,batch_id:1600,train_loss:[1.1453743],train_acc:[0.8125]
    train_pass:0,batch_id:1650,train_loss:[1.3685995],train_acc:[0.75]
    train_pass:0,batch_id:1700,train_loss:[1.0941882],train_acc:[0.84375]
    train_pass:0,batch_id:1750,train_loss:[1.1167028],train_acc:[0.625]
    train_pass:0,batch_id:1800,train_loss:[1.138406],train_acc:[0.75]
    train_pass:0,batch_id:1850,train_loss:[1.0731003],train_acc:[0.84375]
    train_pass:1,batch_id:50,train_loss:[1.1027997],train_acc:[0.6875]
    train_pass:1,batch_id:100,train_loss:[1.0531278],train_acc:[0.75]
    train_pass:1,batch_id:150,train_loss:[1.2336663],train_acc:[0.59375]
    train_pass:1,batch_id:200,train_loss:[1.0995479],train_acc:[0.8125]
    train_pass:1,batch_id:250,train_loss:[1.1541954],train_acc:[0.71875]
    train_pass:1,batch_id:300,train_loss:[0.97426534],train_acc:[0.84375]
    train_pass:1,batch_id:350,train_loss:[0.95332754],train_acc:[0.875]
    train_pass:1,batch_id:400,train_loss:[1.0367774],train_acc:[0.78125]
    train_pass:1,batch_id:450,train_loss:[0.9770886],train_acc:[0.875]
    train_pass:1,batch_id:500,train_loss:[1.2490355],train_acc:[0.65625]
    train_pass:1,batch_id:550,train_loss:[1.0032845],train_acc:[0.8125]
    train_pass:1,batch_id:600,train_loss:[0.9327422],train_acc:[0.875]
    train_pass:1,batch_id:650,train_loss:[0.97442627],train_acc:[0.75]
    train_pass:1,batch_id:700,train_loss:[0.93574905],train_acc:[0.78125]
    train_pass:1,batch_id:750,train_loss:[0.84056807],train_acc:[0.90625]
    train_pass:1,batch_id:800,train_loss:[0.8428248],train_acc:[0.84375]
    train_pass:1,batch_id:850,train_loss:[1.0382046],train_acc:[0.75]
    train_pass:1,batch_id:900,train_loss:[0.83112514],train_acc:[0.8125]
    train_pass:1,batch_id:950,train_loss:[1.0996622],train_acc:[0.59375]
    train_pass:1,batch_id:1000,train_loss:[0.9064267],train_acc:[0.875]
    train_pass:1,batch_id:1050,train_loss:[0.93668413],train_acc:[0.8125]
    train_pass:1,batch_id:1100,train_loss:[0.95547956],train_acc:[0.8125]
    train_pass:1,batch_id:1150,train_loss:[1.0219288],train_acc:[0.75]
    train_pass:1,batch_id:1200,train_loss:[0.917734],train_acc:[0.75]
    train_pass:1,batch_id:1250,train_loss:[0.9722259],train_acc:[0.8125]
    train_pass:1,batch_id:1300,train_loss:[0.9828568],train_acc:[0.75]
    train_pass:1,batch_id:1350,train_loss:[0.9958864],train_acc:[0.78125]
    train_pass:1,batch_id:1400,train_loss:[0.9488727],train_acc:[0.78125]
    train_pass:1,batch_id:1450,train_loss:[0.96769327],train_acc:[0.8125]
    train_pass:1,batch_id:1500,train_loss:[0.7616273],train_acc:[0.90625]
    train_pass:1,batch_id:1550,train_loss:[0.8437017],train_acc:[0.84375]
    train_pass:1,batch_id:1600,train_loss:[0.78343487],train_acc:[0.84375]
    train_pass:1,batch_id:1650,train_loss:[1.1406047],train_acc:[0.75]
    train_pass:1,batch_id:1700,train_loss:[0.75446355],train_acc:[0.84375]
    train_pass:1,batch_id:1750,train_loss:[0.7736151],train_acc:[0.8125]
    train_pass:1,batch_id:1800,train_loss:[0.7356458],train_acc:[0.84375]
    train_pass:1,batch_id:1850,train_loss:[0.68423724],train_acc:[0.8125]
    train_pass:2,batch_id:50,train_loss:[0.77664053],train_acc:[0.78125]
    train_pass:2,batch_id:100,train_loss:[0.7795843],train_acc:[0.875]
    train_pass:2,batch_id:150,train_loss:[0.9681474],train_acc:[0.71875]
    train_pass:2,batch_id:200,train_loss:[0.80061316],train_acc:[0.875]
    train_pass:2,batch_id:250,train_loss:[0.94856167],train_acc:[0.75]
    train_pass:2,batch_id:300,train_loss:[0.70266527],train_acc:[0.875]
    train_pass:2,batch_id:350,train_loss:[0.8074374],train_acc:[0.875]
    train_pass:2,batch_id:400,train_loss:[0.7148435],train_acc:[0.9375]
    train_pass:2,batch_id:450,train_loss:[0.7720412],train_acc:[0.90625]
    train_pass:2,batch_id:500,train_loss:[1.0238059],train_acc:[0.6875]
    train_pass:2,batch_id:550,train_loss:[0.741333],train_acc:[0.90625]
    train_pass:2,batch_id:600,train_loss:[0.75686616],train_acc:[0.875]
    train_pass:2,batch_id:650,train_loss:[0.8216514],train_acc:[0.78125]
    train_pass:2,batch_id:700,train_loss:[0.70247585],train_acc:[0.78125]
    train_pass:2,batch_id:750,train_loss:[0.6528368],train_acc:[0.90625]
    train_pass:2,batch_id:800,train_loss:[0.6920065],train_acc:[0.875]
    train_pass:2,batch_id:850,train_loss:[0.8080095],train_acc:[0.84375]
    train_pass:2,batch_id:900,train_loss:[0.6092522],train_acc:[0.84375]
    train_pass:2,batch_id:950,train_loss:[0.85070574],train_acc:[0.71875]
    train_pass:2,batch_id:1000,train_loss:[0.7449169],train_acc:[0.875]
    train_pass:2,batch_id:1050,train_loss:[0.7185369],train_acc:[0.875]
    train_pass:2,batch_id:1100,train_loss:[0.7696765],train_acc:[0.84375]
    train_pass:2,batch_id:1150,train_loss:[0.8875042],train_acc:[0.8125]
    train_pass:2,batch_id:1200,train_loss:[0.8200632],train_acc:[0.84375]
    train_pass:2,batch_id:1250,train_loss:[0.69583464],train_acc:[0.90625]
    train_pass:2,batch_id:1300,train_loss:[0.82912415],train_acc:[0.78125]
    train_pass:2,batch_id:1350,train_loss:[0.8223548],train_acc:[0.8125]
    train_pass:2,batch_id:1400,train_loss:[0.81956375],train_acc:[0.78125]
    train_pass:2,batch_id:1450,train_loss:[0.8496015],train_acc:[0.78125]
    train_pass:2,batch_id:1500,train_loss:[0.5711907],train_acc:[0.9375]
    train_pass:2,batch_id:1550,train_loss:[0.7187804],train_acc:[0.875]
    train_pass:2,batch_id:1600,train_loss:[0.67966396],train_acc:[0.84375]
    train_pass:2,batch_id:1650,train_loss:[1.0010756],train_acc:[0.71875]
    train_pass:2,batch_id:1700,train_loss:[0.5457661],train_acc:[0.875]
    train_pass:2,batch_id:1750,train_loss:[0.5996306],train_acc:[0.90625]
    train_pass:2,batch_id:1800,train_loss:[0.62291723],train_acc:[0.875]
    train_pass:2,batch_id:1850,train_loss:[0.59090817],train_acc:[0.84375]
    train_pass:3,batch_id:50,train_loss:[0.6881035],train_acc:[0.78125]
    train_pass:3,batch_id:100,train_loss:[0.66095173],train_acc:[0.8125]
    train_pass:3,batch_id:150,train_loss:[0.786123],train_acc:[0.71875]
    train_pass:3,batch_id:200,train_loss:[0.6757283],train_acc:[0.875]
    train_pass:3,batch_id:250,train_loss:[0.78517175],train_acc:[0.78125]
    train_pass:3,batch_id:300,train_loss:[0.6067616],train_acc:[0.875]
    train_pass:3,batch_id:350,train_loss:[0.693491],train_acc:[0.875]
    train_pass:3,batch_id:400,train_loss:[0.58876115],train_acc:[0.90625]
    train_pass:3,batch_id:450,train_loss:[0.63960433],train_acc:[0.875]
    train_pass:3,batch_id:500,train_loss:[0.84366024],train_acc:[0.75]
    train_pass:3,batch_id:550,train_loss:[0.61801255],train_acc:[0.9375]
    train_pass:3,batch_id:600,train_loss:[0.6193491],train_acc:[0.84375]
    train_pass:3,batch_id:650,train_loss:[0.713578],train_acc:[0.84375]
    train_pass:3,batch_id:700,train_loss:[0.59641397],train_acc:[0.84375]
    train_pass:3,batch_id:750,train_loss:[0.5371697],train_acc:[0.90625]
    train_pass:3,batch_id:800,train_loss:[0.54579353],train_acc:[0.875]
    train_pass:3,batch_id:850,train_loss:[0.69590867],train_acc:[0.875]
    train_pass:3,batch_id:900,train_loss:[0.49229026],train_acc:[0.875]
    train_pass:3,batch_id:950,train_loss:[0.74461067],train_acc:[0.71875]
    train_pass:3,batch_id:1000,train_loss:[0.67640686],train_acc:[0.875]
    train_pass:3,batch_id:1050,train_loss:[0.6003245],train_acc:[0.90625]
    train_pass:3,batch_id:1100,train_loss:[0.64893353],train_acc:[0.875]
    train_pass:3,batch_id:1150,train_loss:[0.7879097],train_acc:[0.8125]
    train_pass:3,batch_id:1200,train_loss:[0.7267597],train_acc:[0.84375]
    train_pass:3,batch_id:1250,train_loss:[0.6456135],train_acc:[0.875]
    train_pass:3,batch_id:1300,train_loss:[0.67847884],train_acc:[0.8125]
    train_pass:3,batch_id:1350,train_loss:[0.7757336],train_acc:[0.75]
    train_pass:3,batch_id:1400,train_loss:[0.67404693],train_acc:[0.8125]
    train_pass:3,batch_id:1450,train_loss:[0.7226076],train_acc:[0.84375]
    train_pass:3,batch_id:1500,train_loss:[0.4700474],train_acc:[0.9375]
    train_pass:3,batch_id:1550,train_loss:[0.68000025],train_acc:[0.90625]
    train_pass:3,batch_id:1600,train_loss:[0.562144],train_acc:[0.875]
    train_pass:3,batch_id:1650,train_loss:[0.9104176],train_acc:[0.8125]
    train_pass:3,batch_id:1700,train_loss:[0.4482058],train_acc:[0.96875]
    train_pass:3,batch_id:1750,train_loss:[0.51502436],train_acc:[0.90625]
    train_pass:3,batch_id:1800,train_loss:[0.50665116],train_acc:[0.90625]
    train_pass:3,batch_id:1850,train_loss:[0.5019931],train_acc:[0.875]
    train_pass:4,batch_id:50,train_loss:[0.62397534],train_acc:[0.78125]
    train_pass:4,batch_id:100,train_loss:[0.5621188],train_acc:[0.84375]
    train_pass:4,batch_id:150,train_loss:[0.68850565],train_acc:[0.75]
    train_pass:4,batch_id:200,train_loss:[0.62945414],train_acc:[0.875]
    train_pass:4,batch_id:250,train_loss:[0.73003805],train_acc:[0.78125]
    train_pass:4,batch_id:300,train_loss:[0.5659363],train_acc:[0.90625]
    train_pass:4,batch_id:350,train_loss:[0.6263416],train_acc:[0.875]
    train_pass:4,batch_id:400,train_loss:[0.48545587],train_acc:[0.9375]
    train_pass:4,batch_id:450,train_loss:[0.5512471],train_acc:[0.90625]
    train_pass:4,batch_id:500,train_loss:[0.7472045],train_acc:[0.78125]
    train_pass:4,batch_id:550,train_loss:[0.5334509],train_acc:[0.90625]
    train_pass:4,batch_id:600,train_loss:[0.50799733],train_acc:[0.90625]
    train_pass:4,batch_id:650,train_loss:[0.65166956],train_acc:[0.875]
    train_pass:4,batch_id:700,train_loss:[0.4838255],train_acc:[0.90625]
    train_pass:4,batch_id:750,train_loss:[0.4175865],train_acc:[0.9375]
    train_pass:4,batch_id:800,train_loss:[0.50088865],train_acc:[0.90625]
    train_pass:4,batch_id:850,train_loss:[0.6332574],train_acc:[0.84375]
    train_pass:4,batch_id:900,train_loss:[0.42015922],train_acc:[0.90625]
    train_pass:4,batch_id:950,train_loss:[0.622401],train_acc:[0.8125]
    train_pass:4,batch_id:1000,train_loss:[0.5464834],train_acc:[0.90625]
    train_pass:4,batch_id:1050,train_loss:[0.5117586],train_acc:[0.875]
    train_pass:4,batch_id:1100,train_loss:[0.55323666],train_acc:[0.90625]
    train_pass:4,batch_id:1150,train_loss:[0.7319232],train_acc:[0.84375]
    train_pass:4,batch_id:1200,train_loss:[0.66265965],train_acc:[0.875]
    train_pass:4,batch_id:1250,train_loss:[0.5651799],train_acc:[0.9375]
    train_pass:4,batch_id:1300,train_loss:[0.63153255],train_acc:[0.8125]
    train_pass:4,batch_id:1350,train_loss:[0.74896175],train_acc:[0.75]
    train_pass:4,batch_id:1400,train_loss:[0.6205651],train_acc:[0.8125]
    train_pass:4,batch_id:1450,train_loss:[0.65379846],train_acc:[0.875]
    train_pass:4,batch_id:1500,train_loss:[0.41178426],train_acc:[0.9375]
    train_pass:4,batch_id:1550,train_loss:[0.63206387],train_acc:[0.90625]
    train_pass:4,batch_id:1600,train_loss:[0.51774406],train_acc:[0.875]
    train_pass:4,batch_id:1650,train_loss:[0.7359965],train_acc:[0.8125]
    train_pass:4,batch_id:1700,train_loss:[0.36082542],train_acc:[1.]
    train_pass:4,batch_id:1750,train_loss:[0.46900907],train_acc:[0.90625]
    train_pass:4,batch_id:1800,train_loss:[0.49107864],train_acc:[0.875]
    train_pass:4,batch_id:1850,train_loss:[0.4434061],train_acc:[0.84375]


**PR Curve曲线可视化如下所示：**
![](https://ai-studio-static-online.cdn.bcebos.com/18e5e16b405c4d2c86a5186a88099fb5601f61f78feb4eefb20fb7ecfdfe1f31)


**Histogram可视化如下图所示**

![](https://ai-studio-static-online.cdn.bcebos.com/2194bd38a58f445dbdeabbfdd2ac66093ad29636d5c74f089c8d80e6c010361c)


## 总结
### 本文重点介绍了如何采用Paddle2.0对MNIST手写体识别进行模型训练的全过程
**并且在训练过程中：**

（1）VDL的Image组件，在数据处理过程中可以帮助我们更直观的了解输入的样本

（2）VDL的Scalar组件，在模型训练过程中可以帮助我们动态、实时的了解损失值、准确率等标量的变化

（3）VDL的PR Curve组件，在模型训练过程中可以帮助我们展示每一轮训练后精度与召回率之间的关系，这样我们可以更好的观察到每一轮训练后模型的好坏

（4）VDL的Histogram组件，在模型训练过程中可以帮助我们动态展示训练过程中某个tensor的分布，这样我们可以更好的掌握权值或者梯度的变化。

（5）VDL的Graph组件，在模型保存之后可以帮助我们更详细更直观了解到网络结构，以及tensor运算流的细节。

呼吁大家去VDL Github点star鸭！！

github首页：[https://github.com/PaddlePaddle/VisualDL](http://)

官网：[https://www.paddlepaddle.org.cn/paddle/visualdl](http://)

aistudio项目：
[https://aistudio.baidu.com/aistudio/projectdetail/502834](http://)

[https://aistudio.baidu.com/aistudio/projectdetail/622772](http://)

aistudio论坛：[https://ai.baidu.com/forum/topic/show/960053?pageNo=2](http://)


![](https://ai-studio-static-online.cdn.bcebos.com/5cd92feb62e345c382749281362fc11d68e098f01b224b92a38dcfc8ce486905)


请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
