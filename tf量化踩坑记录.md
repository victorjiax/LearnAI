TF伪量化训练和模型固化

- - -
调试TF过程种遇到很多奇怪的bug，记录原因和解决方案。
最重要的解决方案：1、冷静  2、debug  3、问题排除


>训练时is_training=True；  验证时is_training=False ，不设置的话，验证过程种参数会发生变化，例如>BN的mean和var是统计的均值。
- - -
目前不论MTK、还是三星、高通的NPU对pytorch的支持都不是很好，特别是onnx不支持保存伪量化信息， 因此使用pytorch进行伪量化训练的流程较为困难。本文简单介绍如何使用tensorflow进行伪量化训练。
本文的环境： Tensorflow 1.15.3 (1.14.0 存在 Add_v2 operate 报错), cuda 10.0,  具体对应关系参考
官方链接： https://www.tensorflow.org/install/source?hl=zh-cn#gpu

一、模型训练
       在模型结构设计和验证完成后，统计好模型的参数量和计算量（可以参考文章：[统计网络模型的参数量和计算量 ](https://github.com/victorjiax/LearnAI/blob/master/%E7%BB%9F%E8%AE%A1%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%8F%82%E6%95%B0%E9%87%8F%E5%92%8C%E8%AE%A1%E7%AE%97%E9%87%8F.md) ）确认好模型的收敛能力后即可开始伪量化训练。

   在模型结构定义完成，计算其loss之后，增加以下代码即可设定tensorflow的伪量化训练：

   tf.contrib.quantize.create_training_graph(quant_delay=20)

 >quant_delay是迭代了多少个step后，网络开始量化统计最大值，最小值，并用8bit做反向传播更新梯度。也可以设置quant_delay=0，相当于使用训练好的预训练模型加载了直接微调。

   在定义saver时，用saver = tf.train.Saver(tf.global_variables())或者saver=tf.train.Saver()较为保险（注：默认保存所有的变量），否则有可能存在一些伪量化节点未被保存，在freeze阶段模型load出错。

   在使用tensorpack为训练框架时，可以设置ModelSaver(var_collections=tf.GraphKeys.GLOBAL_VARIABLES) 或ModelSaver(var_collections=None) 确保所有的伪量化节点均被保存；

   如果需要加载预训练的参数，可以新建一个saver对象，加载和保存分开，例如
   ```
   saver_restore = tf.train.Saver(var_list=[v for v in tf.global_variables() if 'quant' not in v.name and 'Quant' not in v.name and "Batch" not in v.name], max_to_keep=100)
   if file_config.config["checkpoint_path"]:
      saver_restore.restore(sess, file_config.config["checkpoint_path"])
   ```

二、模型固化

   在freeze阶段，构建好了inference_graph，加入 tf.contrib.quantize.create_eval_graph(input_graph=tf.get_default_graph())，restore训练完成的ckpt文件，生成固话的pb。
  
```
from tensorflow.tools.graph_transforms import TransformGraph

input_pb='BiSeNetV2_D.25_INT_model_408_tp3.pb'

modify_pb_path='BiSeNetV2_D.25_INT_model_408_tp3_fuse.pb'

 

with tf.Graph().as_default():

    with open(input_pb,"rb") as f:

        net_graph_def = tf.GraphDef()

        net_graph_def.ParseFromString(f.read())

        input_names = ['input_image']

        output_names = ['pred']

        transforms = ["strip_unused_nodes",'strip_unused_nodes(type=float,shape="1,384,384,3")','remove_nodes(op=Identity,op=CheckNumerics)',\

                     'fold_constants','fold_batch_norms','fold_old_batch_norms']

        output_graph = TransformGraph(net_graph_def,input_names,output_names,transforms)

        with tf.gfile.GFile(modify_pb_path,'w') as f:

            f.write(output_graph.SerializeToString())
```
三、模型的节点合并

深度模型中采用BN来提高模型的泛化能力。但是在推理时，模型结构和参数已经固化，由CONV、BN的数学原理可以知道,可以把BN层直接嵌入到conv层（参考：https://zhuanlan.zhihu.com/p/48005099 ，https://zhuanlan.zhihu.com/p/110552861 ），减少运算量，提高网络的运行速度。 

可以使用tensorflow的TransformGraph对模型的节点进行融合，代码参考如下：
```

from tensorflow.tools.graph_transforms import TransformGraph

 

input_pb='BiSeNetV2_D.25_INT_model_408_tp3.pb'

modify_pb_path='BiSeNetV2_D.25_INT_model_408_tp3_fuse.pb'

 

with tf.Graph().as_default():

    with open(input_pb,"rb") as f:

        net_graph_def = tf.GraphDef()

          net_graph_def.ParseFromString(f.read())

        input_names = ['input_image']

        output_names = ['pred']

        transforms = ["strip_unused_nodes",'strip_unused_nodes(type=float,shape="1,384,384,3")','remove_nodes(op=Identity,op=CheckNumerics)',\

                     'fold_constants','fold_batch_norms','fold_old_batch_norms']

        output_graph = TransformGraph(net_graph_def,input_names,output_names,transforms)

        with tf.gfile.GFile(modify_pb_path,'w') as f:

            f.write(output_graph.SerializeToString())

```

四、其它问题

1） 训练过程不收敛或者收敛很慢
   a) 检查训练数据是否解析正常
   b) 检查网络结构是否正常，例如是否存在重复使用softmax
   c) loss 设计是否正确
   d) BN层以及训练参数是否可训练，检查输入的参数
   e）构建tensorboard查看网络构建情况
2）训练收敛，验证过程不收敛
   a）采用BN的话，是否decay设置太低
   b）检查网络结构，例如必要参数是否传入，尤其需要debug
   如果发现不能debug，1）修改导入方式 例如from tensorflow import contrib 
   2）找到源码所在位置，打断点，可以显示相关变量。大部分情况都是变量传入不完整
3） 内存溢出，深层网络很占内存，建议训练0.5 96  0.75 64 mobilenet v3
4）adam 默认学习率0.001
5)  出现在浮点上量化精度猛降后换忙上升，原因暂未解决，估计网络的结构问题，替换为原始的tf_slim网络
6）tf插入伪节点存在坑，部分节点伪量化节点插入不全，如果自行插入伪节点，可能客户端不支持。
7) 浮点基础上量化。存在精度收敛快，效果明显，如果浮点量化效果不如重新训练，需要检查代码


其它：
IN的代码实现
```
def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
#     mu = tf.reduce_mean (net, [1,2], keepdims=True)
#     sigma_sq = tf.reduce_mean (tf.square (net-mu), [1,2])
#     shift = tf.Variable(tf.zeros(var_shape))
#     scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/tf.sqrt(sigma_sq + epsilon)

    #return scale * normalized + shift
    return normalized
   ```
  手动插入量化节点
  from tensorflow.contrib.quantize.python.quant_ops import MovingAvgQuantize
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/80d95d42ba1f4352a80db130072f5607.png)
  
  训练过程问题排查

1、检查训练数据据是否输出正常，可以通过tf.print

tf.Print(input, data, message=None, first_n=None, summarize=None, name=None)

'''
参数param：
input: 是一个tensor，需要打印的张量；
data：data要求是一个list，list的每一个元素是一个张量，里面包含要打印的内容；
message：是需要输出的错误信息；
first_n：指只记录前n次；
summarize：是对每个tensor只打印的条目数量，如果是None，对于每个输入tensor只打印3个元素
name：这个操作op的名字
返回值return:
返回一个 Tensor，和 input的形状一样


  example:

  x = tf.Print(images_, ["images_:", images_], message='images_:  ', summarize=100)  
sess.run([x])
2、通过debug确定向量维度是否相同
3、通过检查模型网络结构，或者将ckpt转pb后，查看结构



！ tf 将pb再训练，只能训练新加入的层结构，对原始的参数，不可训练
