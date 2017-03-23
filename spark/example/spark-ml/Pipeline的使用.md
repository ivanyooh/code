Pipeline的使用 
-----
[source-ML Pipeline](https://spark.apache.org/docs/latest/ml-pipeline.html)

在谈及Pipeline的使用前先明确几个概念：
* 数据框(DataFrame)：底层是对RDD数据的封装的数据类型，结合SparkSQL可以对数据进行各种sql和统计操作，同时Spark2.1对dataframe算子和接口优化使得使用dataframe做数据转换和机器学习更为迅速便利。datafame支持多种数据类型,包括文本，特征向量，标签值，预测值等

* 转换器(Transformer)：转换器就是将一个数据框转换成另一个数据框的算法。例如一个机器学习模型就是一个转换器，我们将特征数据框经过模型转换，输出带有预测标签的数据框

* 估计器(Estimator)：估计器是拟合数据框来产生转换器的算法，例如一个机器学习算法就是一个估计器，通过拟合训练数据产生模型

* 管道(Pipeline)：将多个估计器和转换器串联起来，以实现机器学习的工作流(包括数据转换，模型训练、验证、预测等一系列过程)
* 参数(Parameter)：spark将估计器和转换器使用共同的api指定参数

#### 数据框(Dataframe)
    

