Pipeline的使用
-----
source:[ML Pipeline](https://spark.apache.org/docs/latest/ml-pipeline.html)

在谈及Pipeline的使用前先明确几个概念：
* 数据框(DataFrame)：底层是对RDD数据的封装的数据类型，结合SparkSQL可以对数据进行各种sql和统计操作，同时Spark2.1对dataframe算子和接口优化使得使用dataframe做数据转换和机器学习更为迅速便利。datafame支持多种数据类型,包括文本，特征向量，标签值，预测值等

* 转换器(Transformer)：转换器就是将一个数据框转换成另一个数据框的算法。例如一个*机器学习模型*就是一个转换器，我们将特征数据框经过模型转换，输出带有预测标签的数据框

* 估计器(Estimator)：估计器是拟合数据框来产生转换器的算法，例如一个*机器学习算法*就是一个估计器，通过拟合训练数据产生模型

* 管道(Pipeline)：将多个估计器和转换器串联起来，以实现机器学习的工作流(包括数据转换，模型训练、验证、预测等一系列过程)
* 参数(Parameter)：spark将估计器和转换器使用共同的api指定参数

#### 数据框(Dataframe)

机器学习使用多种数据类型，如向量、文本、图片和结构数据等，管道中的接口使用来自Spark SQL的数据框来支撑各种数据类型

数据框支持多种数据类型，可以从[Spark SQL datatype reference](https://spark.apache.org/docs/2.0.2/sql-programming-guide.html#spark-sql-datatype-reference)了解。除此之外，数据框可以使用机器学习的向量类型

数据框可以从RDD中显式或隐式建立，从[Spark SQL programming guide](https://spark.apache.org/docs/2.0.2/sql-programming-guide.html)了解更多相关信息  

数据框中的列需要命名，例如`'text'`,`'features'`和`'label'`

#### 管道组件(Pipeline components)
> 转换器(Transformers)

  转换器包含特征转换和学习到的模型。技术层面上说，转换器通过``transform()``方法将数据框a转换成另一个数据框b，数据框b相对于a来说多了一些列，例如：
  * 特征转换器的输入数据框a包含列(x1,x2,...,text),经转换器转换后的数据框b的列(x1,x2,...,text,feature vectors)
  * 学习模型转换器的输入数据框a包含列(x1,x2,...,text，feature vectors),经转换器转换后的数据框b的列(x1,x2,...,text,feature vectors，label)，多了预测标签的列

> 估计器(Estimators)

  估计器是通过算法拟合数据来产生模型的过程。估计器通过``fit()``方法对接受数据框来训练产生模型，这个模型就是一个转换器。例如``LogisticRegression``是一个估计器，通过``fit()``方法产生``LogisticRegressionModel``，这是一个转换器

> 管道组件特性

  转换器的``tranform()``和估计器的``fit()``方法都是无状态的。有状态的算法将来或会通过其他概念得到支持

  在管道中的每一个转换器和估计器都有一个唯一的id，这个id在定制参数时非常有用

#### 管道

  机器学习中通过一系列算法来处理数据是很常见的，例如一个文本处理工作流包括：
  * 对文本分词
  * 将每个文本的词转成特征向量
  * 通过特征向量和标签学习一个模型
  管道就代表了这样的一些工作流，包括一些列有序的管道阶段(转换器和估计器)，接下来将会用以上例子进行具体说明

> 工作原理

管道由一系列的阶段组成，每个阶段是转换器或估计器。这些阶段会将数据框按顺序的转化
以下将会文档处理阐明管道工作流程
![文档处理工作流](http://i1.piimg.com/567571/0e6c0bce76e1c101.png)
上图第一行代表了管道处理的三个阶段，前两个阶段(*Tokenizer*和*HashingTF*)是转换器,第三阶段是估计器(*LogisticRegression*)。图中第二行表示数据在管道中的流向，圆柱代表数据框。管道的``fit()``方法在最初的数据框中被调用，数据框中包括文档和对应的标签。转换器Tokenizer调用``transform()``方法对文档进行分词，将词语作为新的列添加到数据框中。转换器HashingTF的``transform()``方法将数据框中词语的列转换成特征向量，同时将特征向量添加到数据框中。接着管道会调用估计器``LogisticRegression``的``fit()``方法产生``LogisticRegressionModel``模型。如果管道中接着还有更多的阶段，则会调用``LogisticRegressionModel``的``transform()``方法对数据框进行转换，然后再将转换后的数据框传递到下一阶段

管道是一个估计器，因此在管道的``fit()``方法被调用后，会产生一个管道模型(*PipelineModel*)，这是一个转换器(注意``管道``和``管道模型``的区别)。管道模型会在测试时被使用，下面将对此说明
![文档处理-管道模型工作流](http://i2.buimg.com/567571/6dbf4733c197be92.png)
上图中管道模型和原来的管道一样有相同数目的阶段，但所有在管道中的估计器都变成转换器。当管道模型的``transform()``方法在测试数据集上被调用时，数据会依次通过管道的每个阶段，每个阶段的``transform()``方法会被调用将数据转化成新的数据传递到下个阶段

管道和管道模型确保训练数据和测试数据经过相同的特征处理

#### 更多

*DAG管道*：管道阶段都是有序队列，这里给出的例子都是线性管道，每个阶段都是使用上一阶段产生的数据作为输入。管道的数据流向也可以是非线性的有向无环图(DAG)，这类型的管道依赖每阶段指定输入和输出列的名，如果指定管道是DAG形式，则每个阶段必须指定其拓扑序列

*运行时检查*：管道运行在各种类型的数据框上，因而不能进行编译时检查，管道和管道模型在实际运行管道之前进行运行时检查。这种检查方法使用数据框摘要，这个摘要描述列的数据类型

*唯一的管道阶段*：管道的每个阶段必须是唯一的实例，例如 ``myHashingTF``实例不能进入管道两次，因为``myHashingTF``有唯一的ID，如果进入管道两次则管道的两个阶段有相同的id，但``myHashingTF1``和``myHashingTF2``可以进入相同的管道，因为他们是不同的实例(他们的类型是``HashingTF``)

#### 参数

MLlib包中估计器和转换器使用相同的接口来指定参数

``Param``是已命名的具有完整文档的参数，``ParamMap``是一组``参数,值``键值对

有两种主要的方法对算法进行传参：

1.  向实例传参。例如``lr``是``LogisticRegression``的实例，可以通过``lr.setMaxIter(10)``来让``lr.fit()``进行10次迭代。这个接口与MLlib包中的相似

2.  向``transform()``或``fit()``方法传递``ParamMap``参数，所有在``ParamMap``里的参数都会被重写覆盖

在``ParamMap``中的参数属于指定的实例，例如``LogisticRegression``有``lr1``和``lr2``两个实例，则可以同时设置两个实例的``maxIter``:``ParamMap(lr1.maxIter -> 10, lr2.maxIter -> 20)``,这对于一个管道中两个算法的相同方法参数设置非常有用

#### 保存和加载模型

自spark1.6起，保存和加载管道模型已有接口，可以保存部分基础转换器模型，支持的算法模型请参考相关相关算法接口文档

### 示例：Estimator, Transformer, and Param
语言：scala

```scala
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row

// 准备一组训练数据，包含标签和特征向量 (label, features)
val training = spark.createDataFrame(Seq(
  (1.0, Vectors.dense(0.0, 1.1, 0.1)),
  (0.0, Vectors.dense(2.0, 1.0, -1.0)),
  (0.0, Vectors.dense(2.0, 1.3, 1.0)),
  (1.0, Vectors.dense(0.0, 1.2, -0.5))
)).toDF("label", "features")

// 新建一个LogisticRegression实例，这是一个估计器
val lr = new LogisticRegression()
// Print out the parameters, documentation, and any default values.
println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

// 设置参数
lr.setMaxIter(10)
  .setRegParam(0.01)

// 训练LogisticRegression模型
val model1 = lr.fit(training)

// 输出模型参数
println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)

// 通过ParamMap设置参数
val paramMap = ParamMap(lr.maxIter -> 20)
  .put(lr.maxIter, 30)  //这将会覆盖原来的maxIter参数
  .put(lr.regParam -> 0.1, lr.threshold -> 0.55)  // 设置多个参数

// 改变输出列名
val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")  
// 合并paramMap
val paramMapCombined = paramMap ++ paramMap2

// 通过paramMapCombined传递参数学习新的模型
// paramMapCombined将会覆盖之前通过lr.set* 方法设置的参数
val model2 = lr.fit(training, paramMapCombined)
println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)

// 准备测试数据
val test = spark.createDataFrame(Seq(
  (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
  (0.0, Vectors.dense(3.0, 2.0, -0.1)),
  (1.0, Vectors.dense(0.0, 2.2, -1.5))
)).toDF("label", "features")

// 在测试数据上通过Transformer.transform()方法进行预测
// LogisticRegression.transform()方法只需要使用features列
// model2.transform() 输出的数据框中将会有'myProbability'列，因为我们之前通过
// lr.probabilityCol方法重命名'probability' 列
model2.transform(test)
  .select("features", "label", "myProbability", "prediction")
  .collect()
  .foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
    println(s"($features, $label) -> prob=$prob, prediction=$prediction")
  }
```
