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

机器学习使用多种数据类型，如向量、文本、图片和结构数据等，管道中的接口使用来自Spark SQL的数据框来支撑各种数据类型.

数据框支持多种数据类型，可以从[Spark SQL datatype reference](https://spark.apache.org/docs/2.0.2/sql-programming-guide.html#spark-sql-datatype-reference)了解。除此之外，数据框可以使用机器学习的向量类型.

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

  转换器的``tranform()``和估计器的``fit()``方法都是无状态的。有状态的算法将来或会通过其他概念得到支持。

  在管道中的每一个转换器和估计器都有一个唯一的id，这个id在定制参数时非常有用

> 管道

  机器学习中通过一系列算法来处理数据是很常见的，例如一个文本处理工作流包括：
  * 对文本分词
  * 将每个文本的词转成特征向量
  * 通过特征向量和标签学习一个模型
  管道就代表了这样的一些工作流，包括一些列有序的管道阶段(转换器和估计器)，接下来将会用以上例子进行具体说明

> 工作原理

  管道由一系列的阶段组成，每个阶段是转换器或估计器。这些阶段会将数据框按顺序的转化。
  以下将会文档处理阐明管道工作流程
![文档处理工作流](http://i1.piimg.com/567571/0e6c0bce76e1c101.png)
上图第一行代表了管道处理的三个阶段，前两个阶段(*Tokenizer*和*HashingTF*)是转换器,第三阶段是估计器(*LogisticRegression*)。图中第二行表示数据在管道中的流向，圆柱代表数据框。管道的``fit()``方法在最初的数据框中被调用，数据框中包括文档和对应的标签。转换器Tokenizer调用``transform()``方法对文档进行分词，将词语作为新的列添加到数据框中。转换器HashingTF的``transform()``方法将数据框中词语的列转换成特征向量，同时将特征向量添加到数据框中。接着管道会调用估计器``LogisticRegression``的``fit()``方法产生``LogisticRegressionModel``模型。如果管道中接着还有更多的阶段，则会调用``LogisticRegressionModel``的``transform()``方法对数据框进行转换，然后再将转换后的数据框传递到下一阶段。

管道是一个估计器，因此在管道的``fit()``方法被调用后，会产生一个管道模型(*PipelineModel*)，这是一个转换器。管道模型会在测试时被使用，下面将对此说明
![文档处理-管道模型工作流](http://i2.buimg.com/567571/6dbf4733c197be92.png)
