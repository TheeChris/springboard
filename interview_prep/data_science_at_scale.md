Practice Interview Questions

# Data Science at Scale

1. **What is MapReduce? What kind of problems is it used for?**

   MapReduce is a programming paradigm at the heart of Apache Hadoop. It is used for clustered, scale-out data processing solutions.

2. **Do you know what the advantages are of Spark over MapReduce?**

   Spark can do processing in-memory, while Hadoop MapReduce has to read from and write to a disk. As a result, the speed of processing differs significantly – Spark may be up to 100 times faster. However, Hadoop MapReduce is able to work with far larger data sets than Spark.

3. **Can you explain the difference between batch processing and stream processing?**

   Batch processing, as the name suggests, processes blocks (batches) of data that have been stored. Stream processing, however, processes data in real-time as it streams in. Hadoop is better suited for batch processing, whereas Spark is better for stream processing.

4. **What is a Spark Driver?**

   A Spark Driver is a task scheduler that houses all of the tasks for a Spark application and coordinates their execution.

   ![Spark Driver](https://jaceklaskowski.gitbooks.io/mastering-apache-spark/images/spark-driver.png)

5. **What is a Spark Executor?**

   The Executor is responsible for executing the tasks within a Driver. It provides in-memory storage for RDDs.

6. **Can you explain the difference between a Spark RDD and a Spark DataFrame?**

   Resilient Distributed Datasets (RDDs) and DataFrames are immutable distributions of data.  DataFrames are organized into named columns, like a database table, and allow data to be structured. RDDs are used for low-level transformations and actions of data that do not impose structure.

7. **What programming languages does Spark support with APIs?**

   Python, Java, and R. It natively supports Scala.

8. **What Machine Learning libraries do you know for Spark?**

   MLlib is Spark's machine learning library. It provides a large array of algorithms and utilities.

9. **What is sparkContext?**

   A sparkContext is the heart of a Spark application. It serves as the execution environment. A sparkContext offers a lot of functionality such as obtaining the current status of the application, setting configurations, creating distributed entities, and accessing, running and cancelling jobs.

10. **What is the difference between a Transformer and an Estimator?**

    A transformer extracts, transforms and selects features from a DataFrame and outputs a new DataFrame with appended columns of predicted values or transformed features. An estimator takes the transformed DataFrame and fits a learning algorithm and transforms the DataFrame with the predicted output.

11. **What is the Pipeline API used for?**

    The Pipeline API is used to easily create a machine learning pipeline by chaining multiple algorithms into a single workflow (pipeline).

12. **What is a Fully Connected Layer?**

    Fully connected layers connect one neuron to every other neuron in a network. They are useful in learning features and classification, but they do not easily scale with high-dimensional data such as images.

    ![fully connected layers](http://cs231n.github.io/assets/nn1/neural_net2.jpeg)

13. **What is a pooling layer?**

    A pooling layer combines the outputs of neuron clusters at one layer into a single neuron in the next layer. An example is a max pooling layer, which uses the maximum value of a cluster of neurons from the previous layer.

    ![pooling layer](http://cs231n.github.io/assets/cnn/maxpool.jpeg)

14. **What are Recurrent Neural Networks?**

    Recurrent Neural Networks (RNNs) are a neural network  where each node in a layer is connected via a directed graph to every other node in the successive layer. They can be connected by cyclic or acyclic graphs. These graphs allow the network to use information from previous passes, serving as a temporary, degrading memory. A basic, fully recurrent network is pictured below, but there a several other types of RNNs.

    ![Basic Unfolded RNN](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/800px-Recurrent_neural_network_unfold.svg.png)

15. **What sort of improvement does an LSTM bring over other RNNs?**

    LSTMs are able to maintain cyclic graphs without degradation by using an additional "memory" output. The network is trained as to what data to put into the additional "memory", thereby protecting it from being degraded from new inputs.

16. **Can you explain the TensorFlow programming model?**

    Tensorflow is a computational framework that uses Python to construct machine learning models as various levels of abstraction. High-level APIs (similar to scikit-learn) can be used to implement predefined architectures. Lower-level APIs are used to build models by defining a series of mathematical operations (algorithms) using libraries for model components.

    ![TensorFlow Hierarchy](https://developers.google.com/machine-learning/crash-course/images/TFHierarchy.svg)

17. **What types of activation functions do you know?**

    * Rectified Linear Unit (ReLU): range between 0 and infinity.

    * Leaky ReLu: range between -infinity and infinity

    * Step Function

    * Sigmoid (Logistic) Activation Function: range between 0 and 1. Useful for predicting the probability of an outcome.

    * Softmax Function: a logistic activation function used for multiclass classification

    * tanh (Logistic) Function: sigmoidal but range is between 1 and -1. 

    * Linear Function

18. **Why do Deep Neural Networks train faster on GPUs?**

GPUs are well-suited for deep neural networks because they are optimized for efficient matrix multiplication and convolution, allowing them to transmit larger amounts of data at a time. This is achieved through high bandwidth main memory, hiding sacrificed latency under thread parallelism, and L1 memory which is easily programmable. While a CPU's latency optimization make it faster, it can only carry small amount of data at a time.