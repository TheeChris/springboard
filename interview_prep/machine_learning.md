# Practice Interview Questions: Machine Learning

## Fundamentals
1. **Can you make the distinction between an algorithm and a model?**
    - An algorithm is a series of steps used to obtain an output. These steps, while clearly defined, lack enough information (the parameters) to produce an output on their own.
        - e.g. linear regression, gradient boosting, support vector machine, etc
    - A model is a well-defined, mathematical simplification used to explain a specific concept or phenomenon. Algorithms may be used as the instructions for the output of a model.
        - e.g. A random forest was used to create a model predicting the movement of disease vectors among Ash trees.
1. **What’s the difference between supervised and unsupervised learning?** 
    - Supervised learning uses labeled data to make predictions
    - Unsupervised learning uses unlabeled data to make prediction
1. **What’s the difference between a regression and a classification problem? How about clustering?**
    - Regression is a supervised learning approach that is used to map input features to a continuous numeric output.
    - Classification is a supervised learning approach that is used to map input features to a discrete number of outputs/classes. 
    - Clustering is an unsupervised learning approach that is used group data in such a way that each point in a cluster are ideally more similar to each other than to points in other clusters.
1. **Why do we use a train/test split?** 
    - To avoid overfitting and better generalization to unseen data.
1. **What is cross-validation used for? What types of cross-validation do you know?**
    - Cross-validation is used to test how well a model will generalize to unseen data.
    - Types of cross-valication:
        - **Holdout Method**: the traditional methods of holding out a specific test set that is not used in model building.
        - **K-fold**: splitting the data into K equally-sized groups, where one of the groups is used as a test set and the other K-1 groups are used for training the model. This is repeated until each group has been used as the test set and the results are averaged.
        - **Stratified K-fold**: similar to the K-fold method, but with a slight variation to deal with imbalanced outputs; each group is constructed so that it contains approximately the same ratio of outputs values as the complete dataset.
1. **What is generalization error?**
    - Generalization or out-of-sample error is a measure of how well an algorithm is able to predict outcomes with new or unseen data. Learning curve plots are used to visualize generalization error values through the learning process. 
1. **What is the bias-variance trade off?**
    - The bias-variance trade off is a tradeoff in algorithm complexity that results in the total errors of the model. An algorithm with low complexity will likely have high bias and more errors. An algorithm with a lot of complexity will likely have more errors due to high variance. An algorithm somewhere in between will have the least number of errors and the right amount of bias and variance.
      ​    - Models with high bias will underfit to the training data.
      - Models with high variance will overfit to the training data.
1. **What is the difference between overfit and underfit?**
    - Overfit is a model that reads the training data too literally and does not take into account the probable variability in population data. This will result in high accuracy/validation on training data, but low accuracy/validation on test or population data.
    - Underfit is a model that ignores the variance in the training data and results in all around low accuracy/validation.
1. **What accuracy metrics do you know, both for classification and for regression? When would you use one metric vs the other?**
     * Classification metrics
       * **Accuracy**: the total true positives and true negatives divided by the total number of predictions
       * **Precision**: the number of true positives divided by the total number of positive predictions (true and false); this is useful when the true negative rate is important, especially if true negatives are rare
       * **Recal**l (aka sensitivity, true positive rate, hit rate): the number of true positives divided by the total of true positives plus false negatives; useful when the true positive rate is important
       * **Log Loss**: quantify the accuracy by penalizing false classifications; this takes into account the probability of each classification
       * **F1 score**: the harmonic mean of precision and recall
       * **Area Under the Curve** (AUC-ROC) - visualize true positives plotted against false positives.
     * Regression metrics
       * **Residual Sum-of-Squares** (RSS): quantify the amount of error in a fitted model
       * **Explained Sum-of-Squares** (ESS): measure variance explained by the model
       * **Total Sum-of-Squares** (TSS): shows the variance in the predictions explained by the model and the variance that is attributed to error
       * **Coefficient of Determination** (R<sup>2</sup> and Adjusted R<sup>2</sup>): the percent of the variance in the response variable that can be explained by the model
       * **Mean Squared Error** (MSE): the average of the sum-of-squares error over their respective degrees of freedom
       * **Root Mean Squared Error** (RMSE): a measure of error in the same units as the target variable
       * **Mean Absolute Error** (MAE): similar to MSE, but since the values are not being squared, MAE is more robust to outliers
       * **F-statistic**: provides a p-value for the model using 
1. **What is the curse of dimensionality?**
     * The curse of dimensionality refers to the fact that a highly dimensional space is largely sparse, making modelling difficult. Typically, as the dimensionality increases, so must the sample size.
1. **Why do you need to set the random seed prior to running certain ML algorithms?**
     - Certain algorithms will utilize a number randomizer. If a random seed is not set for this randomizer, then a different seed may be used each time the algorithm is ran, resulting in slightly different outcomes.

## Regression
1. **Can you explain the difference between Linear and Logistic Regression?**

   Linear regression attempts to fit the data into a linear equation. This is used to predict a continuous variable. Logistic regression, in its simple form, uses log-odds to predict a binary response variable (e.g. 0/1, True/False, Hot Dog/Not Hot Dog) or a discrete response in multinomial or ordinal logistic regression. 

1. **How are the coefficients in a Linear Regression interpreted?**

   The sign of the coefficient will indicate whether the dependent and independent variables are positively or negatively correlated. The value of the coefficient is the mean response in dependent variable for every unit of change in the independent variables. 

1. **How is the intercept in a Linear Regression interpreted?**

   The intercept represents and prevents bias in the model by forcing the residual mean to 0.

1. **Can the coefficients in a Logistic Regression be directly interpreted?**

   Yes. The coefficients represent the rate of change in the log-odds of the dependent variable as the independent variable changes.

1. **What is the Adjusted R-Squared? What range of values can it take?**

   Since R<sup>2</sup> will be artificially increased as more features variables are added, Adjusted R<sup>2</sup> accounts for the number of features and will not increase in value if a feature does not improve the explained variance. The range of Adjusted R<sup>2</sup> is the same as R<sup>2</sup> (0-1), where 0 implies that the model explains none of the variance and 1 implies that all of the variance is explained.

1. **How does Logistic Regression work “under the hood”? Can you explain Gradient Descent?**

   Assuming a binomial distribution of the predictor variable, logistic regression models the log-odds of the mean using Maximum Likelihood Estimation as coefficients for the feature variables. Gradient Descent iteratively adjusts parameter values using calculus (partial derivatives) in order to minimize a loss function.

## SVM
1. **Can you explain how a Support Vector Machine works?**

   Support vector machines are a supervised learning algorithm that uses a set of labelled data points to create boundary lines that maximize the space between the points closest to the line. New points are then classified based upon which side of the boundary they fall upon.

1. **What is the kernel trick?**

   The kernel trick allows a Support Vector Machine to classify non-linear data points by using the dot product between two vectors.

1. **Where does the “support vector” term come in the SVM name?**

   Support vector refers to the orthogonal vector from the boundary that measures the space between the boundary line and its nearest points. This vector is what the SVM is trying to maximize in order to reduce misclassification of unseen data points.

1. **What kind of kernels exist for SVMs?**

   * Linear: for categories that are linearly separable

   * Radial Basis Function (RBF): a linear classifier for non-linear data. RBF separates the data points into a higher dimension in order to be able to divide the data points.

   ![RBF kernel](img/rbf_kernel.png)

   * Polynomial: a non-linear classifier (k(x,y)=(&gamma;⋅x<sup>T</sup>y + r)<sup>d</sup>, where d is the number of dimensions). Similar to RBF, but not as popular, except in NLP, where RBF tends to overfit.
   * Sigmoid: similar to the logistic regression function (k(x,y)=tanh(&gamma;⋅x<sup>T</sup>y+r))

## Trees
1. **How do Decision Trees work?**

   Decision trees work by making a series of binary splits where each split is determined by trying to minimize a given cost function.

   ![decision tree](img/decision_tree.png)

1. **What criteria does a tree-based algorithm use to decide on a split?**

   A tree-based algorithm is a greedy algorithm that makes a decision split based on minimizing a predetermined cost function (impurity).

1. **How does the Random Forest algorithm work? What are the sources of randomness?**

   Rather than a single decision tree, a random forest builds a series of decision trees and combines the results. Randomness comes from the fact that a random subset of features are used for each decision tree. Random decision thresholds for each feature can be used as well.

1. **How is feature importance calculated by the Random Forest?**

   Feature importance is measured by averaging the amount that each feature in a tree decreases impurity.

## Other Supervised Learning
1. **What is the difference between Bagging and Boosting?**

   Bagging is short for bootstrap aggregating and involves bootstrapping samples of the data to create multiple models. The predictions of those models are then aggregated to generate a final prediction. With bagging, all models run independent of each other and are then aggregated at completion. With boosting, however, each models dictates how the proceeding model will run by generating a set of weights.

1. **How do Gradient Boosted Machines work?**

   First, a loss function is defined. Second, a weak learner (decision tree) is generated. The decision tree is kept a weak learner by limiting it in some way (number of layers, nodes, splits, etc). Weak learners are added to the model one at a time using gradient descent to minimize the loss function. The output of each weak learner is added to the sequent of previous outputs in an attempt to correct the final output. Weak learners are added until a fixed number or an acceptable loss has been reached.

1. **What is Regularization, and what types do you know?**

   Regularization is a type of regression that shrinks the coefficients in an attempt to avoid overfitting. 

   * Ridge regression (L2 norm): adds a penalty equal to the sum of the squared value of the coefficients. This will force the parameters to be relatively small, the bigger the penalization, the smaller (and the more robust) the coefficients are.
   * Lasso regression (L1): adds a penalty equal to the sum of the absolute value of the coefficients. This  will shrink some parameters to zero, meaning that some variables will not play any role in the model.
   * ElasticNet (L1/L2): a combination of L1 and L2, where penalty is applied to the sum of the absolute values and to the sum of the squared values. Lambda will determine the ration between L1 and L2.

1. **Can a Random Forest and a GBM be easily parallelized? Why/why not?**

   Since random forests are independent weak learners, they can be parallelized. Gradient boosting machines, however, are sequential and, therefore, cannot be easily parallelized. 

## Unsupervised Learning
1. **How does PCA work? What are the uses cases for it?**

   Principal component analysis is a type of feature extraction that creates "new" features through combinations of existing features by projecting high-dimensional data onto a lower dimensional space.

1. **How can you determine the optimal number of principal components?**

   You can determine the optimal number of components by calculating the amount of variance explained by the number of components. 100% of variance explained would be equal to using all features, so you want to choose an acceptable number below 100%. By plotting the data, you can gain a sense of where the point of diminishing returns may be by finding the "elbow" in the line.

1. **How does the K-Means algorithm work? What are its limitations?**

   First, *k* centroids are randomly initiated. Next, data points are assigned to clusters based on their proximity to the centroids. Then, an iterative process moves the centroids until they are optimally place. For each centroid, the mean of the values of all points belonging to that cluster are calculated and this mean becomes the new centroid. Once the mean of all values in a cluster no longer moves the centroid (or some other criteria has been met), the process is stopped and clusters have been determined. Future points will be assigned to a cluster based on their distance from the centroids. One major limitation of this algorithm is that the clusters are highly variable due to the fact that they can change based on the random seeding of the initial centroids. Another limitation is that clusters are assumed to be spherically shaped, which is an assumption without much basis in reality.

1. **What other clustering algorithms do you know?**

   * **Affinity Propagation**: does not require the number of clusters ![$K$](https://render.githubusercontent.com/render/math?math=K&mode=inline) to be known in advance; uses a "message passing" paradigm to cluster points based on their similarity.
   * **Spectral Clustering**: uses the eigenvalues of a similarity matrix to reduce the dimensionality of the data before clustering in a lower dimensional space. This is tangentially similar to what we did to visualize k-means clusters using PCA. The number of clusters must be known a priori.
   * **Hierachical Clustering / Ward's Method**: take a set of data and successively divide the observations into more and more clusters at each layer of the hierarchy. Ward's method is used to determine when two clusters in the hierarchy should be combined into one. It is basically an extension of hierarchical clustering. Hierarchical clustering is *divisive*, that is, all observations are part of the same cluster at first, and at each successive iteration, the clusters are made smaller and smaller. With hierarchical clustering, a hierarchy is constructed, and there is not really the concept of "number of clusters." The number of clusters simply determines how low or how high in the hierarchy we reference and can be determined empirically or by looking at the [dendogram](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.cluster.hierarchy.dendrogram.html).
   * **Agglomerative Clustering**: is similar to hierarchical clustering but but is not divisive, it is *agglomerative*. That is, every observation is placed into its own cluster and at each iteration or level or the hierarchy, observations are merged into fewer and fewer clusters until convergence. Similar to hierarchical clustering, the constructed hierarchy contains all possible numbers of clusters and it is up to the analyst to pick the number by reviewing statistics or the dendogram.
   * **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: based on point density rather than distance. It groups together points with many nearby neighbors. It does not require knowing the number of clusters a priori, but does require specifying the neighborhood size.
   * **Mean-shift**: a sliding-window-based algorithm that attempts to find dense areas of data points using centroids that converge. 
   * **Gaussian Mixture Models**: these assume that the data points are Gaussian distributed; this is a less restrictive than the spherical assumption of k-means. With GMM, there are two parameters: the mean and the standard deviation. 

1. **How can you assess the quality of clustering?**

   * **Elbow sum-of-squares plot**: the number of clusters (*k*) is plotted against their related inertia (or sum-of-squared error). Find the "elbow" in the plot to determine the ideal number of clusters that reduces the amount of error without sacrificing overfitting and computational expense.

   * **Silhouette score**: measures how well each datapoint ![$x_i$](https://render.githubusercontent.com/render/math?math=x_i&mode=inline) "fits" its assigned cluster *and also* how poorly it fits into other clusters. The silhouette score ranges from -1 (a poor clustering) to +1 (a very dense clustering) with 0 denoting the situation where clusters overlap. Some criteria for the silhouette coefficient is provided in the table below.

     | Range      | Interpretation                                 |
     | ---------- | ---------------------------------------------- |
     | 0.71 - 1.0 | A strong structure has been found.             |
     | 0.51 - 0.7 | A reasonable structure has been found.         |
     | 0.26 - 0.5 | The structure is weak and could be artificial. |
     | < 0.25     | No substantial structure has been found.       |

   * **Gap statistic**: builds on the sum-of-squares established in the Elbow method , and compares it to the sum-of-squares of a "null distribution," that is, a random set of points with no clustering. The estimate for the optimal number of clusters *k* is the value for which the log of the sum-of-squared error falls the farthest below that of the reference distribution.