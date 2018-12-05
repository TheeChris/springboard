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
       * Accuracy: the total true positives and true negatives divided by the total number of predictions
       * Precision: the number of true positives divided by the total number of positive predictions (true and false); this is useful when the true negative rate is important, especially if true negatives are rare
       * Recall (aka sensitivity, true positive rate, hit rate): the number of true positives divided by the total of true positives plus false negatives; useful when the true positive rate is important
       * Log Loss
       * F1 score: the harmonic mean of precision and recall
       * Area Under the Curve (AUC) - visualize thresholds for logistic regression
     * Regression metrics
       * Residual Sum-of-Squares (RSS): quantify the amount of error in a fitted model
       * Explained Sum-of-Squares (ESS): measure variance explained by the model
       * Total Sum-of-Squares (TSS): shows the variance in the predictions explained by the model and the variance that is attributed to error
       * Coefficient of Determination (R<sup>2</sup> and Adjusted R<sup>2</sup>): the percent of the variance in the response variable that can be explained by the model
       * Mean Squared Error (MSE): the average of the sum-of-squares error over their respective degrees of freedom
       * Root Mean Squared Error (RMSE): a measure of error in the same units as the target variable
       * Mean Absolute Error (MAE): similar to MSE, but since the values are not being squared, MAE is more robust to outliers
       * F-statistic: provides a p-value for the model using 
1. **What is the curse of dimensionality?**
     * The curse of dimensionality refers to the fact that a highly dimensional space is largely sparse, making modelling difficult. Typically, as the dimensionality increases, so must the sample size.
1. **Why do you need to set the random seed prior to running certain ML algorithms?**
     - Certain algorithms will utilize a number randomizer. If a random seed is not set for this randomizer, then a different seed may be used each time the algorithm is ran, resulting in slightly different outcomes.

## Regression
1. Can you explain the difference between Linear and Logistic Regression?
1. How are the coefficients in a Linear Regression interpreted?
1. How is the intercept in a Linear Regression interpreted?
1. Can the coefficients in a Logistic Regression be directly interpreted?
1. What is the Adjusted R-Squared? What range of values can it take?
1. Why is the Adjusted R-Squared a better measure than the regular R-Squared?
1. How does Logistic Regression work “under the hood”? Can you explain Gradient Descent?

## SVM
1. Can you explain how a Support Vector Machine works?
1. What is the kernel trick?
1. Where does the “support vector” term come in the SVM name?
1. What kind of kernels exist for SVMs?

## Trees
1. How do Decision Trees work?
1. What criteria does a tree-based algorithm use to decide on a split?
1. How does the Random Forest algorithm work? What are the sources of randomness?
1. How is feature importance calculated by the Random Forest?

## Other Supervised Learning
1. What is the difference between Bagging and Boosting?
1. How do Gradient Boosted Machines work?
1. What is Regularization, and what types do you know? Avoid overfitting, L1 and L2 regularization. L1 used as dim reduction, L2 better for overall generalization, bonus points for ElasticNet
1. Can a Random Forest and a GBM be easily parallelized? Why/why not?

## Unsupervised Learning
1. How does PCA work? What are the uses cases for it?
1. How can you determine the optimal number of principal components?
1. How does the K-Means algorithm work? What are its limitations?
1. What other clustering algorithms do you know?
1. How can you assess the quality of clustering?
1. Can you explain in detail any other clustering algorithms besides K-Means?