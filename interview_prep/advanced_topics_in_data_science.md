Practice Interview Questions

# Advanced Topics in Data Science

1. What are stop words?

   Stop words are words like 'a', 'the', or 'an' that are commonly found in text data, but which provide little or no value to a NLP model. Therefore, stop words are commonly filtered out of text as a part of pre-processing.

2. Can you explain tokenization and lemmatization?

   1. Tokenization is the act of taking a text and breaking it into pieces (i.e. tokens) such as words or sentences.
   2. Lemmatization (or stemming) is the idea of using a word stem rather than the entire word in order to process all forms of the same word in the same way. For example, "spell", "spelling", and "spelled" might lemmatized to "spell"

3. Can you explain what is a Bag-of-Words?

   1. A bag-of-words is a process of simplifying a text by utilizing a set of words from the text and discarding word order or grammar.

4. What is the advantage of using TF-IDF instead of simple word counts?

   1. TF-IDF uses word counts in the context of the number of documents in which the word appears. This allows us to differentiate between a word that is important in a specific document, as opposed to frequently used in many or all documents.

5. Can you explain how word2vec works?

   1. Word2vec is a method that uses shallow neural nets to generate word embeddings (words represented as vectors). These vectors not only allow for individual words to be represented mathematically, but the spatial relationships between the vectors also have meaning. In the example below from the TensorFlow website, you can see that the distance and direction between 'man' and 'woman' is the same as between 'king' and 'queen'. The same is true of other linguistic relationships as well. 

      ![word2vec](https://www.tensorflow.org/images/linear-relationships.png)

6. What is topic modeling?

   Topic modeling is used to determine the probabilities that different topics are associated with a given text.

7. Can you explain how Latent Dirichlet Allocation works?

   Latent Dirichlet Allocation (LDA) is a type of topic model. The model consists of two matrices. The first is a matrix of probabilities of selecting a particular part (e.g. token) given a particular topic. The second is a matrix of probabilities of selecting a particular topic when sampling a particular document.

8. What ML algorithms are most commonly used for text classification tasks?

   Naïve Bayes

   Support Vector Machines

   K-Means Clustering (if predetermined categories are not available)

9. How would you explain what a Recommender System is to a non-technical person?

   Recommender systems use information about users and/or products to make predictions about a particular user's rating of or preference for another product.

10. What types of Recommender Systems do you know? Can you explain the differences between them?

    1. Content-based: recommend items similar to those liked/purchased by the user in the past
    2. Collaborative filtering: recommend items liked/purchased by similar users

11. What is the cold start problem?

    The cold start problem refers to the problem of trying to recommend similar items when the user has no purchase/rating history. Collaborative filtering can help to solve some of the cold start problem. 

12. Where does the term “deep” come in the definition of Deep Learning?

    "Deep" refers to the fact that the neural network has multiple hidden layers. 

13. What is a perceptron?

14. How does backpropagation work?

15. How does Stochastic Gradient Descent differ from regular Gradient Descent?

16. Do you know any other optimization algorithms?

17. What types of layer do you know in Deep Neural Networks?

18. What types of activation functions do you know?

19. Why do Deep Neural Networks train faster on a GPU?

20. How does a convolution work?

21. Is time series modeling a regression or a classification problem?

22. What is the difference between an AR and an MA component in a time series?

23. How does ARIMA forecasting work? Why is the I term important?

24. How can you ensure that you don’t overfit when building a time series model? Can you use regular Cross-Validation methods?

25. What methods do you know for working with time series data in pandas? Open-ended

26. How can you cluster time series data?