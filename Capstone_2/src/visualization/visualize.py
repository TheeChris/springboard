import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import validation_curve

def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    im = ax.imshow(cm, cmap=cmap, aspect='auto')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.7)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes)
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_yticks(tick_marks, classes)
    ax.grid(linewidth=0)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
def draw_roc_curve(model, features, target, ax):
    '''
    A function to draw the ROC curve and compute ROC-AUC score.
    model = a fitted model
    features = an array of processed feature data
    target = an array of the target variable
    '''
    probs = model.predict_proba(features)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(target, preds)
    roc_auc = auc(fpr, tpr)
    
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    
def draw_validation_curve(model, param_range, param, name, ax,
                          features, labels, metric='roc_auc', cv=3, n_jobs=1, title=None):
    '''
    A function to draw a validation curve using cross-validation.
    model = the algorithm or pipeline used to model the data
    param_range = the inputs to validate
    param = the hyperparameter to evaluate
    name = name of the hyperparameter to use for the graph axis
    metric = evaluation metric
    title = name of the algorithm to use for the graph title
    '''
    train_scores, test_scores = validation_curve(model, 
                                                 features, labels, 
                                                 param_name=param, 
                                                 param_range=param_range,
                                                 cv=cv, scoring=metric, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with %s" % title)
    plt.xlabel(name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")