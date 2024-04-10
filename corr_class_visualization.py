from itertools import cycle
from sklearn.metrics import auc, roc_curve, RocCurveDisplay
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics


def plot_confusion_matrix(ref, pred, max_no_comp):
    cm = metrics.confusion_matrix(ref, pred)
    counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    percentages = ["({0:.2%})".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(counts, percentages)]
    labels = np.asarray(labels).reshape(max_no_comp, max_no_comp)
    plt.figure(figsize=(10, 10))
    plt.rc('font', size=15)
    # requires seaborn >= 0.13.0
    sns.heatmap(cm, xticklabels=np.arange(1, max_no_comp+1), yticklabels=np.arange(1, max_no_comp+1), annot=labels,
                fmt='', cmap='Blues_r', cbar=False, linewidths=.5, square=True)
    plt.rc('font', size=15)
    plt.ylabel('Reference label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_confusion_matrix_binary(ref, pred):
    cm = metrics.confusion_matrix(ref, pred)
    counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    percentages = ["({0:.2%})".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(10, 10))
    plt.rc('font', size=15)
    # requires seaborn >= 0.13.0
    sns.heatmap(cm, xticklabels=["1", "≥2"], yticklabels=["1", "≥2"], annot=labels,
                fmt='', cmap='Blues_r', cbar=False, linewidths=.5, square=True)
    plt.rc('font', size=15)
    plt.ylabel('Reference label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_loss_curve(train_losses, val_losses, test_loss):
    plt.figure(figsize=(10, 10))
    plt.rc('font', size=15)
    plt.plot(val_losses, 'ro', label='validation')
    plt.plot(train_losses, 'b-', label='training')
    plt.plot([0, len(val_losses)-1], [test_loss, test_loss], 'g--', label='test')
    plt.ylabel('Cross entropy loss')
    plt.xlabel('Epoch #')
    plt.legend()
    plt.show()


def plot_corrects(val_correct_prct, test_correct_prct):
    plt.figure(figsize=(10, 10))
    plt.rc('font', size=15)
    plt.plot(val_correct_prct, 'bo-', label='validation')
    plt.plot([0, len(val_correct_prct)-1], [test_correct_prct, test_correct_prct], 'g-', label='test')
    plt.ylabel('Test correct (%)')
    plt.xlabel('Epoch #')
    plt.legend()
    plt.show()


def plot_noise_dependency(noiselvls, accs, binary):
    plt.figure(figsize=(10, 10))
    plt.rc('font', size=15)
    plt.plot(np.unique(noiselvls), accs, 'o')
    plt.xlabel("Noise level (%)")
    plt.ylabel("Test accuracy")
    if binary:
        plt.title('binary')
    else:
        plt.title('multiclass')
    plt.show()


def plot_multiclass_roc(ref, pred_score):
    target_names = np.unique(ref)
    n_classes = len(target_names)
    y_test = ref
    y_score = pred_score

    label_binarizer = LabelBinarizer().fit(y_test)
    y_onehot_test = label_binarizer.transform(y_test)

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print(f"Micro-averaged One-vs-Rest ROC AUC score: {roc_auc['micro']:.4f}")

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"Macro-averaged One-vs-Rest ROC AUC score: {roc_auc['macro']:.4f}")

    fig, ax = plt.subplots(figsize=(10, 10))

    plt.rc('font', size=15)

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["darkcyan", "darkorange", "royalblue", "forestgreen", "darkred"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"n={target_names[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == n_classes-1),
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=[0, 1],
        ylim=[0, 1]
    )

    plt.show()
