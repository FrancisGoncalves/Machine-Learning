# ------------------------------------------- WSU - CPT_S 570 - HW2 -----------------------------------------
# ----------------------------------------------- Linear Kernel - best C ---------------------------
# ---------------------------------------- Author: Francisco Goncalves ------------------------------------------

from loadingData import trainingData, testingData
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def linearKernelBestC():
    print('\nThis will take a few minutes. Please wait...')
    clf = svm.SVC(C=0.1, kernel='linear', max_iter=1000)
    clf.fit(trainingData.examples, trainingData.labels)
    predictions = clf.predict(testingData.examples)
    accuracy_testing = clf.score(testingData.examples, testingData.labels)
    print('\n The testing accuracy is ' + str(accuracy_testing))
    cm = confusion_matrix(testingData.labels, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()
