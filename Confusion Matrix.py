import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

actual = np.random.binomial(1, 0.9, size=1000)
predicted = np.random.binomial(1, 0.9, size=1000)

confusionMatrix = metrics.confusion_matrix(actual, predicted)
print(confusionMatrix)

confusionMatrixDisplay = metrics.ConfusionMatrixDisplay(confusion_matrix = confusionMatrix, display_labels = [False, True])

confusionMatrixDisplay.plot()
plt.show()
