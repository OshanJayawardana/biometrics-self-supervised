predictions = resnettssd.predict(x_test)
y_pred=[]
for i in range(predictions.shape[0]):
    y_pred.append((np.where(predictions[i] == np.amax(predictions[i])))[0][0])

fpr, tpr, threshold = roc_curve(y_test, np.array(y_pred), pos_label=classes)
fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print("EER : ", EER)