from tensorflow.keras.models import load_model

model = load_model('models/resnet50.h5')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

tsdata = ImageDataGenerator(rescale=1./255, rotation_range=30.0)
testdata = tsdata.flow_from_directory(directory="images/test", target_size=(224,224), color_mode='grayscale')

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

y_pred = []
y_true = []

for I, (X, Y) in enumerate(testdata):
    yp = model.predict(X)

    for v in Y.tolist():
        y_true.append(v.index(max(v)))

    for v in yp.tolist():
        y_pred.append(v.index(max(v)))
        
    if I == 100:
        break
    

cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=["Blade", "Gun", "Knife", "Shuriken"])
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Blade", "Gun", "Knife", "Shuriken"])

print(classification_report(y_true=y_true, y_pred=y_pred, labels=["Blade", "Gun", "Knife", "Shuriken"]))
