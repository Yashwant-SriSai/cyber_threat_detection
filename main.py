import tkinter
from tkinter import messagebox, filedialog, Text, Scrollbar, Button, Label, END
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# TensorFlow/Keras Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM

main = tkinter.Tk()
main.title("Cyber Threat Detection Based on Artificial Neural Networks Using Event Profiles")
main.geometry("1300x1200")
main.config(bg='turquoise')

# Global Variables
le = preprocessing.LabelEncoder()
filename = None
feature_extraction = None
X, Y = None, None
doc = []
label_names = None
X_train, X_test, y_train, y_test = None, None, None, None

# Metrics Globals
lstm_acc, cnn_acc, svm_acc, knn_acc, dt_acc, random_acc, nb_acc = 0,0,0,0,0,0,0
lstm_precision, cnn_precision, svm_precision, knn_precision, dt_precision, random_precision, nb_precision = 0,0,0,0,0,0,0
lstm_recall, cnn_recall, svm_recall, knn_recall, dt_recall, random_recall, nb_recall = 0,0,0,0,0,0,0
lstm_fm, cnn_fm, svm_fm, knn_fm, dt_fm, random_fm, nb_fm = 0,0,0,0,0,0,0

def upload():
    global filename, X, Y, doc, label_names
    filename = filedialog.askopenfilename(initialdir="datasets")
    if not filename:
        return
    
    try:
        dataset = pd.read_csv(filename)
        # Attempt to find the label column automatically if 'labels' doesn't exist
        if 'labels' not in dataset.columns:
             # Assuming last column is the label if not named 'labels'
             dataset.rename(columns={dataset.columns[-1]: 'labels'}, inplace=True)

        label_names = dataset.labels.unique()
        dataset['labels'] = le.fit_transform(dataset['labels'])
        
        cols = dataset.shape[1]
        cols = cols - 1
        X = dataset.values[:, 0:cols]
        Y = dataset.values[:, cols]
        Y = Y.astype('int')
        
        doc.clear()
        for i in range(len(X)):
            strs = ''
            for j in range(len(X[i])):
                strs += str(X[i, j]) + " "
            doc.append(strs.strip())
            
        text.delete('1.0', END)
        text.insert(END, filename + ' Loaded\n')
        text.insert(END, "Total dataset size : " + str(len(dataset)) + "\n")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {str(e)}")

def tfidf():
    global X, feature_extraction
    if len(doc) == 0:
        messagebox.showwarning("Warning", "Upload dataset first!")
        return
    
    text.insert(END, 'Running TF-IDF... this may take a moment...\n')
    main.update()
    
    feature_extraction = TfidfVectorizer()
    # Limiting features to avoid MemoryError on large datasets
    # If you have a lot of RAM, you can remove max_features
    feature_extraction = TfidfVectorizer(max_features=1000) 
    tfidf_data = feature_extraction.fit_transform(doc)
    X = tfidf_data.toarray()
    
    text.insert(END, 'TF-IDF processing completed\n')

def eventVector():
    global X_train, X_test, y_train, y_test
    if X is None:
        messagebox.showwarning("Warning", "Run TF-IDF first!")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.delete('1.0', END)
    text.insert(END, 'Total unique events found in dataset are\n\n')
    text.insert(END, str(label_names) + "\n\n")
    text.insert(END, "Total dataset size : " + str(len(X)) + "\n")
    text.insert(END, "Data used for training : " + str(len(X_train)) + "\n")
    text.insert(END, "Data used for testing  : " + str(len(X_test)) + "\n")

def neuralNetwork():
    global lstm_acc, lstm_precision, lstm_fm, lstm_recall
    global cnn_acc, cnn_precision, cnn_fm, cnn_recall
    
    text.delete('1.0', END)
    text.insert(END, "Training Neural Networks... Please wait...\n")
    main.update()

    Y1 = Y.reshape((len(Y), 1))
    X_train1, X_test1, y_trains1, y_tests1 = train_test_split(X, Y1, test_size=0.2, random_state=0)
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y_trains1)
    y_train1 = enc.transform(y_trains1).toarray()
    y_test1 = enc.transform(y_tests1).toarray()
    
    # Reshaping for LSTM
    X_train2 = X_train1.reshape((X_train1.shape[0], X_train1.shape[1], 1))
    X_test2 = X_test1.reshape((X_test1.shape[0], X_test1.shape[1], 1))
    
    # --- LSTM Model ---
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train1.shape[1], 1)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train1.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train2, y_train1, epochs=1, batch_size=64, verbose=0) # Epochs reduced for speed
    
    prediction_data = model.predict(X_test2)
    prediction_data = np.argmax(prediction_data, axis=1)
    y_test_true = np.argmax(y_test1, axis=1)
    
    lstm_acc = accuracy_score(y_test_true, prediction_data) * 100
    lstm_precision = precision_score(y_test_true, prediction_data, average='macro', zero_division=0) * 100
    lstm_recall = recall_score(y_test_true, prediction_data, average='macro', zero_division=0) * 100
    lstm_fm = f1_score(y_test_true, prediction_data, average='macro', zero_division=0) * 100
    
    text.insert(END, "Deep Learning LSTM Extension Accuracy\n\n")
    text.insert(END, "LSTM Accuracy  : " + str(lstm_acc) + "\n")
    text.insert(END, "LSTM Precision : " + str(lstm_precision) + "\n")
    text.insert(END, "LSTM Recall    : " + str(lstm_recall) + "\n")
    text.insert(END, "LSTM Fmeasure  : " + str(lstm_fm) + "\n\n")
    
    # --- CNN Model ---
    # Note: This is a simple DNN/MLP, not a 2D CNN (which requires image data), but keeping name from original code
    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(X_train1.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(y_train1.shape[1]))
    cnn_model.add(Activation('softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    cnn_model.fit(X_train1, y_train1, epochs=5, batch_size=128, verbose=0)
    
    prediction_data = cnn_model.predict(X_test1)
    prediction_data = np.argmax(prediction_data, axis=1)
    
    cnn_acc = accuracy_score(y_test_true, prediction_data) * 100
    cnn_precision = precision_score(y_test_true, prediction_data, average='macro', zero_division=0) * 100
    cnn_recall = recall_score(y_test_true, prediction_data, average='macro', zero_division=0) * 100
    cnn_fm = f1_score(y_test_true, prediction_data, average='macro', zero_division=0) * 100
    
    text.insert(END, "Deep Learning CNN Accuracy\n\n")
    text.insert(END, "CNN Accuracy  : " + str(cnn_acc) + "\n")
    text.insert(END, "CNN Precision : " + str(cnn_precision) + "\n")
    text.insert(END, "CNN Recall    : " + str(cnn_recall) + "\n")
    text.insert(END, "CNN Fmeasure  : " + str(cnn_fm) + "\n")

def svmClassifier():
    global svm_acc, svm_precision, svm_fm, svm_recall
    text.delete('1.0', END)
    cls = svm.SVC(C=2.0, gamma='scale', kernel='linear', random_state=0)
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    
    svm_acc = accuracy_score(y_test, prediction_data) * 100
    svm_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    svm_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    svm_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    
    text.insert(END, "SVM Precision : " + str(svm_precision) + "\n")
    text.insert(END, "SVM Recall : " + str(svm_recall) + "\n")
    text.insert(END, "SVM FMeasure : " + str(svm_fm) + "\n")
    text.insert(END, "SVM Accuracy : " + str(svm_acc) + "\n")

def knn():
    global knn_precision, knn_recall, knn_fm, knn_acc
    text.delete('1.0', END)
    cls = KNeighborsClassifier(n_neighbors=10)
    cls.fit(X_train, y_train)
    text.insert(END, "KNN Prediction Results\n\n")
    prediction_data = cls.predict(X_test)
    
    knn_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    knn_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    knn_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    knn_acc = accuracy_score(y_test, prediction_data) * 100
    
    text.insert(END, "KNN Precision : " + str(knn_precision) + "\n")
    text.insert(END, "KNN Recall : " + str(knn_recall) + "\n")
    text.insert(END, "KNN FMeasure : " + str(knn_fm) + "\n")
    text.insert(END, "KNN Accuracy : " + str(knn_acc) + "\n")

def randomForest():
    global random_acc, random_precision, random_recall, random_fm
    text.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=5, random_state=0)
    cls.fit(X_train, y_train)
    text.insert(END, "Random Forest Prediction Results\n")
    prediction_data = cls.predict(X_test)
    
    random_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    random_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    random_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    random_acc = accuracy_score(y_test, prediction_data) * 100
    
    text.insert(END, "Random Forest Precision : " + str(random_precision) + "\n")
    text.insert(END, "Random Forest Recall : " + str(random_recall) + "\n")
    text.insert(END, "Random Forest FMeasure : " + str(random_fm) + "\n")
    text.insert(END, "Random Forest Accuracy : " + str(random_acc) + "\n")

def naiveBayes():
    global nb_precision, nb_recall, nb_fm, nb_acc
    text.delete('1.0', END)
    cls = BernoulliNB(binarize=0.0)
    cls.fit(X_train, y_train)
    text.insert(END, "Naive Bayes Prediction Results\n\n")
    prediction_data = cls.predict(X_test)
    
    nb_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    nb_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    nb_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    nb_acc = accuracy_score(y_test, prediction_data) * 100
    
    text.insert(END, "Naive Bayes Precision : " + str(nb_precision) + "\n")
    text.insert(END, "Naive Bayes Recall : " + str(nb_recall) + "\n")
    text.insert(END, "Naive Bayes FMeasure : " + str(nb_fm) + "\n")
    text.insert(END, "Naive Bayes Accuracy : " + str(nb_acc) + "\n")

def decisionTree():
    global dt_acc, dt_precision, dt_recall, dt_fm
    text.delete('1.0', END)
    cls = DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=3, min_samples_split=50, min_samples_leaf=20, max_features=5)
    cls.fit(X_train, y_train)
    text.insert(END, "Decision Tree Prediction Results\n")
    prediction_data = cls.predict(X_test)
    
    dt_precision = precision_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    dt_recall = recall_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    dt_fm = f1_score(y_test, prediction_data, average='macro', zero_division=0) * 100
    dt_acc = accuracy_score(y_test, prediction_data) * 100
    
    text.insert(END, "Decision Tree Precision : " + str(dt_precision) + "\n")
    text.insert(END, "Decision Tree Recall : " + str(dt_recall) + "\n")
    text.insert(END, "Decision Tree FMeasure : " + str(dt_fm) + "\n")
    text.insert(END, "Decision Tree Accuracy : " + str(dt_acc) + "\n")

def graph():
    height = [knn_acc, nb_acc, dt_acc, svm_acc, random_acc, lstm_acc, cnn_acc]
    bars = ('KNN', 'NB', 'DT', 'SVM', 'RF', 'LSTM', 'CNN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Accuracy Comparison")
    plt.show()

def precisiongraph():
    height = [knn_precision, nb_precision, dt_precision, svm_precision, random_precision, lstm_precision, cnn_precision]
    bars = ('KNN', 'NB', 'DT', 'SVM', 'RF', 'LSTM', 'CNN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Precision Comparison")
    plt.show()

def recallgraph():
    height = [knn_recall, nb_recall, dt_recall, svm_recall, random_recall, lstm_recall, cnn_recall]
    bars = ('KNN', 'NB', 'DT', 'SVM', 'RF', 'LSTM', 'CNN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Recall Comparison")
    plt.show()

def fmeasuregraph():
    height = [knn_fm, nb_fm, dt_fm, svm_fm, random_fm, lstm_fm, cnn_fm]
    bars = ('KNN', 'NB', 'DT', 'SVM', 'RF', 'LSTM', 'CNN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("F-Measure Comparison")
    plt.show()

# GUI Components
font = ('times', 16, 'bold')
title = Label(main, text='Cyber Threat Detection Based on Artificial Neural Networks Using Event Profiles')
title.config(bg='darkviolet', fg='gold', font=font, height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

# Buttons
uploadButton = Button(main, text="Upload Train Dataset", command=upload, font=font1)
uploadButton.place(x=50, y=550)

preprocessButton = Button(main, text="Run Preprocessing TF-IDF", command=tfidf, font=font1)
preprocessButton.place(x=240, y=550)

eventButton = Button(main, text="Generate Event Vector", command=eventVector, font=font1)
eventButton.place(x=535, y=550)

nnButton = Button(main, text="Neural Network Profiling", command=neuralNetwork, font=font1)
nnButton.place(x=730, y=550)

svmButton = Button(main, text="Run SVM Algorithm", command=svmClassifier, font=font1)
svmButton.place(x=950, y=550)

knnButton = Button(main, text="Run KNN Algorithm", command=knn, font=font1)
knnButton.place(x=1130, y=550)

rfButton = Button(main, text="Run Random Forest Algorithm", command=randomForest, font=font1)
rfButton.place(x=50, y=600)

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=naiveBayes, font=font1)
nbButton.place(x=320, y=600)

dtButton = Button(main, text="Run Decision Tree Algorithm", command=decisionTree, font=font1)
dtButton.place(x=570, y=600)

graphButton = Button(main, text="Accuracy Graph", command=graph, font=font1)
graphButton.place(x=830, y=600)

precisionButton = Button(main, text="Precision Graph", command=precisiongraph, font=font1)
precisionButton.place(x=1080, y=600)

recallButton = Button(main, text="Recall Graph", command=recallgraph, font=font1)
recallButton.place(x=50, y=650)

fmButton = Button(main, text="FMeasure Graph", command=fmeasuregraph, font=font1)
fmButton.place(x=320, y=650)

main.mainloop()