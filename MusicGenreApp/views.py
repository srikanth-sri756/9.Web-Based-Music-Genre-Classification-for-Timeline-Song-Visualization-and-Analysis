from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import pymysql
import librosa
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from django.views.decorators.csrf import csrf_exempt
from keras.models import model_from_json

labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')
X = X.astype('float32')

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def runMetrics(name,cls):
    predict = cls.predict(X_test)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    output = ''
    font = '<font size='' color=black>'
    arr = ['Algorithm Name','Accuracy','Average Precision (AP)','Recall','AUC']
    output += '<table border="1" align="center"><tr>'
    for i in range(len(arr)):
        output += '<th><font size="" color="black">'+arr[i]+'</th>'
    output += "</tr>"
    output += '<tr><td><font size="" color="black">'+name+'</td><td><font size="" color="black">'+str(a)+'</td><td><font size="" color="black">'+str(p)+'</td><td><font size="" color="black">'+str(r)+'</td><td><font size="" color="black">'+str(f)+'</td></tr>'
    conf_matrix = confusion_matrix(y_test, predict) 
    return output, conf_matrix

def TrainSVM(request):
    if request.method == 'GET':
        svm_cls = svm.SVC(kernel = 'linear')
        svm_cls.fit(X_train, y_train)
        output, conf_matrix = runMetrics("SVM Algorithm",svm_cls)
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,10])
        plt.title("SVM Algorithm Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()
        context= {'data':output}
        return render(request, 'ViewOutput.html', context)

def TrainDT(request):
    if request.method == 'GET':
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        output, conf_matrix = runMetrics("Decision Tree Ensemble Algorithm",dt)
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,10])
        plt.title("Decision Tree Ensemble Algorithm Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()
        context= {'data':output}
        return render(request, 'ViewOutput.html', context)    
    
    
def TrainFF(request):
    if request.method == 'GET':
        ff = MLPClassifier()
        ff.fit(X_train, y_train)
        output, conf_matrix = runMetrics("Feed Forward Neural Network Algorithm",ff)
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,10])
        plt.title("Feed Forward Neural Network Algorithm Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()
        context= {'data':output}
        return render(request, 'ViewOutput.html', context) 

def TrainLSTM(request):
    if request.method == 'GET':
        Y1 = to_categorical(Y)
        X1 = X.reshape(X.shape[0],X.shape[1],1)
        print(X1.shape)
        input_shape = (X1.shape[1], X1.shape[2])
        if os.path.exists('model/model.json'):
            with open('model/model.json', "r") as json_file:
                loaded_model_json = json_file.read()
                classifier = model_from_json(loaded_model_json)
            json_file.close()
            classifier.load_weights("model/model_weights.h5")
            classifier._make_predict_function()
        else:
            model = Sequential()
            model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=input_shape))
            model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
            model.add(Dense(units=Y1.shape[1], activation="softmax"))
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
            batch_size = 16  # num of training examples per minibatch
            num_epochs = 150
            hist = model.fit(X1, Y1, batch_size=batch_size, epochs=num_epochs)
            model.save_weights('model/model_weights.h5')
            model_json = model.to_json()
            with open("model/model.json", "w") as jsonFile:
                jsonFile.write(model_json)
            jsonFile.close() 
            f = open('model/history.pckl', 'wb')
            pickle.dump(hist.history, f)
            f.close()
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)
        predict = classifier.predict(X_test)
        predict = np.argmax(predict, axis=1)
        testY = np.argmax(y_test, axis=1)
        p = precision_score(testY, predict,average='macro') * 100
        r = recall_score(testY, predict,average='macro') * 100
        f = f1_score(testY, predict,average='macro') * 100
        a = accuracy_score(testY,predict)*100
        output = '<table border="1" align="center">'
        font = "<font size='' color=black>"
        arr = ['Algorithm Name','Accuracy','Average Precision (AP)','Recall','AUC']
        output += "<tr>"
        for i in range(len(arr)):
            output += '<th><font size="" color="black">'+arr[i]+"</th>"
        output += "</tr>"
        output += "<tr><td>"+font+"LSTM Algorithm</td><td>"+font+str(a)+"</td><td>"+font+str(p)+"</td><td>"+font+str(r)+"</td><td>"+font+str(f)+"</td></tr>"
        conf_matrix = confusion_matrix(testY, predict) 
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,10])
        plt.title("LSTM Algorithm Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()
        context= {'data':output}
        return render(request, 'ViewOutput.html', context) 

def ClassificationAction(request):
    if request.method == 'POST':
        myfile = request.FILES['t1']
        fname = request.FILES['t1'].name
        print(fname)
        x, sr = librosa.load('testMusicFiles/'+fname)
        mfccs = librosa.feature.mfcc(x, sr=sr)
        value = []
        for e in mfccs:
            value.append(np.mean(e))
        temp = []
        temp.append(value)
        value = np.asarray(temp)
        value = value.reshape(value.shape[0],value.shape[1],1)
        print(value.shape)
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()
        predict = classifier.predict(value)
        predict = np.argmax(predict)
        predict = labels[predict]
        context= {'data':"Uploaded Music Genre Classified As: "+str(predict)}
        return render(request, 'Classification.html', context) 


def Classification(request):
    if request.method == 'GET':
       return render(request, 'Classification.html', {})
    
def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def LoginAction(request):
    global uname
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MusicGenre',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+uname}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'Login.html', context)

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        gender = request.POST.get('t4', False)
        email = request.POST.get('t5', False)
        address = request.POST.get('t6', False)
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MusicGenre',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break
        if output == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MusicGenre',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,gender,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+gender+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = 'Signup Process Completed'
        context= {'data':output}
        return render(request, 'Signup.html', context)
      
