import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from tkinter import *
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support


labels=['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
Geography={'France':0,'Spain':1,'Germany':2}
Gender={'Female':0,'Male':1}

dataset = pd.read_csv('Churn_Modelling.csv')
dataset['Geography']=dataset['Geography'].map(Geography).to_frame()
dataset['Gender']=dataset['Gender'].map(Gender).to_frame()
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#initalization of naive bayes classifier
gnb=GaussianNB()

#initalization of decision tree classifier
clf=tree.DecisionTreeClassifier()

#initalization of K neighbours classifier
neigh = KNeighborsClassifier(n_neighbors=100)

#training all 3 models
gnb.fit(X_train,y_train)
clf.fit(X_train,y_train)
neigh.fit(X_train,y_train)


pred=gnb.predict(X_test)
score=accuracy_score(y_test,pred)

print ("\nAccuracy of Naive bayes classifier is : "+str(score*100))
print ("Confusion matrix:")
print (confusion_matrix(y_test, pred))

modelNaiveBayesCV= cross_val_score(gnb,X_train,y_train,cv=10) #modelNaiveBayesCV= cross_validation.cross_val_score(gnb,X_train,y_train,cv=10)

print("\nCross Validation Accuracy of Naive bayes classifier is : "+str(modelNaiveBayesCV.mean()*100))
matrix=precision_recall_fscore_support(y_test, pred)
print ("Precision Value of Yes : "+str(matrix[0][0]))
print ("Precision Value of No : "+str(matrix[0][1]))
print ("Recall Value of Yes : "+str(matrix[1][0]))
print ("Recall Value of No : "+str(matrix[1][1]))

pred=clf.predict(X_test)
score=accuracy_score(y_test,pred)

print ("\nAccuracy of Decision tree classifier is : "+str(score*100))
print ("Confusion matrix:")
print (confusion_matrix(y_test, pred))

Dtree= cross_val_score(clf,X_train,y_train,cv=10) 
print("\nCross Validation Accuracy of Decision tree classifier is : "+str(Dtree.mean()*100))
matrix=precision_recall_fscore_support(y_test, pred)
print ("Precision Value of Yes : "+str(matrix[0][0]))
print ("Precision Value of No : "+str(matrix[0][1]))
print ("Recall Value of Yes : "+str(matrix[1][0]))
print ("Recall Value of No : "+str(matrix[1][1]))

pred=neigh.predict(X_test)
score=accuracy_score(y_test,pred)

print ("\nAccuracy of KNN classifier is : "+str(score*100))
print ("Confusion matrix:")
print (confusion_matrix(y_test, pred))

KNN= cross_val_score(neigh,X_train,y_train,cv=10)

print("\nCross Validation Accuracy of KNN classifier is : "+str(KNN.mean()*100))
matrix=precision_recall_fscore_support(y_test, pred)
print ("Precision Value of Yes : "+str(matrix[0][0]))
print ("Precision Value of No : "+str(matrix[0][1]))
print ("Recall Value of Yes : "+str(matrix[1][0]))
print ("Recall Value of No : "+str(matrix[1][1]))

plt.plot(modelNaiveBayesCV,"-p",label="Naive Bayes")
plt.plot(Dtree,"-p",label="Decision tree")
plt.plot(KNN,"-p",label="KNN")
plt.title("Cross Validation Scores")
plt.xlabel("Fold Number")
plt.ylabel("Accuracy Score")
plt.legend(loc='upper right')
plt.show()
master = Tk()
master.title( "Enter Customer Details to predict" )
root = Frame(master)
root1 = Frame(master)

root.pack()
root1.pack( side = BOTTOM )
entry=[0]*10
Label(root,text=labels[0]).pack(side=LEFT)
entry[0] = Entry(root, width=10)
entry[0].pack(side=LEFT,padx=10,pady=10)

Label(root,text=labels[1]).pack(side=LEFT)
entry[1] = Entry(root, width=10)
entry[1].pack(side=LEFT,padx=10,pady=10)

Label(root,text=labels[2]).pack(side=LEFT)
entry[2] = Entry(root, width=10)
entry[2].pack(side=LEFT,padx=10,pady=10)

Label(root,text=labels[3]).pack(side=LEFT)
entry[3] = Entry(root, width=10)
entry[3].pack(side=LEFT,padx=10,pady=10)

Label(root,text=labels[4]).pack(side=LEFT)
entry[4] = Entry(root, width=10)
entry[4].pack(side=LEFT,padx=10,pady=10)

Label(root1,text=labels[5]).pack(side=LEFT)
entry[5] = Entry(root1, width=10)
entry[5].pack(side=LEFT,padx=10,pady=10)

Label(root1,text=labels[6]).pack(side=LEFT)
entry[6] = Entry(root1, width=10)
entry[6].pack(side=LEFT,padx=10,pady=10)

Label(root1,text=labels[7]).pack(side=LEFT)
entry[7] = Entry(root1, width=10)
entry[7].pack(side=LEFT,padx=10,pady=10)

Label(root1,text=labels[8]).pack(side=LEFT)
entry[8] = Entry(root1, width=10)
entry[8].pack(side=LEFT,padx=10,pady=10)

Label(root1,text=labels[9]).pack(side=LEFT)
entry[9] = Entry(root1, width=10)
entry[9].pack(side=LEFT,padx=10,pady=10)

def onok():
    pred_input=[]
    for i in range(10):
        if(i==1 or i==2):
            pred_input.append(entry[i].get())
        else:
            pred_input.append(int(entry[i].get()))
	
    pred_input[1] = Geography.get(pred_input[1])
    pred_input[2] = Gender.get(pred_input[2])
    
    np.array(pred_input)
    input1=gnb.predict([pred_input])
    print ("Naive bayes prediction")
    if(input1[0]==0):
        print("NO")
    else:
        print("YES")
    input1=clf.predict([pred_input])
    print ("Decision Tree prediction")
    if(input1[0]==0):
        print("NO")
    else:
        print("YES")
    input1=neigh.predict([pred_input])
    print ("KNN prediction")
    if(input1[0]==0):
        print("NO")
    else:
        print("YES")

Button(root1, text='OK', command=onok).pack(side=RIGHT)

root.mainloop()
root1.mainloop()



