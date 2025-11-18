import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time as t
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
#import random as rnd
import warnings as w
w.filterwarnings('ignore')
data = pd.read_csv("AI-Data.csv")
# Create interactive plotly subplots - 3 graphs per row
fig = make_subplots(
    rows=4, cols=3,
    subplot_titles=('Correlation Heatmap', 'Class Count', 'Semester-wise',
                   'Gender-wise', 'Nationality-wise', 'Grade-wise', 
                   'Section-wise', 'Topic-wise', 'Stage-wise',
                   'Absent Days-wise', '', ''),
    specs=[[{"type": "heatmap"}, {"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {}, {}]]
)

# 1. Correlation Heatmap
corr_matrix = data.corr(numeric_only=True)
fig.add_trace(go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale='RdBu'), row=1, col=1)

# 2. Class Count
class_counts = data['Class'].value_counts().reindex(['L', 'M', 'H'])
fig.add_trace(go.Bar(x=class_counts.index, y=class_counts.values, name='Class Count'), row=1, col=2)

# 3. Semester-wise
sem_class = data.groupby(['Semester', 'Class']).size().unstack(fill_value=0)
for cls in ['L', 'M', 'H']:
    fig.add_trace(go.Bar(x=sem_class.index, y=sem_class[cls], name=f'Class {cls}', showlegend=False), row=1, col=3)

# 4. Gender-wise
gender_class = data.groupby(['gender', 'Class']).size().unstack(fill_value=0)
for cls in ['L', 'M', 'H']:
    fig.add_trace(go.Bar(x=gender_class.index, y=gender_class[cls], name=f'Class {cls}', showlegend=False), row=2, col=1)

# 5. Nationality-wise
nat_class = data.groupby(['NationalITy', 'Class']).size().unstack(fill_value=0)
for cls in ['L', 'M', 'H']:
    fig.add_trace(go.Bar(x=nat_class.index, y=nat_class[cls], name=f'Class {cls}', showlegend=False), row=2, col=2)

# 6. Grade-wise
grade_class = data.groupby(['GradeID', 'Class']).size().unstack(fill_value=0)
for cls in ['L', 'M', 'H']:
    fig.add_trace(go.Bar(x=grade_class.index, y=grade_class[cls], name=f'Class {cls}', showlegend=False), row=2, col=3)

# 7. Section-wise
sec_class = data.groupby(['SectionID', 'Class']).size().unstack(fill_value=0)
for cls in ['L', 'M', 'H']:
    fig.add_trace(go.Bar(x=sec_class.index, y=sec_class[cls], name=f'Class {cls}', showlegend=False), row=3, col=1)

# 8. Topic-wise
topic_class = data.groupby(['Topic', 'Class']).size().unstack(fill_value=0)
for cls in ['L', 'M', 'H']:
    fig.add_trace(go.Bar(x=topic_class.index, y=topic_class[cls], name=f'Class {cls}', showlegend=False), row=3, col=2)

# 9. Stage-wise
stage_class = data.groupby(['StageID', 'Class']).size().unstack(fill_value=0)
for cls in ['L', 'M', 'H']:
    fig.add_trace(go.Bar(x=stage_class.index, y=stage_class[cls], name=f'Class {cls}', showlegend=False), row=3, col=3)

# 10. Absent Days-wise
absent_class = data.groupby(['StudentAbsenceDays', 'Class']).size().unstack(fill_value=0)
for cls in ['L', 'M', 'H']:
    fig.add_trace(go.Bar(x=absent_class.index, y=absent_class[cls], name=f'Class {cls}', showlegend=False), row=4, col=1)

fig.update_layout(height=1200, title_text="Student Performance Analysis - All Visualizations")
fig.show()
#cor = data.corr()
#print(cor)
data = data.drop("gender", axis=1)
data = data.drop("StageID", axis=1)
data = data.drop("GradeID", axis=1)
data = data.drop("NationalITy", axis=1)
data = data.drop("PlaceofBirth", axis=1)
data = data.drop("SectionID", axis=1)
data = data.drop("Topic", axis=1)
data = data.drop("Semester", axis=1)
data = data.drop("Relation", axis=1)
data = data.drop("ParentschoolSatisfaction", axis=1)
data = data.drop("ParentAnsweringSurvey", axis=1)
#data = data.drop("VisITedResources", axis=1)
data = data.drop("AnnouncementsView", axis=1)
u.shuffle(data)
countD = 0
countP = 0
countL = 0
countR = 0
countN = 0
gradeID_dict = {"G-01" : 1,
                "G-02" : 2,
                "G-03" : 3,
                "G-04" : 4,
                "G-05" : 5,
                "G-06" : 6,
                "G-07" : 7,
                "G-08" : 8,
                "G-09" : 9,
                "G-10" : 10,
                "G-11" : 11,
                "G-12" : 12}
data = data.replace({"GradeID" : gradeID_dict})
#sig = []
for column in data.columns:
    if data[column].dtype == type(object):
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])
ind = int(len(data) * 0.70)
feats = data.values[:, 0:4]
lbls = data.values[:,4]
feats_Train = feats[0:ind]
feats_Test = feats[(ind+1):len(feats)]
lbls_Train = lbls[0:ind]
lbls_Test = lbls[(ind+1):len(lbls)]
modelD = tr.DecisionTreeClassifier()
modelD.fit(feats_Train, lbls_Train)
lbls_predD = modelD.predict(feats_Test)
for a,b in zip(lbls_Test, lbls_predD):
    if(a==b):
        countD += 1
accD = (countD/len(lbls_Test))
print("\nAccuracy measures using Decision Tree:")
print(m.classification_report(lbls_Test, lbls_predD),"\n")
print("\nAccuracy using Decision Tree: ", str(round(accD, 3)))
t.sleep(1)
modelR = es.RandomForestClassifier()
modelR.fit(feats_Train, lbls_Train)
lbls_predR = modelR.predict(feats_Test)
for a,b in zip(lbls_Test, lbls_predR):
    if(a==b):
        countR += 1
print("\nAccuracy Measures for Random Forest Classifier: \n")
#print("\nConfusion Matrix: \n", m.confusion_matrix(lbls_Test, lbls_predR))
print("\n", m.classification_report(lbls_Test,lbls_predR))
accR = countR/len(lbls_Test)
print("\nAccuracy using Random Forest: ", str(round(accR, 3)))
t.sleep(1)
modelP = lm.Perceptron()
modelP.fit(feats_Train, lbls_Train)
lbls_predP = modelP.predict(feats_Test)
for a,b in zip(lbls_Test, lbls_predP):
    if a == b:
        countP += 1
accP = countP/len(lbls_Test)
print("\nAccuracy measures using Linear Model Perceptron:")
print(m.classification_report(lbls_Test, lbls_predP),"\n") 
print("\nAccuracy using Linear Model Perceptron: ", str(round(accP, 3)), "\n")
t.sleep(1)
modelL = lm.LogisticRegression()
modelL.fit(feats_Train, lbls_Train)
lbls_predL = modelL.predict(feats_Test)
for a,b in zip(lbls_Test, lbls_predL):
    if a == b:
        countL += 1
accL = countL/len(lbls_Test)
print("\nAccuracy measures using Linear Model Logistic Regression:")
print(m.classification_report(lbls_Test, lbls_predL),"\n")
print("\nAccuracy using Linear Model Logistic Regression: ", str(round(accL, 3)), "\n")
t.sleep(1)
modelN = nn.MLPClassifier(activation="logistic")
modelN.fit(feats_Train, lbls_Train)
lbls_predN = modelN.predict(feats_Test)
for a,b in zip(lbls_Test, lbls_predN):
    #sig.append(1/(1+ np.exp(-b)))
    if a==b:
        countN += 1
#print("\nAverage value of Sigmoid Function: ", str(round(np.average(sig), 3)))
print("\nAccuracy measures using MLP Classifier:")
print(m.classification_report(lbls_Test, lbls_predN),"\n")
accN = countN/len(lbls_Test)
print("\nAccuracy using Neural Network MLP Classifier: ", str(round(accN, 3)), "\n")
choice = input("Do you want to test specific input (y or n): ")
if(choice.lower()=="y"):
    gen = input("Enter Gender (M or F): ")
    if (gen.upper() == "M"):
       gen = 1
    elif (gen.upper() == "F"):
       gen = 0
    nat = input("Enter Nationality: ")
    pob = input("Place of Birth: ")
    gra = input("Grade ID as (G-<grade>): ")
    if(gra == "G-02"):
        gra = 2
    elif (gra == "G-04"):
        gra = 4
    elif (gra == "G-05"):
        gra = 5
    elif (gra == "G-06"):
        gra = 6
    elif (gra == "G-07"):
        gra = 7
    elif (gra == "G-08"):
        gra = 8
    elif (gra == "G-09"):
        gra = 9
    elif (gra == "G-10"):
        gra = 10
    elif (gra == "G-11"):
        gra = 11
    elif (gra == "G-12"):
        gra = 12
    sec = input("Enter Section: ")
    top = input("Enter Topic: ")
    sem = input("Enter Semester (F or S): ")
    if (sem.upper() == "F"):
       sem = 0
    elif (sem.upper() == "S"):
       sem = 1
    rel = input("Enter Relation (Father or Mum): ")
    if (rel == "Father"):
       rel = 0
    elif (rel == "Mum"):
       rel = 1
    rai = int(input("Enter raised hands: "))
    res = int(input("Enter Visited Resources: "))
    ann = int(input("Enter announcements viewed: "))
    dis = int(input("Enter no. of Discussions: "))
    sur = input("Enter Parent Answered Survey (Y or N): ")
    if (sur.upper() == "Y"):
       sur = 1
    elif (sur.upper() == "N"):
       sur = 0
    sat = input("Enter Parent School Satisfaction (Good or Bad): ")
    if (sat == "Good"):
       sat = 1
    elif (sat == "Bad"):
       sat = 0
    absc = input("Enter No. of Abscenes(Under-7 or Above-7): ")
    if (absc == "Under-7"):
       absc = 1
    elif (absc == "Above-7"):
       absc = 0
    arr = np.array([rai, res, dis, absc])
    #arr = np.array([gen, rnd.randint(0, 30), rnd.randint(0, 30), sta, gra, rnd.randint(0, 30), rnd.randint(0, 30), sem, rel, rai, res, ann, dis, sur, sat, absc])
    predD = modelD.predict(arr.reshape(1, -1))
    predR = modelR.predict(arr.reshape(1, -1))
    predP = modelP.predict(arr.reshape(1, -1))
    predL = modelL.predict(arr.reshape(1, -1))
    predN = modelN.predict(arr.reshape(1, -1))
    if (predD == 0):
        predD = "H"
    elif (predD == 1):
        predD = "M"
    elif (predD == 2):
        predD = "L"
    if (predR == 0):
        predR = "H"
    elif (predR == 1):
        predR = "M"
    elif (predR == 2):
        predR = "L"
    if (predP == 0):
        predP = "H"
    elif (predP == 1):
        predP = "M"
    elif (predP == 2):
        predP = "L"
    if (predL == 0):
        predL = "H"
    elif (predL == 1):
        predL = "M"
    elif (predL == 2):
        predL = "L"
    if (predN == 0):
        predN = "H"
    elif (predN == 1):
        predN = "M"
    elif (predN == 2):
        predN = "L"
    t.sleep(1)
    print("\nUsing Decision Tree Classifier: ", predD)
    t.sleep(1)
    print("Using Random Forest Classifier: ", predR)
    t.sleep(1)
    print("Using Linear Model Perceptron: ", predP)
    t.sleep(1)
    print("Using Linear Model Logisitic Regression: ", predL)
    t.sleep(1)
    print("Using Neural Network MLP Classifier: ", predN)
    print("\nExiting...")
    t.sleep(1)
else:
    print("Exiting..")
    t.sleep(1)