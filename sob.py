import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn import svm
from sklearn import metrics 
from sklearn import preprocessing 



with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "Datasets", "Pre-Processing", "Modelling", "Implementation"],  # required
        icons=["house","folder", "file-bar-graph", "card-list", "calculator"],  # optional
        menu_icon="menu-up",  # optional
        default_index=0,  # optional
        )


if selected == "Home":
    st.title(f'Aplikasi Web Data Mining')
    st.write(""" ### Klasifikasi tingkat kematian bayi (kehamilan) menggunakan Metode Decision tree, Random forest, dan SVM
    """)
    #img = Image.open('jantung.jpg')
    #st.image(img, use_column_width=False)
    st.write('Cardiotocograms (CTGs) adalah pilihan yang sederhana dan terjangkau untuk menilai kesehatan janin, memungkinkan profesional kesehatan untuk mengambil tindakan untuk mencegah kematian anak dan ibu. Peralatan itu sendiri bekerja dengan mengirimkan pulsa ultrasound dan membaca responsnya, sehingga menjelaskan detak jantung janin (FHR), gerakan janin, kontraksi rahim, dan banyak lagi.')


if selected == "Datasets":
    st.title(f"{selected}")
    data_hf = pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/machine-Learning/main/Global%20cancer%20incidence%20both%20sexes%20(1).csv")
    st.write("Dataset Fetal HEalth : ", data_hf) 
    st.write('Jumlah baris dan kolom :', data_hf.shape)
    X=data_hf.iloc[:,0:12].values 
    y=data_hf.iloc[:,12].values
    st.write('Dataset Description :')
    st.write('1. baseline_value: Baseline Fetal Heart Rate (FHR)')
    st.write('2. accelerations: Number of accelerations per second')
    st.write('3. fetal_movement: Number of fetal movements per second')
    st.write('4. uterine_contractions: Number of uterine contractions per second')
    st.write('5. light_decelerations: Number of LDs per second')
    st.write('6. severe_decelerations: Number of SDs per second')
    st.write('7. prolongued_decelerations: Number of PDs per second')
    st.write('8. abnormal_short_term_variability: Percentage of time with abnormal short term variability')
    st.write('9. mean_value_of_short_term_variability: Mean value of short term variability')
    st.write('10. spercentage_of_time_with_abnormal_long_term_variabilityex: Percentage of time with abnormal long term variability')
    
    st.write("Dataset Fetal Health Download : (https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) ")

if selected == "Pre-Processing":
    st.title(f"{selected}")
    data_hf = pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/machine-Learning/main/Global%20cancer%20incidence%20both%20sexes%20(1).csv")
    X=data_hf.iloc[:,0:5].values 
    y=data_hf.iloc[:,5].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    st.write("Hasil Preprocesing : ", scaled)

    #Train and Test split
    X_train,X_test,y_train,y_test=train_test_split(scaled,y,test_size=0.3,random_state=0)


if selected == "Modelling":
    st.title(f"{selected}")
    st.write(""" ### Decision Tree, Random Forest, SVM """)
    data_hf = pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/machine-Learning/main/Global%20cancer%20incidence%20both%20sexes%20(1).csv")
    X=data_hf.iloc[:,0:5].values 
    y=data_hf.iloc[:,5].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform

    #Train and Test split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    
    # Decision Tree
    decision_tree = DecisionTreeClassifier() 
    decision_tree.fit(X_train, y_train)  
    Y_pred = decision_tree.predict(X_test) 
    accuracy_dt=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
    
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for DecisionTree\n',cm)
    print('accuracy_DecisionTree: %.3f' %accuracy)
    print('precision_DecisionTree: %.3f' %precision)
    print('recall_DecisionTree: %.3f' %recall)
    print('f1-score_DecisionTree : %.3f' %f1)
    
    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    Y_prediction = random_forest.predict(X_test)
    accuracy_rf=round(accuracy_score(y_test,Y_prediction)* 100, 2)
    acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
    
    cm = confusion_matrix(y_test, Y_prediction)
    accuracy = accuracy_score(y_test,Y_prediction)
    precision =precision_score(y_test, Y_prediction,average='micro')
    recall =  recall_score(y_test, Y_prediction,average='micro')
    f1 = f1_score(y_test,Y_prediction,average='micro')
    print('Confusion matrix for Random Forest\n',cm)
    print('accuracy_random_Forest : %.3f' %accuracy)
    print('precision_random_Forest : %.3f' %precision)
    print('recall_random_Forest : %.3f' %recall)
    print('f1-score_random_Forest : %.3f' %f1)
    
    #SVM
    SVM = svm.SVC(kernel='linear') 
    SVM.fit(X_train, y_train)
    Y_prediction = SVM.predict(X_test)
    accuracy_SVM=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_SVM = round(SVM.score(X_train, y_train) * 100, 2)
    
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for SVM\n',cm)
    print('accuracy_SVM : %.3f' %accuracy)
    print('precision_SVM : %.3f' %precision)
    print('recall_SVM : %.3f' %recall)
    print('f1-score_SVM : %.3f' %f1)
    st.write("""
    #### Akurasi:""" )
    results = pd.DataFrame({
        'Model': ['Decision Tree','Random Forest','SVM'],
        'Score': [ acc_decision_tree,acc_random_forest, acc_SVM ],
        'Accuracy_score':[accuracy_dt,accuracy_rf,accuracy_SVM]})
    
    result_df = results.sort_values(by='Accuracy_score', ascending=False)
    result_df = result_df.reset_index(drop=True)
    result_df.head(9)
    st.write(result_df)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(['Decision Tree', 'Random Forest','SVM'],[accuracy_dt, accuracy_rf, accuracy_SVM])
    plt.show()
    st.pyplot(fig)



if selected == "Implementation":
    st.title(f"{selected}")
    st.write("""
            ### Pilih Metode yang anda inginkan :"""
            )
    algoritma=st.selectbox('Pilih', ('Decision Tree','Random Forest','SVM'))

    data_hf = pd.read_csv("https://raw.githubusercontent.com/08-Ahlaqul-Karimah/machine-Learning/main/Global%20cancer%20incidence%20both%20sexes%20(1).csv")
    X=data_hf.iloc[:,0:5].values 
    y=data_hf.iloc[:,5].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform

    #Train and Test split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    
    # Decision Tree
    decision_tree = DecisionTreeClassifier() 
    decision_tree.fit(X_train, y_train)  
    Y_pred = decision_tree.predict(X_test) 
    accuracy_dt=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
    
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for DecisionTree\n',cm)
    print('accuracy_DecisionTree: %.3f' %accuracy)
    print('precision_DecisionTree: %.3f' %precision)
    print('recall_DecisionTree: %.3f' %recall)
    print('f1-score_DecisionTree : %.3f' %f1)
    
    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    Y_prediction = random_forest.predict(X_test)
    accuracy_rf=round(accuracy_score(y_test,Y_prediction)* 100, 2)
    acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
    
    cm = confusion_matrix(y_test, Y_prediction)
    accuracy = accuracy_score(y_test,Y_prediction)
    precision =precision_score(y_test, Y_prediction,average='micro')
    recall =  recall_score(y_test, Y_prediction,average='micro')
    f1 = f1_score(y_test,Y_prediction,average='micro')
    print('Confusion matrix for Random Forest\n',cm)
    print('accuracy_random_Forest : %.3f' %accuracy)
    print('precision_random_Forest : %.3f' %precision)
    print('recall_random_Forest : %.3f' %recall)
    print('f1-score_random_Forest : %.3f' %f1)
    
    #SVM
    SVM = svm.SVC(kernel='linear') 
    SVM.fit(X_train, y_train)
    Y_prediction = SVM.predict(X_test)
    accuracy_SVM=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_SVM = round(SVM.score(X_train, y_train) * 100, 2)
    
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for SVM\n',cm)
    print('accuracy_SVM : %.3f' %accuracy)
    print('precision_SVM : %.3f' %precision)
    print('recall_SVM : %.3f' %recall)
    print('f1-score_SVM : %.3f' %f1)
        
    st.write("""
            ### Input Data :"""
            )
    index = st.sidebar.number_input("index =", min_value=0 ,max_value=33)
    rank = st.sidebar.number_input("rank =", min_value=1, max_value=33)
    cancer = st.sidebar.number_input("cancer =", min_value=0 , max_value=34)
    new_cases_in_2020 = st.sidebar.number_input("new cases in 2020 =", min_value=179000, max_value=18001000)
    ofallcancers = st.sidebar.number_input("of all cancers =", min_value=0.1, max_value=125000)
    submit = st.button("Submit")
    if submit :
        if algoritma == 'Decision Tree' :
            X_new = np.array([[index, rank, cancer, new_cases_in_2020_ofallcancers]])
            prediksi = decision_tree.predict(X_new)
            if prediksi == 1 :
                st.write(""" ## Hasil Prediksi : resiko bayi meninggal tinggi""")
            else : 
                st.write("""## Hasil Prediksi : resiko bayi meninggal rendah""")
        elif algoritma == 'Random Forest' :
            X_new = np.array([[index, rank, cancer, new_cases_in_2020_ofallcancers]])
            prediksi = random_forest.predict(X_new)
            if prediksi == 1 :
                st.write("""## Hasil Prediksi : resiko bayi meninggal tinggi""")
            else : 
                st.write("""## Hasil Prediksi : resiko bayi meninggal rendah""")
        else :
            X_new = np.array([[index, rank, cancer, new_cases_in_2020_ofallcancers]])
            prediksi = SVM.predict(X_new)
            if prediksi == 1 :
                st.write("""## Hasil Prediksi : resiko bayi meninggal tinggi""")
            else : 
                st.write("""## Hasil Prediksi : resiko bayi meninggal rendah""")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib



st.title("PENAMBANGAN DATA")
st.write("By: Febrian Achmad Syahputra")
st.write("Grade: Penambangan Data C")
upload_data,deskripsi, preporcessing, modeling, implementation = st.tabs(["Upload Data",'deskripsi' ,"Prepocessing", "Modeling", "Implementation"])


with upload_data:
    st.write("""# Upload File""")
    st.write("Dataset yang digunakan adalah harumanis mango classification dataset yang diambil dari https://www.kaggle.com/code/kwonnnyr/harumanis-mango-classification-with-2-method/data")
    st.write("Pada Dataset yang saya gunakan terdapat 3 fitur yaitu weight,lenght,dan circumference. Pada weight akan menunjukan berat dari mangga ,kemudian pada lenght akan menunjukan panjang dari mangga , dan yang terakhir terdapat circumference yang menunjukan lingkar dari mangga ")

    
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name) 
        st.dataframe(df)

with deskripsi:

    st.write("Dataset yang digunakan adalah harumanis mango classification dataset yang diambil dari https://www.kaggle.com/code/kwonnnyr/harumanis-mango-classification-with-2-method/data")
    st.write("Pada Dataset yang saya gunakan terdapat 3 fitur yaitu weight,lenght,dan circumference. Pada weight akan menunjukan berat dari mangga ,kemudian pada lenght akan menunjukan panjang dari mangga , dan yang terakhir terdapat circumference yang menunjukan lingkar dari mangga ")

with preporcessing:
    st.write("""# Preprocessing""")
    df[['No','Weight','Length','Circumference','Grade']].agg(['min','max'])
    X = df.drop(labels = ['Grade','No'],axis = 1)
    y = df['Grade']
    "### Normalize data hasil"
    X

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape


    labels = pd.get_dummies(df.Grade).columns.values.tolist()
    
    "### Label"
    labels

    # """## Normalisasi MinMax Scaler"""


    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    y

    le.inverse_transform(y)

with modeling:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # X_train = cv.fit_transform(X_train)
    # X_test = cv.fit_transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

# with modeling:

#     st.markdown("# Model")
#     # membagi data menjadi data testing(20%) dan training(80%)
    # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

#     # X_train.shape, X_test.shape, y_train.shape, y_test.shape

#     nb = st.checkbox("Metode Naive Bayes")
#     knn = st.checkbox("Metode KNN")
#     dt = st.checkbox("Metode Decision Tree")
#     sb = st.button("submit")

#     #Naive Bayes
#     # Feature Scaling to bring the variable in a single scale
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)

#     GaussianNB(priors=None)
#     # Fitting Naive Bayes Classification to the Training set with linear kernel
#     nvklasifikasi = GaussianNB()
#     nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

#     # Predicting the Test set results
#     y_pred = nvklasifikasi.predict(X_test)
        
#     y_compare = np.vstack((y_test,y_pred)).T
#     nvklasifikasi.predict_proba(X_test)

#     akurasi = round(100 * accuracy_score(y_test, y_pred))

#     #Decision tree
#     dt = DecisionTreeClassifier()
#     dt.fit(X_train, y_train)

#     # prediction
#     dt.score(X_test, y_test)
#     y_pred = dt.predict(X_test)
#     #Accuracy
#     akur = round(100 * accuracy_score(y_test,y_pred))

#     K=10
#     knn=KNeighborsClassifier(n_neighbors=K)
#     knn.fit(X_train,y_train)
#     y_pred=knn.predict(X_test)

#     skor_akurasi = round(100 * accuracy_score(y_test,y_pred))
    

#     if nb:
#         if sb:

#             """## Naive Bayes"""
            
#             st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))

#     if knn:
#         if sb:
#             """## KNN"""

#             st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    
#     if dt:
#         if sb:
#             """## Decision Tree"""
#             st.write('Model Decission Tree Accuracy Score: {0:0.2f}'.format(akur))

with implementation:
    st.write("# Implementation")
    Weight = st.number_input('Masukkan Berat Mangga')
    Length = st.number_input('Masukkan Panjnag Mangga')
    circumference = st.number_input('Masukkan Lingkar Mangga')

    def submit():
        # input
        inputs = np.array([[
            Weight,Length,circumference
        ]])
        # st.write(inputs)
        # baru = pd.DataFrame(inputs)
        # input = pd.get_dummies(baru)
        # st.write(input)
        # inputan = np.array(input)
        # import label encoder
        le = joblib.load("le.save")
        model1 = joblib.load("tre.joblib")
        y_pred3 = model1.predict(inputs)
        if le.inverse_transform(y_pred3)[0]==1:
            hasilakhir='B'
        else :
            hasilakhir='A'
        st.write(f"Berdasarkan data yang Anda masukkan, maka mangga dinyatakan mangga memiliki grade: {hasilakhir}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()

