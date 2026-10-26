import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Campus Facility Reservation Analytics", page_icon="🏫", layout="wide")
st.title("🏫 Campus Facility Reservation & Monitoring")
st.caption("Data Mining and Systems Analysis · Decision-support dashboard")

@st.cache_data
def demo_data(n=510):
    r=np.random.default_rng(42)
    d=pd.DataFrame({"Reservation_ID":[f"RES{i+1000}" for i in range(n)],
      "Facility_Type":r.choice(["Classroom","Computer Lab","Auditorium","Gymnasium","Conference Room","Sports Field"],n),
      "User_Category":r.choice(["Student","Faculty","Staff","Student Organization","External Guest"],n),
      "Purpose":r.choice(["Class Activity","Meeting","Seminar","Sports Event","Org Event","Exam/Review"],n),
      "Day_of_Week":r.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],n),
      "Time_Slot":r.choice(["Morning","Afternoon","Evening"],n),
      "Lead_Time_Days":r.integers(0,45,n),"Participants":r.integers(5,120,n),
      "Duration_Hours":r.integers(1,13,n),
      "Usage_Status":r.choice(["Used","Cancelled","No-Show"],n,p=[.60,.25,.15])})
    d["Actual_Attendance"]=np.where(d.Usage_Status=="Used",(d.Participants*r.uniform(.5,1,n)).round(),np.nan)
    return d

with st.sidebar:
    st.header("Data source")
    uploaded=st.file_uploader("Upload facility_reservations.csv",type="csv")
raw=pd.read_csv(uploaded) if uploaded else demo_data()
df=raw.copy()
if "User_Type" not in df and "User_Category" in df: df["User_Type"]=df["User_Category"]
if "Lead_Time" not in df and "Lead_Time_Days" in df: df["Lead_Time"]=df["Lead_Time_Days"]
if "Lead_Time_Days" not in df and "Lead_Time" in df: df["Lead_Time_Days"]=df["Lead_Time"]
needed=[c for c in ["Usage_Status","Participants"] if c not in df]
if needed: st.error("CSV is missing required columns: "+", ".join(needed)); st.stop()
original=len(df); dup=int(df.duplicated().sum()); df=df.drop_duplicates().copy()
for c in df.select_dtypes(include=["object","string"]).columns: df[c]=df[c].astype("string").str.strip()
df["Participants"]=pd.to_numeric(df["Participants"],errors="coerce")
df.loc[df["Participants"]<=0,"Participants"]=np.nan
lead="Lead_Time_Days" if "Lead_Time_Days" in df else "Lead_Time"
if lead in df:
    df[lead]=pd.to_numeric(df[lead],errors="coerce"); df.loc[df[lead]<0,lead]=np.nan
pre=len(df); df=df.dropna(subset=["Usage_Status","Participants"]).copy(); dropped=pre-len(df)
def outcome(x):
    x=str(x).lower().strip()
    return "Completed" if x=="used" else ("Not Completed" if x in ["cancelled","canceled","no-show","no show","noshow"] else np.nan)
df["Reservation_Outcome"]=df["Usage_Status"].map(outcome).astype("string")
df=df.dropna(subset=["Reservation_Outcome"]).copy()
if df.empty: st.error("No valid rows remain after cleaning."); st.stop()
if lead in df: df["Lead_Time_Category"]=pd.cut(df[lead],[-np.inf,2,7,14,np.inf],labels=["Last Minute","Short Lead","Medium Lead","Early Booking"])
if "Actual_Attendance" in df:
    df["Actual_Attendance"]=pd.to_numeric(df["Actual_Attendance"],errors="coerce")
    df["Attendance_Ratio"]=df["Actual_Attendance"]/df["Participants"].where(df["Participants"]>0)

m=st.columns(4)
for col,label,value in zip(m,["Cleaned reservations","Facilities","Duplicates removed","Rows dropped"],[len(df),df["Facility_Type"].nunique() if "Facility_Type" in df else "—",dup,dropped]): col.metric(label,value)
st.caption("Data source: uploaded CSV" if uploaded else "Data source: synthetic demonstration data (not real campus records)")
overview,cleaning,eda,model,req=st.tabs(["Overview","Data cleaning","Exploratory analysis","Prediction model","Requirements & use case"])
with overview:
    st.subheader("Dataset preview"); st.dataframe(df.head(10),use_container_width=True)
    a,b=st.columns(2)
    a.markdown("**Data types**"); a.dataframe(raw.dtypes.astype(str).rename("Type"))
    b.markdown("**Numerical summary**"); b.dataframe(raw.describe().T,use_container_width=True)
    cat=raw.select_dtypes(include=["object","category","string"])
    if not cat.empty: st.markdown("**Categorical summary**"); st.dataframe(cat.describe().T,use_container_width=True)
with cleaning:
    st.subheader("Cleaning summary")
    st.write({"Original rows":original,"Duplicates removed":dup,"Invalid/missing rows dropped":dropped,"Cleaned rows":len(df)})
    st.write("Rules: remove duplicates, trim text, mark non-positive participant counts and negative lead times as missing, and exclude records without a valid outcome or participant count.")
    st.dataframe(df.isna().sum().rename("Missing values"))
    st.download_button("Download cleaned CSV",df.to_csv(index=False).encode(),"facility_reservations_clean.csv","text/csv")
with eda:
    a,b=st.columns(2)
    a.subheader("Outcome distribution"); a.bar_chart(df["Reservation_Outcome"].value_counts())
    group="Facility_Type" if "Facility_Type" in df else "Reservation_Outcome"
    b.subheader("Outcome rate by "+group+" (%)")
    b.dataframe((pd.crosstab(df[group],df.Reservation_Outcome,normalize="index")*100).round(2))
    if "Facility_Type" in df:
        fig,ax=plt.subplots(figsize=(10,4)); sns.countplot(data=df,x="Facility_Type",hue="Reservation_Outcome",ax=ax)
        ax.tick_params(axis="x",rotation=25); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    if lead in df:
        fig,ax=plt.subplots(figsize=(8,4)); sns.boxplot(data=df,x="Reservation_Outcome",y=lead,ax=ax)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    if "Lead_Time_Category" in df: st.dataframe((pd.crosstab(df.Lead_Time_Category,df.Reservation_Outcome,normalize="index")*100).round(2))
with model:
    st.subheader("Decision tree classifier")
    predictors=[c for c in [lead,"Participants","Facility_Type","User_Type","Lead_Time_Category"] if c in df]
    X=pd.get_dummies(df[predictors],dummy_na=True); y=df.Reservation_Outcome
    if y.nunique()<2 or y.value_counts().min()<2: st.warning("Need at least two outcome classes with two records each.")
    else:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        clf=DecisionTreeClassifier(max_depth=4,random_state=42).fit(Xtr,ytr); pred=clf.predict(Xte)
        st.metric("Accuracy",f"{accuracy_score(yte,pred):.1%}")
        st.dataframe(pd.DataFrame(classification_report(yte,pred,output_dict=True,zero_division=0)).T)
        st.dataframe(pd.DataFrame(confusion_matrix(yte,pred,labels=clf.classes_),index=clf.classes_,columns=clf.classes_))
        fig,ax=plt.subplots(figsize=(14,7)); plot_tree(clf,feature_names=list(X.columns),class_names=list(clf.classes_),filled=True,rounded=True,ax=ax,fontsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.markdown("**Try a reservation**")
        with st.form("prediction"):
            vals={}
            for c in predictors:
                if c in [lead,"Participants"]: vals[c]=st.number_input(c,min_value=0.0,value=7.0 if c==lead else 30.0)
                else:
                    opts=sorted(df[c].dropna().astype(str).unique()); vals[c]=st.selectbox(c,opts if len(opts) else ["Unknown"])
            go=st.form_submit_button("Predict outcome")
        if go:
            row=pd.DataFrame([vals])
            if "Lead_Time_Category" in predictors: row["Lead_Time_Category"]=pd.cut(pd.to_numeric(row[lead]),[-np.inf,2,7,14,np.inf],labels=["Last Minute","Short Lead","Medium Lead","Early Booking"]).astype(str)
            row=pd.get_dummies(row,dummy_na=True).reindex(columns=X.columns,fill_value=0)
            st.success("Predicted outcome: "+clf.predict(row)[0])
            st.caption("This is an estimate, not a guaranteed outcome.")
with req:
    st.subheader("Functional requirements")
    st.dataframe(pd.DataFrame([
      ["FR-DM-01","Evaluate reservation risk at submission using the Decision Tree model and flag high-risk reservations."],
      ["FR-DM-02","Send a re-confirmation prompt 24 hours before high-risk events."],
      ["FR-DM-03","Validate participant counts against facility capacity and reject zero or negative values."]],
      columns=["Requirement ID","Functional requirement"]))
    st.subheader("Use case overview")
    st.graphviz_chart("digraph { rankdir=LR; user [label=\"Facility User\",shape=box]; admin [label=\"Facility Administrator\",shape=box]; submit [label=\"Submit Reservation\",shape=ellipse]; risk [label=\"Evaluate Reservation Risk\",shape=ellipse]; confirm [label=\"Send Confirmation Prompt\",shape=ellipse]; dash [label=\"View Analytics Dashboard\",shape=ellipse]; user->submit; submit->risk; risk->confirm; admin->dash; admin->risk; }")
    st.info("Requirements are proposals from the notebook scenario. Live bookings and automated messaging are not integrated.")
