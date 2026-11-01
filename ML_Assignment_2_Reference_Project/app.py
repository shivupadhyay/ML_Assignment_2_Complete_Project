"""Streamlit evaluation app: saved estimators are never fitted on uploaded data."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from train_models import calculate_metrics

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Classification Model Benchmark', page_icon='🧪', layout='wide')
st.markdown('''<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
[data-testid="stMetric"] {background: #f4f7fa; border: 1px solid #e4e9f0; border-radius: 12px; padding: 12px;}
.hero {padding: 20px 24px; background: linear-gradient(105deg,#17325c,#2365a0); color: white; border-radius: 18px; margin-bottom: 20px;}
.hero h1 {color: white; margin-bottom: 3px;}
.hero p {color: #dcebf8; margin-bottom: 0;}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Classification Model Benchmark</h1><p>Five classifiers · One dataset · Six evaluation metrics · No test-set retraining</p></div>', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    meta = json.loads((ROOT/'model'/'metadata.json').read_text())
    models = {name:joblib.load(ROOT/'model'/f'{slug}.joblib') for name,slug in meta['model_files'].items()}
    return meta, models

try:
    metadata, models = load_assets()
except (OSError, ValueError) as exc:
    st.error(f'Model artifacts unavailable: {exc}. Run python train_models.py first.')
    st.stop()

st.sidebar.header('Experiment controls')
choice = st.sidebar.selectbox('Select a trained model', list(models), index=0)
show_all = st.sidebar.checkbox('Show comparison of all five models', value=True)
st.sidebar.info('Upload the held-out test CSV, including all 30 feature columns and target. The app only calls predict/predict_proba.')

st.subheader('1 · Test dataset')
source = st.radio('Choose test data', ['Bundled held-out test_data.csv','Upload test CSV'], horizontal=True)
try:
    if source == 'Upload test CSV':
        uploaded = st.file_uploader('Choose a CSV test file', type=['csv'])
        if uploaded is None:
            st.info('Upload a CSV file or choose the bundled sample to begin.')
            st.stop()
        data = pd.read_csv(uploaded)
    else:
        data = pd.read_csv(ROOT/'test_data.csv')
except Exception as exc:
    st.error(f'Cannot read CSV: {exc}')
    st.stop()
features = metadata['feature_names']
required = features+[metadata['target_column']]
missing = [c for c in required if c not in data.columns]
if missing:
    st.error('Missing required columns: '+', '.join(missing))
    st.stop()
if len(data)==0:
    st.error('Empty CSV file.')
    st.stop()
if data[metadata['target_column']].isna().any():
    st.error('Target contains null values.')
    st.stop()
try:
    X = data[features].apply(pd.to_numeric, errors='raise')
    y = pd.to_numeric(data[metadata['target_column']], errors='raise').astype(int)
except (ValueError, TypeError):
    st.error('All features and the target must be numeric; target must contain 0/1.')
    st.stop()
if not set(y.unique()).issubset({0,1}):
    st.error('Target must contain only class codes 0 (malignant) and 1 (benign).')
    st.stop()
if y.nunique() < 2:
    st.warning('Only one actual class occurs in this CSV. ROC AUC is undefined; both classes are required for all six metrics.')

c1,c2,c3 = st.columns(3)
c1.metric('Test instances', len(data))
c2.metric('Input features', len(features))
c3.metric('Missing feature values', int(X.isna().sum().sum()))
with st.expander('Preview uploaded dataset and class labels'):
    st.write('Class 0 = malignant; class 1 = benign (positive class).')
    st.dataframe(data.head(10), use_container_width=True)
    st.caption('Extra columns, if supplied, are ignored; feature order is enforced from the saved training metadata.')

@st.cache_data(show_spinner=False)
def evaluate_all(test_frame):
    feature_set = test_frame[features].apply(pd.to_numeric)
    actual = test_frame[metadata['target_column']].astype(int).to_numpy()
    output={}
    for name, model in models.items():
        pred = model.predict(feature_set)
        pos_index=list(model.classes_).index(1)
        prob=model.predict_proba(feature_set)[:,pos_index]
        if len(np.unique(actual)) == 2:
            metrics=calculate_metrics(actual,pred,prob)
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
            metrics={'Accuracy':float(accuracy_score(actual,pred)), 'AUC':float('nan'),
                     'Precision':float(precision_score(actual,pred,zero_division=0)),
                     'Recall':float(recall_score(actual,pred,zero_division=0)),
                     'F1':float(f1_score(actual,pred,zero_division=0)),
                     'MCC':float(matthews_corrcoef(actual,pred))}
        output[name]={'metrics':metrics,'pred':pred,'prob':prob}
    return output

out=evaluate_all(data[required].copy())
st.subheader('2 · Selected model metrics')
selected=out[choice]
cols=st.columns(6)
for col,(key,val) in zip(cols,selected['metrics'].items()):
    col.metric(key,'N/A' if np.isnan(val) else f'{val:.4f}')

left,right=st.columns(2)
with left:
    st.markdown('**Confusion matrix**')
    fig, ax=plt.subplots(figsize=(5,3.7))
    ConfusionMatrixDisplay(confusion_matrix(y,selected['pred'],labels=[0,1]),display_labels=['Malignant (0)','Benign (1)']).plot(ax=ax,cmap='Blues',colorbar=False,values_format='d')
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown('**Classification report**')
    report=classification_report(y,selected['pred'],labels=[0,1],target_names=['Malignant (0)','Benign (1)'],output_dict=True,zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4),use_container_width=True)
    if y.nunique()==2:
        fpr,tpr,_=roc_curve(y,selected['prob'],pos_label=1)
        fig,ax=plt.subplots(figsize=(5,3))
        ax.plot(fpr,tpr,label=f'ROC AUC = {auc(fpr,tpr):.4f}')
        ax.plot([0,1],[0,1],'--',label='Chance')
        ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        plt.close(fig)

if show_all:
    st.subheader('3 · All-model comparison on the same uploaded test data')
    table=pd.DataFrame([{'ML Model Name':key,**value['metrics']} for key,value in out.items()])
    st.dataframe(table.style.format({k:'{:.4f}' for k in ['Accuracy','AUC','Precision','Recall','F1','MCC']}),use_container_width=True,hide_index=True)
    winner=table.sort_values(['F1','AUC'],ascending=False,na_position='last').iloc[0]['ML Model Name']
    st.success(f'Highest F1 on these test rows: {winner}. Compare AUC, class-wise recall and MCC before selecting a deployment model.')
    st.download_button('Download evaluation table',table.to_csv(index=False).encode(),'evaluation_results.csv','text/csv')

st.caption('Reproducible split: random_state=42, stratified 80/20. Saved pipelines contain imputation, scaling, and fitted classifier. No training is performed within this application.')
