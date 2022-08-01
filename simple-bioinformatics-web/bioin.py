# Import Libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import rdkit
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

##########################
# Custom function
##########################
## Calculate molecular descriptors
def AromaticProportion(m):
    aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aa_count = []
    for i in aromatic_atoms:
        if i==True:
            aa_count.append(1)
    AromaticAtom = sum(aa_count)
    HeavyAtom = Descriptors.HeavyAtomCount(m)
    AR = AromaticAtom / HeavyAtom
    return AR

def generate(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)
        
    baseData = np.arange(1,1)
    i = 0
    for mol in moldata:
        
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)
        
        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion])
        
        if (i==0):
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
            i = i+1
            
    columnNames = ["MolLogP","MolWt","NumRotatableBonds","AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData, columns = columnNames)
    
    return descriptors


###########################
# Page Title
###########################

image = Image.open('solubility-logo.jpg')

st.image(image, use_column_width = True)

st.write("""
# Molecular Solubility Prediction Web App

This app predicts the **Solubility (LogS)** values of molecules!

Data obtained from John S. Delaney
""")

################################
# Input Molecules
################################

st.sidebar.header('User Input Features')

# Read SMILES input
SMILES_input = "NCCCC\nCCC\nCN"

SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
SMILES = "C\n" + SMILES # Adds C as a dummy, first item
SMILES = SMILES.split('\n')

st.header('Input SMILES')
SMILES[1:] # Skips the dummy first item

## Calculate molecular descriptors
st.header("Computed molecular descriptors")
X_desc = generate(SMILES)
X_desc[1:] # skips the dummy first item

#######################
# Pre-built model
#######################

st.sidebar.header('Machine Learning Algorithm')
ml_option = st.sidebar.selectbox('What ML algorithm to use?',('Random Forest', 'SVM'))

# Reads in saved model
# load_model = pickle.load(open('solubility_model.pkl','rb'))

# Random forest
df = pd.read_csv('https://raw.githubusercontent.com/emreustundag1618/streamlit-apps/main/simple-bioinformatics-web/delaney_solubility_with_descriptors.csv')
X = df.drop(['logS'], axis = 1)
Y = df.logS

st.header("Predicted LogS Values")

if ml_option == "Random Forest":
    st.subheader('Random Forest')
    rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
    rf.fit(X, Y)
    prediction = rf.predict(X_desc)
    prediction[1:] # skips the dummy first item
    
if ml_option == "SVM":
    st.subheader("SVM")
    svr = SVR()
    svr.fit(X, Y)
    prediction_svr = svr.predict(X_desc)
    prediction_svr[1:]
    

















