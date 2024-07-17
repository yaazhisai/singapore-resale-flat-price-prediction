import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as pt


st.title("SINGAPORE RESALE FLAT PRICE PREDICTION")
# st.markdown("""
# <style>
#     [data-testid=stSidebar] {
#         background-color:black;
#     }
# </style>
# """, unsafe_allow_html=True)

# st.markdown("""
# 	<style>
# 	.stSelectbox:first-of-type > div[data-baseweb="select"] > div {
# 	      background-color:steelblue;
#     	      padding: 10px;
# 	}
# 	</style>
# """, unsafe_allow_html=True)

df_new1=pd.read_csv('df_clean.csv')


with st.sidebar:
   option=st.selectbox("SELECT ONE:",("ANALYSIS","PREDICTION"),index=None,placeholder=" ")

if option=="ANALYSIS":
    st.write("ANALYSIS OF COUNT OF UNITS SOLD YEARWISE")
    f=plt.figure(figsize=(15,8))
    sns.countplot(x='saleyear', data=df_new1,color='red')
    plt.title('Bar Plot: Date of lease commencement')
    plt.xlabel('Saleyear')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    st.pyplot(f)

    # Line plot
    st.write("FLAT MODEL AND RESALE PRICE ANALYSIS")
    f1=plt.figure(figsize=(15,8))
    ss1=df_new1.groupby(['flat_model'])[['resale_price']].mean()
    ss1.reset_index(inplace= True)
    sns.lineplot(x='flat_model', y='resale_price', data=df_new1,color='black')
    plt.title('Line Plot: Flat Model vs Resale price')
    plt.xticks(rotation=90)
    st.pyplot(f1)

    
    st.write("FLAT TYPE AND RESALE PRICE ANALYSIS")
    f2 =plt.figure(figsize=(15,8))
    ss2=df_new1.groupby(['flat_type'])[['resale_price']].mean()
    ss2.reset_index(inplace= True)
    sns.barplot(ss2, x="flat_type", y="resale_price",color='green')
    #plt.title('Scatter Plot: Flat Type vs Resale price')
    #plt.xticks(rotation=90)
    st.pyplot(f2)
    


    # sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
    st.write("REMAINING LEASE YEAR AND RESALE PRICE ANALYSIS")
    f3=plt.figure(figsize=(15, 8))
    ss3=df_new1.groupby(['rem_lease_year'])[['resale_price']].mean()
    ss3.reset_index(inplace= True)
    sns.barplot(data=ss3,x='rem_lease_year',y='resale_price',color='yellow')
    plt.title('BAR PLOT: Remaining lease year vs Resale price')
    plt.xticks(rotation=90)
    st.pyplot(f3)

    # Line plot
    st.write("AREA AND RESALE PRICE ANALYSIS")
    f4=plt.figure(figsize=(15,8))
    ss4=df_new1.groupby(['floor_area_sqm'])[['resale_price']].mean()
    ss4.reset_index(inplace= True)
    sns.lineplot(x='floor_area_sqm', y='resale_price', data=ss4,color='green')
    plt.title('Line Plot: Floor area(sqm) vs Resale price')
    plt.xticks(rotation=90)
    st.pyplot(f4)
    
    st.write("TOWN AND RESALE PRICE ANALYSIS")
    f5=plt.figure(figsize=(15, 8))
    ss=df_new1.groupby(['town'])[['resale_price']].mean()
    ss.reset_index(inplace= True)
    sns.barplot(data=ss, x="town", y="resale_price", color="purple")
    plt.xticks(rotation=90)
    st.pyplot(f5)



elif option=='PREDICTION':
    with st.form("my-form"):
        t=df_new1['town'].unique()
        ft=df_new1['flat_type'].unique()
        
        fm=df_new1['flat_model'].unique()
        
        sr=df_new1['storey_range'].unique()
        
        floor_area_sqm=df_new1['floor_area_sqm'].unique()
        lease_commence_date=df_new1['lease_commence_date'].unique()
        salemonth=df_new1['salemonth'].unique()
        saleyear=df_new1['saleyear'].unique()
        rem_lease_year=df_new1['rem_lease_year'].unique()

        col1, col2, col3 = st.columns([5, 1,5])
        with col1:
            st.write(' ')
            town1=st.selectbox('TOWN/AREA',t,help='Designated residential area with its own amenities, infrastructure, and community facilities')
            town_en=df_new1[df_new1['town']==town1]['town_en'].iloc[0]
            flat_type1=st.selectbox('FLAT TYPE',ft,help='Classification of units by room size. They range from 2 to 5 rooms, 3Gen units, and Executive units.')
            flat_type_en=df_new1[df_new1['flat_type']==flat_type1]['flat_type_en'].iloc[0]
            storey_range1=st.selectbox('FLOOR RANGE',sr,help='Estimated range of floors the unit sold was located on')
            storey_en=df_new1[df_new1['storey_range']==storey_range1]['storey_en'].iloc[0]
            floor_area_sqm=st.selectbox('FLOOR AREA',floor_area_sqm,help='Total interior space within the unit, measured in square meters')
            flat_model1=st.selectbox('FLAT MODEL',fm,help='Classification of units by generation of which the flat was made, ranging from New Generation, DBSS, Improved, Apartment')
            flat_model_en=df_new1[df_new1['flat_model']==flat_model1]['flat_model_en'].iloc[0]


        with col3:
            st.write('  ')
            lease_commence_date1=st.selectbox('LEASE START YEAR',lease_commence_date,help='Starting point of a lease agreement, marking the beginning of the lease term during which the tenant has the right to use and occupy the leased property')
            salemonth1=st.selectbox('MONTH OF SALE',salemonth,help='The month at which sale has happened')
            saleyear1=st.selectbox('YEAR OF SALE',saleyear,help='The year at which sale has happened')
            rem_lease_year=st.selectbox('REMAINLING LEASE PERIOD',rem_lease_year,help='Remaining number of years the buyer can use this property')
            st.write(' ')
            st.write('  ')
            submit_bt = st.form_submit_button(label='Predict resale Price',use_container_width=150)
            # st.markdown('''
            #     ''', unsafe_allow_html=True)

            if submit_bt:
                with open(r'DecisionTreeRegressor_pkl','rb') as f:
                    model=pickle.load(f)
                    # print(town_en,flat_type_en,flat_model_en,storey_en,floor_area_sqm,rem_lease_year)
                    data = np.array([town_en, 
                                    storey_en,
                                    np.log(float(floor_area_sqm)), 
                                    flat_model_en,
                                    flat_type_en, 
                                    saleyear1,
                                    salemonth1,
                                    lease_commence_date1,
                                    np.log(float(rem_lease_year))]).reshape(1,-1)
                    
                    #print(data)
                    y_pred = model.predict(data)
                    #print(y_pred[0])
                    #inverse transformation 
                    price_bfr = np.exp(y_pred[0])
                    price_aft = np.round(price_bfr,2)

                    st.write(f"THE PREDICTED RESALE PRICE IS:$ {price_aft} ")


