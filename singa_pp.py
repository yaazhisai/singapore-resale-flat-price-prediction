import streamlit as st
import pickle 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as pt



st.title("SINGAPORE RESALE FLAT PRICE PREDICTION")
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color:black;
    }
</style>
""", unsafe_allow_html=True)

# st.markdown("""
# 	<style>
# 	.stSelectbox:first-of-type > div[data-baseweb="select"] > div {
# 	      background-color:steelblue;
#     	      padding: 10px;
# 	}
# 	</style>
# """, unsafe_allow_html=True)

if 'df_new1' not in st.session_state.keys():
    df_new1=pd.read_csv('df_clean.csv')
    st.session_state['df_new1']=df_new1
else:
    df_new1 = st.session_state['df_new1']

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

    st.write("TOWN VS RESALE VS FLATTYPE ANALYSIS")
    fig=sns.relplot(df_new1, x="town", y="resale_price", hue="flat_type")
    plt.xticks(rotation=90)
    st.pyplot(fig)



elif option=='PREDICTION':
    with st.form("my-form"):
        town=['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH','BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
        'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST','KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG',
        'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN','LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS',
        'PUNGGOL']
        town_dict={'ANG MO KIO':0., 'BEDOK':1., 'BISHAN':2., 'BUKIT BATOK':3., 'BUKIT MERAH':4.,'BUKIT TIMAH':6., 'CENTRAL AREA':7., 'CHOA CHU KANG':8., 'CLEMENTI':9,
        'GEYLANG':10., 'HOUGANG':11., 'JURONG EAST':12., 'JURONG WEST':13.,'KALLANG/WHAMPOA':14., 'MARINE PARADE':16., 'QUEENSTOWN':19., 'SENGKANG':21.,
        'SERANGOON':22., 'TAMPINES':23., 'TOA PAYOH':24., 'WOODLANDS':25., 'YISHUN':26.,'LIM CHU KANG':15., 'SEMBAWANG':20., 'BUKIT PANJANG':5., 'PASIR RIS':17.,
        'PUNGGOL':18.}
        flat_type=['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE','MULTI GENERATION']
        flat_type_dict={'1 ROOM':1,'3 ROOM':3,'4 ROOM':4,'5 ROOM':5,'2 ROOM':2,'EXECUTIVE':6,'MULTI GENERATION':7}
        storey_range=['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15','19 TO 21', '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30',
        '31 TO 33', '40 TO 42', '37 TO 39', '34 TO 36', '06 TO 10','01 TO 05', '11 TO 15', '16 TO 20', '21 TO 25', '26 TO 30',
        '36 TO 40', '31 TO 35', '46 TO 48', '43 TO 45', '49 TO 51']
        storey_range_dict={'10 TO 12':5., '04 TO 06':2., '07 TO 09':4., '01 TO 03':0., '13 TO 15':7.,'19 TO 21':10., '16 TO 18':8., '25 TO 27':13., '22 TO 24':12., '28 TO 30':15.,
        '31 TO 33':16., '40 TO 42':21., '37 TO 39':20., '34 TO 36':18., '06 TO 10':3.,'01 TO 05':1., '11 TO 15':6., '16 TO 20':9., '21 TO 25':11., '26 TO 30':14.,
        '36 TO 40':19., '31 TO 35':17., '46 TO 48':22., '43 TO 45':23., '49 TO 51':24.}
        floor_area_sqm=[ 31. ,  73. ,  67. ,  82. ,  74. ,  88. ,  89. ,  83. ,  68. , 75. ,  81. ,  91. ,  92. ,  97. ,  90. ,  98. ,  99. , 100. , 93. , 103. , 119. , 120. , 118. , 121. , 135. , 117. ,  45. ,
            65. ,  59. ,  70. ,  76. ,  84. , 104. , 105. , 125. , 132. , 139. , 123. , 143. , 151. ,  69. , 106. , 107. , 116. , 149. , 141. , 146. , 148. , 145. , 154. , 150. ,  29. ,  51. ,  61. ,
            63. ,  64. ,  72. ,  58. ,  66. ,  60. ,  53. ,  54. ,  56. , 77. , 133. , 131. , 115. ,  43. ,  38. ,  41. ,  85. , 111. , 101. , 112. , 137. , 127. , 147. , 163. ,  50. ,  40. ,  60.3,
            62. ,  55. ,  57. ,  52. ,  63.1, 102. ,  83.1, 126. , 140. , 142. ,  71. , 108. , 144. ,  96. , 114. , 157. , 152. , 155. , 87. , 109. , 110. ,  94. , 134. , 122. , 128. ,  78. ,  46. ,
            42. ,  49. ,  47. ,  86. , 156. ,  79. ,  80. , 124. ,  28. , 113. ,  95. , 160. , 136. ,  48. , 138. , 161. ,  39. , 130. , 159. , 206. ,  68.2,  64.9, 129. , 165. , 153. , 166. , 210. ,
            59.2,  73.1,  48.1, 174. ,  74.9, 164. , 158. ,  37. , 198. , 173. , 199. , 261. , 179. ,  69.7, 246. , 171. , 181. ,  44. , 169. , 189. ,  67.2, 222. ,  64.8, 250. ,  74.8, 215. , 237. ,
            59.1, 185. , 297. , 243. ,  69.9, 162. , 170. , 190. , 175. , 187. , 241. , 186. ,  56.4, 177. , 176. ,  75.9, 184. , 178. , 69.2, 172. , 168. , 167. , 152.4,  64.7, 195. , 188. , 225. ,
        307. , 182. , 131.1, 192. , 183. , 280. , 180. , 221. , 193. , 207. ,  89.1,  88.1, 259. , 266. ,  87.1,  34. ,  35. , 239. , 249. ,  68.8, 100.2, 208. , 189.4]
        flat_model=['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED','MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE',
        '2-ROOM', 'MULTI GENERATION', 'PREMIUM APARTMENT', 'Adjoined flat','Premium Maisonette', 'Model A2', 'DBSS', 'Type S1', 'Type S2',
        'Premium Apartment Loft', '3Gen']
        flat_model_dict={'IMPROVED':5., 'NEW GENERATION':11., 'MODEL A':7., 'STANDARD':16., 'SIMPLIFIED':15.,'MODEL A-MAISONETTE':8., 'APARTMENT':2., 'MAISONETTE':6., 'TERRACE':17.,
        '2-ROOM':0., 'MULTI GENERATION':9., 'PREMIUM APARTMENT':12., 'Adjoined flat':3.,'Premium Maisonette':14., 'Model A2':10., 'DBSS':4., 'Type S1':18., 'Type S2':19.,
        'Premium Apartment Loft':13., '3Gen':1.}
        salemonth=[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
        saleyear=[1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
        2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,2023, 2024]
        rem_lease_year=[86, 85, 87, 88, 93, 89, 94, 90, 91, 95, 81, 92, 82, 78, 84, 80, 83,76, 79, 77, 97, 96, 98, 75, 99, 74, 73, 72, 71, 70, 69, 68, 67, 66,
        65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,48, 47, 46, 45, 44, 43, 42, 41]
        
        col1, col2, col3 = st.columns([5, 1,5])
        with col1:
            st.write(' ')
            town=st.selectbox('TOWN/AREA',town,help='Designated residential area with its own amenities, infrastructure, and community facilities')
            flat_type=st.selectbox('FLAT TYPE',flat_type,help='Classification of units by room size. They range from 2 to 5 rooms, 3Gen units, and Executive units.')
            storey_range=st.selectbox('FLOOR RANGE',storey_range,help='Estimated range of floors the unit sold was located on')
            floor_area_sqm=st.selectbox('FLOOR AREA',floor_area_sqm,help='Total interior space within the unit, measured in square meters')
        with col3:
            st.write('  ')
            flat_model=st.selectbox('FLAT MODEL',flat_model,help='Classification of units by generation of which the flat was made, ranging from New Generation, DBSS, Improved, Apartment')
            salemonth=st.selectbox('MONTH OF SALE',salemonth,help='The month at which sale has happened')
            saleyear=st.selectbox('YEAR OF SALE',saleyear,help='The year at which sale has happened')
            rem_lease_year=st.selectbox('REMAINLING LEASE PERIOD',rem_lease_year,help='Remaining number of years the buyer can use this property')
            st.write(' ')
            st.write('  ')
        submit_bt = st.form_submit_button(label='Predict resale Price',use_container_width=150)
        st.markdown('''
                ''', unsafe_allow_html=True)

        if submit_bt:
            if 'model' not in st.session_state.keys():
                with open(r'linearreg_pkl','rb') as f:
                    model=pickle.load(f)
                    st.session_state['model']=model
            else:
                model = st.session_state['model'] 

            data = np.array([[ town_dict[town], 
                            flat_type_dict[flat_type], 
                            storey_range_dict[storey_range],
                            np.log(float(floor_area_sqm)), 
                            flat_model_dict[flat_model],
                            np.log(float(saleyear)),
                            salemonth,
                            np.log(float(rem_lease_year))
                        ]])
            y_pred = model.predict(data)
            #print(y_pred[0])
            #inverse transformation 
            price_bfr = np.exp(y_pred[0])
            price_aft = np.round(price_bfr,2)

            st.write(f"THE PREDICTED RESALE PRICE IS:${price_aft}")
            
