
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pydeck as pdk
import altair as alt
st.set_page_config(layout="wide")
#import data
data_in=pd.read_csv('changed_input.csv')  # input csv file with columns of input variables
data_out=pd.read_csv('changed_output.csv') #output csv file with column of results
X_all=np.array(data_in)
y_all=np.array(data_out)

#import more data
cities = pd.read_csv('cali_input_.csv')
lat = cities['lat'].values
lon = cities['long'].values
area = cities['area'].values

#sidebar code
st.sidebar.markdown("## Select Data Intensity and Slope Angle")
names = cities['Region'].values
select_event = st.sidebar.selectbox(' Which city do you want to modify?', names)
str_t0 = st.sidebar.slider('Rainfall Intensity(mm/hr)', 1, 100, 4)
t0 = np.log(float(str_t0))
slope_angle = st.sidebar.slider('Slope Angle(degrees)', 20, 35, 28)
#modify data using sidebar
city_index = names.tolist().index(select_event)
X_all[440+city_index, 0] = t0
X_all[440+city_index, 1] = slope_angle
#model code
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
# normalize the data attributes
X_all = scaler_X.fit_transform(X_all)
X=np.array(X_all[1:353,:]) #specify portion of data to use for training
y_all = scaler_y.fit_transform(y_all)
y=np.array(y_all[1:353,:]) #repeat here
#X_all.shape

model = load_model('rainfall_model.h5')
test_x=X_all[440:456,:] # specify testing dataset outside of training dataset
predictions=model.predict(test_x, batch_size=10, verbose=0)
y_pred=scaler_y.inverse_transform(predictions)
#print("ADFSEIJADOIFW", y_pred)
#print(y_pred)

#title and graphs
st.title('Rainfall-induced Landslide simulator')
st.markdown("""
 * Landslides are frequently caused by rainfall. The two largest parameters affecting when they occur are rainfall intensity and slope angle.
 * Use the menu to the left to modify a cities rainfall intensity and slope angle which the city rests on.
 * Each cities failure time is shown below in the bar chart.
""")
#col1, col2 = st.beta_columns(2)
#col1.dataframe(failure)
#col2.dataframe(names)
#st.bar_chart(data)
#st.x_label
data = np.power(10, y_pred)
data = data.flatten()
data_tuples =   list(zip(names, data))

#print(cities)
#data = pd.DataFrame({'a': names, 'b': data})
df = pd.DataFrame(
  data_tuples,
 columns=['California cities', 'Failure Time in hours'])
c = alt.Chart(df).mark_bar().encode(x='California cities', y='Failure Time in hours', size='Failure Time in hours', color='Failure Time in hours', tooltip=['California cities', 'Failure Time in hours'])

st.altair_chart(c, use_container_width=True)

# #print(failure)
# line_chart = alt.Chart(data).mark_line(interpolate='basis').encode(
#     alt.X('x', title='Year'),
#     alt.Y('y', title='Amount in liters'),
#     color='category:N'
# ).properties(
#     title='Sales of consumer goods'
# )

# st.altair_chart(line_chart)