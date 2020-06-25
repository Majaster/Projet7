import streamlit as st
import time
import requests
import json
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
csfont = {'fontname':'Nexa Bold'} # Tuning font for plots
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import plotly.figure_factory as ff

from PIL import Image

import shap
# ------------------------------------------------------


# HTML requests to retrieve data -----------------------

# Test clients
response = requests.get("https://datascientist.pythonanywhere.com/clients")
content = json.loads(response.content.decode('utf-8'))
clients = pd.read_json(content)
clients = clients.loc[clients['DAYS_EMPLOYED']<0]

# User picks a client in the sidebar
client_ID = st.sidebar.selectbox('Choose a client', clients['SK_ID_CURR'])

# Client data chosen by user
response = requests.get("https://datascientist.pythonanywhere.com/api?id="+str(client_ID))
content = json.loads(response.content.decode('utf-8'))
chosen_client = pd.read_json(content)
chosen_client = chosen_client.loc[chosen_client['DAYS_EMPLOYED']<0]

# Cluster data where chosen client was predicted to be
response = requests.get("https://datascientist.pythonanywhere.com/cluster?id="+str(client_ID))
content = json.loads(response.content.decode('utf-8'))
cluster_client = pd.read_json(content)
cluster_client = cluster_client.loc[cluster_client['DAYS_EMPLOYED']<0]

# Shows descriptive informations of the chosen client in the sidebar
st.sidebar.markdown('Gender : ' + chosen_client['CODE_GENDER'].values[0])
st.sidebar.markdown('Children : ' + str(chosen_client['CNT_CHILDREN'].values[0]))
st.sidebar.markdown('Owns car : ' + chosen_client['FLAG_OWN_CAR'].values[0])
st.sidebar.markdown('Credit Amount : ' + str(np.int(chosen_client['AMT_CREDIT'].values[0])))
st.sidebar.markdown('Annuity Amount : ' + str(np.int(chosen_client['AMT_ANNUITY'].values[0])))


# Start of the main page ------------------------------

# Display an image at the top
# response = requests.get("https://datascientist.pythonanywhere.com/image")
# image = Image.open(response.content)
# st.image(image, caption=None, use_column_width=True)


# Title + markdown + text
"""
# Trust
Your dashboard to know **everything about your client** :sunglasses:
"""

"""
Choose a client on the sidebar. First table is the client's data. \n
Then you can choose several options to filter true all clients data to compare between the client, all the clients and 
the group you filtered.
"""


# Means for all clients
proba_mean = clients['Proba'].mean()
amt_credit_mean = clients['AMT_CREDIT'].mean()
amt_annuity_mean = clients['AMT_ANNUITY'].mean()
days_birth_mean = clients['DAYS_BIRTH'].mean()
days_employed_mean = clients['DAYS_EMPLOYED'].mean()
days_credit_max_mean = clients['DAYS_CREDIT_max'].mean()

# Means for cluster of chosen clients
proba_cluster_mean = cluster_client['Proba'].mean()
amt_credit_cluster_mean = cluster_client['AMT_CREDIT'].mean()
amt_annuity_cluster_mean = cluster_client['AMT_ANNUITY'].mean()
days_birth_cluster_mean = cluster_client['DAYS_BIRTH'].mean()
days_employed_cluster_mean = cluster_client['DAYS_EMPLOYED'].mean()
days_credit_max_cluster_mean = cluster_client['bureau_DAYS_CREDIT_max'].mean()

# Part of data of chosen client to display 
chosen_client_cut = chosen_client[['AMT_CREDIT','AMT_ANNUITY','DAYS_BIRTH',
                                   'DAYS_EMPLOYED','DAYS_CREDIT_max']]

st.subheader('Client data')
st.dataframe(chosen_client_cut, height=100)

# Features for chosen client
proba_client = chosen_client['Proba'].values[0]
amt_credit_client = chosen_client['AMT_CREDIT'].values[0]
amt_annuity_client = chosen_client['AMT_ANNUITY'].values[0]
days_birth_client = chosen_client['DAYS_BIRTH'].values[0]
days_employed_client = chosen_client['DAYS_EMPLOYED'].values[0]
days_credit_max_client = chosen_client['DAYS_CREDIT_max'].values[0]

# Indicator A/B/C/D depending on model probabiity
if proba_client < 0.25:
    indicator_client = 'A'
elif proba_client < 0.5:
    indicator_client = 'B'
elif proba_client < 0.75:
    indicator_client = 'C'
else:
    indicator_client = 'D'


# Show all clients
if st.checkbox('Show all clients data'):
    clients


# User can choose several features to group similar clients ('Man' by default)
st.subheader('Filter data with listed options')

# Copy of clients dataframe
clients_filtered = clients.copy()

# User chooses options from list
filters = st.multiselect('Clustering filters :',
                                 ['Men','Women','High credit amount','Days employed'], 
                                 ['Men'])
    
# Mask to filter dataframe
# selection = df['Gender'].isin(cluster_options)
if 'Men' in filters:
    clients_filtered = clients_filtered.loc[clients_filtered['CODE_GENDER']=='M']
if 'Women' in filters:
    clients_filtered = clients_filtered.loc[clients_filtered['CODE_GENDER']=='F']
if 'High credit amount' in filters:
    clients_filtered = clients_filtered.loc[clients_filtered['AMT_CREDIT'] > 5e5]

# Show cluster data
if st.checkbox('Show filtered data'):
    clients_filtered

# Features means of the cluster
proba_filtered_mean = clients_filtered['Proba'].mean()
amt_credit_filtered_mean = clients_filtered['AMT_CREDIT'].mean()
amt_annuity_filtered_mean = clients_filtered['AMT_ANNUITY'].mean()
days_birth_filtered_mean = clients_filtered['DAYS_BIRTH'].mean()
days_employed_filtered_mean = clients_filtered['DAYS_EMPLOYED'].mean()
days_credit_max_filtered_mean = clients_filtered['DAYS_CREDIT_max'].mean()




# ----------------------------------------------

#                  DASHBOARD

# ----------------------------------------------
st.subheader('Main Dashboard')

fig = plt.figure(constrained_layout=True)
plt.rc('axes', edgecolor='#1F94DA') # color of axis

# colors of class letters (A = excellent/green, D = bad/red)
colors_dict = {'A':'#20b437', 'B':'#ddd13b', 'C':'#ed7919', 'D':'#dc1029'} 

# colors to indicate good or bad indicators (green / red)
good = '#20b437' 
bad = '#dc1029'

# matplotlib gridspec
spec = fig.add_gridspec(3, 4, wspace=0.0, hspace=0.0) 

# Create our own color map (blue gradient)
N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(50/256, 0.05, N)
vals[:, 1] = np.linspace(75/256, 0.08, N)
vals[:, 2] = np.linspace(180/256, 0.3, N)
custom_cm = ListedColormap(vals)

# Graph 1 : indicator + message ---------------------------------
ax = fig.add_subplot(spec[:-1, :2])
# ax.set_facecolor(color)

lim = plt.xlim() + plt.ylim() 
ax.imshow([[0,0],[1,1]], cmap=custom_cm, interpolation='bicubic', extent=lim)

# Indicator A/B/C/D
ax.annotate(indicator_client, (0.5, 0.5), xycoords='axes fraction', va='center', ha='center', 
           fontsize=150, color=colors_dict[indicator_client]) #, **csfont)

# Message "Accepter" or "Refuser"
if proba_client >= 0.5:
    ax.annotate("REFUSER", (0.25, 0.1), xycoords='axes fraction', 
                va='center', ha='center', fontsize=14, color=bad)
else:
    ax.annotate("ACCEPTER", (0.25, 0.1), xycoords='axes fraction', 
                va='center', ha='center', fontsize=14, color=good)

# Threshold (all data - gray)
ax.annotate("%.2f"%(proba_mean), (0.85, 0.08), xycoords='axes fraction', va='center', ha='center',
           fontsize=14, color='#a7b1c8')

# Little arrow near client's target to show if it's superior (bad/red) or inferior (good/green) than threshold
if proba_client >= proba_mean:
    ax.annotate("%.2f"%(proba_client), (0.85, 0.15), 
                xycoords='axes fraction', va='center', ha='center', fontsize=14, color=bad)
    ax.scatter(0.74, 0.15, marker='^', s=20, color=bad) 
else:
    ax.annotate("%.2f"%(proba_client), (0.85, 0.15), 
                xycoords='axes fraction', va='center', ha='center', fontsize=14, color=good)
    ax.scatter(0.74, 0.15, marker='v', s=20, color=good) 

ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_xticks([])
ax.set_yticks([])  

# Graph 2 : days_birth ---------------------------------------
ax2 = fig.add_subplot(spec[-1, :2])
ax2.set_facecolor('#e4eef9')

x = [1, 2, 3, 4]
xlabels = ['All','Filters','Similar','Client']
y = np.abs([days_birth_mean / 365., days_birth_filtered_mean / 365., 
            days_birth_cluster_mean / 365., days_birth_client / 365.])

plt.plot([x[0],x[0]], [0,y[0]], lw=1.5, color='#443028')
plt.plot([x[1],x[1]], [0,y[1]], lw=1.5, color='#94563a')
plt.plot([x[2],x[2]], [0,y[2]], lw=1.5, color='#cb7b48')
plt.plot([x[3],x[3]], [0,y[3]], lw=1.5, color='#f1943c')
plt.scatter(x[0],y[0], lw=16, color='#443028')
plt.scatter(x[1],y[1], lw=16, color='#94563a')
plt.scatter(x[2],y[2], lw=16, color='#cb7b48')
plt.scatter(x[3],y[3], lw=16, color='#f1943c')
ax2.annotate(str(np.round(y[0],1)), (x[0], y[0]+14), ha='center', fontsize=8, color='#443028') #, **csfont)
ax2.annotate(str(np.round(y[1],1)), (x[1], y[1]+14), ha='center', fontsize=8, color='#94563a') #, **csfont)
ax2.annotate(str(np.round(y[2],1)), (x[2], y[2]+14), ha='center', fontsize=8, color='#cb7b48') #, **csfont)
ax2.annotate(str(np.round(y[3],1)), (x[3], y[3]+14), ha='center', fontsize=8, color='#f1943c') #, **csfont)

ax2.set_xlim([0,5])
ax2.set_ylim([0,np.max(y)*2])
ax2.set_xticks(x)
ax2.set_xticklabels(xlabels)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax2.annotate('Age (years)', (0.5, 0.83), xycoords='axes fraction', ha='center', fontsize=11, color='#443028') #, **csfont)
# ax2.set_title('Age (years)', fontsize=12, color='#0F2354', **csfont)

# Graph 3 : amt_credit_client --------------------------------------
ax3 = fig.add_subplot(spec[0, 2:])
ax3.set_facecolor('#e4eef9')

x = [1, 2, 3, 4]
xlabels = ['All','Group','Similar','Client']
y = [amt_credit_mean, amt_credit_filtered_mean, amt_credit_cluster_mean, amt_credit_client]

plt.bar(x, y, color=['#194575','#3f80de','#2cbee5','#f1943c'])

ax3.set_xlim([0,5])
ax3.set_ylim([0,np.max(y)*1.5])
ax3.set_xticks(x)
ax3.set_xticklabels(xlabels)
ax3.tick_params(axis='both', which='major', labelsize=8)
ax3.annotate('Credit amount', (0.5, 0.83), xycoords='axes fraction', ha='center', fontsize=11, color='#194575') #, **csfont)

# Graph 4 : amount_annuity -----------------------------------------
ax4 = fig.add_subplot(spec[1, 2:])
ax4.set_facecolor('#e4eef9')

x = [1, 2, 3, 4]
xlabels = ['All','Group','Similar','Client']
y = [amt_annuity_mean, amt_annuity_filtered_mean, amt_annuity_cluster_mean, amt_annuity_client]

plt.bar(x, y, color=['#721935','#e22768','#f92236','#f1943c'])

ax4.set_xlim([0,5])
ax4.set_ylim([0,np.max(y)*1.5])
ax4.set_xticks(x)
ax4.set_xticklabels(xlabels)
ax4.tick_params(axis='both', which='major', labelsize=8)
ax4.annotate('Annuity amount', (0.5, 0.83), xycoords='axes fraction', ha='center', fontsize=11, color='#721935')#, **csfont)

# Graph 5 : days_employed -------------------------------------------
ax5 = fig.add_subplot(spec[2, 2:])
ax5.set_facecolor('#e4eef9')

x = [1, 2, 3, 4]
xlabels = ['All','Group','Similar','Client']
y = np.abs([days_employed_mean, days_employed_filtered_mean, days_employed_cluster_mean, days_employed_client])

plt.bar(x, y, color=['#245148','#24c18a','#2aca42','#f1943c'])

ax5.set_xlim([0,5])
ax5.set_ylim([0,np.max(y)*1.5])
ax5.set_xticks(x)
ax5.set_xticklabels(xlabels)
ax5.tick_params(axis='both', which='major', labelsize=8)
ax5.annotate('Days employed', (0.5, 0.83), xycoords='axes fraction', ha='center', fontsize=11, color='#245148') #, **csfont)

# Display the dashboard
st.pyplot()

"""
**Note**

A : confiance élevée \n
B : confiance plutôt élevée \n
C : confiance plutôt basse \n
D : confiance basse \n
"""



# ----------------------------------------------

#                 SHAP VALUES

# ----------------------------------------------
# st.subheader('Interpretability')

# Plot SHAP values for the cluster where the client was predicted to be
# dd = pd.read_csv('dd.csv')
# arr = np.load('arr.npy')

# fig, ax = plt.subplots(constrained_layout=True)
# shap.summary_plot(arr, dd)

# ax.tick_params(axis='both', which='major', labelsize=8)
# ax.set_xlabel('yo',fontsize=10)
# st.pyplot()




# Progress bar ------------------------------------
# 'Starting a long computation...'
# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# n = 10
# for i in range(n):
#     # Update the progress bar with each iteration.
#     factor = np.int(100/n)
#     latest_iteration.text(f'{(i+1) * factor}%')
#     bar.progress((i+1) * factor)
#     time.sleep(0.05)

# '...and now we\'re done!'


# Plot with altair ---------------------------------
# df = pd.DataFrame(
#     np.random.randn(200, 3),
#     columns=['a', 'b', 'c'])

# c = alt.Chart(df).mark_circle().encode(
#     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
# c


# Plot with Plotly ---------------------------------
# Add histogram data
# x1 = np.random.randn(200) - 2
# x2 = np.random.randn(200)
# x3 = np.random.randn(200) + 2

# # Group data together
# hist_data = [x1, x2, x3]
# group_labels = ['Group 1', 'Group 2', 'Group 3']

# # Create distplot with custom bin_size
# fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])

# # Plot!
# st.plotly_chart(fig, use_container_width=True)











