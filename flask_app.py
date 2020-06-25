#!/usr/bin/python3.7
from flask import Flask, jsonify, request, url_for, redirect, render_template
import json
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

path_files = '/home/DataScientist/mysite/static/Files/'


# Clients data (test)
clients = pd.read_csv(path_files+'tab_clients.csv')

# Valid clients data
valid_clients = pd.read_csv(path_files+'clients_data.csv')

# Get data of test clients
@app.route('/clients')
def get_clients():
    clients_json = clients.to_json()
    return jsonify(clients_json)


# Get data of chosen client (user picks an ID in the dashboard)
@app.route('/api')
def get_client():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No valid id provided. Please specify a valid id."

    client_chosen = clients.loc[clients['SK_ID_CURR']==id]
    client_chosen = client_chosen.to_json()
    return jsonify(client_chosen)

# Get data of cluster where chosen client was predicted to be
@app.route('/cluster')
def get_cluster():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No valid id provided. Please specify a valid id."

    client_chosen = clients.loc[clients['SK_ID_CURR']==id]
    proba = client_chosen['Proba'].values
    label = client_chosen['Label'].values[0]

    # Convert proba to target prediction
    if proba >= 0.5:
        target = 1
    else:
        target = 0

    # Get cluster within valid data
    cluster = valid_clients.loc[(valid_clients['Label']==label) & (valid_clients['TARGET']==target)]
    cluster_json = cluster.to_json()
    return jsonify(cluster_json)

# @app.route('/display/<filename>')
# def display_image(filename):
# 	#print('display_image filename: ' + filename)
# 	return redirect(url_for('static', filename=filename), code=301)


# @app.route('/image')
# def uploaded_file():
#     path = 'static/image_app.jpg'
#     return render_template('temp.html')



