from flask import Flask, render_template, redirect, url_for, flash, request , session
import mysql.connector
import pandas as pd
import pickle
import sklearn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import googlemaps
app = Flask(__name__)
with open('models/risk_tolerance_model.pkl', 'rb') as file:
    rt_model = pickle.load(file)
with open('scalers/risk_tolerance_encoder.pkl', 'rb') as file:
    encoder_risk_tolerance = pickle.load(file)
with open('scalers/risk_mft.pkl', 'rb') as file:
    encoder_mft = pickle.load(file)
with open('scalers/stock_scaler.pkl', 'rb') as file:
    stock_scaler = pickle.load(file)
stock_model = load_model('models/lstm_model.h5') 
with open('scalers/encoder_advice.pkl', 'rb') as file:
    encoder_adv = pickle.load(file)
with open('models/rf_model_advice.pkl', 'rb') as file:
    adv_model = pickle.load(file)
with open('models/rf_emotions.pkl', 'rb') as file:
    emo_model = pickle.load(file)
with open('scalers/Emotion_encoder.pkl', 'rb') as file:
    scalar1 = pickle.load(file)
with open('scalers/scaler_emotions.pkl', 'rb') as file:
    scalar2 = pickle.load(file)
# Load the LSTM model
mut_model = joblib.load('models/rf_model.joblib')

# Load the scaler
mut_scaler = joblib.load('scalers/fund_scaler.pkl')

# MySQL configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'tiaa',
}
app.secret_key = 'Tiaa'
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/login',methods=['POST'])
def login():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    username_variable = request.form.get('uname')
    password_variable = request.form.get('pwd')
    query = "SELECT * FROM user WHERE Name LIKE %s AND Pass LIKE %s"
    cursor.execute(query, (f'%{username_variable}%', f'%{password_variable}%'))
    user_data = cursor.fetchone()
    # Close the cursor and connection
    cursor.close()
    connection.close()
    if user_data:
        # Log in the user and store user ID in session
        session['user_id'] = user_data[0]  # Assuming the user ID is the first column in the result
        session['username'] = user_data[1]
        session['mail'] = user_data[2]
        session['phone'] = user_data[3]  # Assuming the username is the second column in the result
        flash('Login successful', 'success')
        return redirect(url_for('home'))
    else:
        flash('Login failed. Check your username and password.', 'danger')
        return redirect(url_for('index'))
@app.route('/sign', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        username = request.form.get('uname')
        phone  = request.form.get('phone')
        mail =  request.form.get('mail')
        password = request.form.get('pwd')

        # Example query to insert a new user into the database
        query = "INSERT INTO user (Name, Pass,Email,Phone) VALUES (%s, %s,%s,%s)"
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        try:
            cursor.execute(query, (username, password,mail,phone))
            connection.commit()
            flash('Registration successful. You can now log in.', 'success')
            return redirect(url_for('index'))
        except mysql.connector.Error as err:
            flash(f'Registration failed. Error: {err}', 'danger')
            connection.rollback()
        finally:
            cursor.close()
            connection.close()

    return render_template('index.html')
@app.route('/home')
def home():
    # Retrieve user ID from session
    user_id = session.get('user_id')
    username = session.get('username')
    query = "SELECT * FROM transactions WHERE User_Id like %s ORDER BY Date DESC LIMIT 1;"
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute(query, (user_id,))
    data = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    connection.close()
    if user_id:
        return render_template('home.html',user=username,data=data)
    else:
        return render_template('index.html')
@app.route('/user')
def user():
    connection = mysql.connector.connect(**db_config)
    user_id = session.get('user_id')
    user_name = session.get('username')
    user_mail = session.get('mail')
    user_phone = session.get('phone')
# SQL query to fetch records
    query = "SELECT * FROM User_Data where User_ID like %s"
    cursor = connection.cursor()
    cursor.execute(query,(user_id,))

# Fetch the records
    records1 = cursor.fetchall()

# Get column names from the cursor description
    columns = [column[0] for column in cursor.description]

# Create a DataFrame from the records and columns
    user_data = pd.DataFrame(records1, columns=columns)
    query = "SELECT * FROM transactions where User_Id like %s"
    cursor = connection.cursor()
    cursor.execute(query,(user_id,))

# Fetch the records
    records2 = cursor.fetchall()

# Get column names from the cursor description
    columns = [column[0] for column in cursor.description]

# Create a DataFrame from the records and columns
    transaction_data = pd.DataFrame(records2, columns=columns)
    transaction_data.rename(columns={'User_Id': 'User_ID'}, inplace=True)
    transaction_data
    # Combine user data with risk tolerance
    user_transaction_data = pd.merge(user_data, transaction_data, on='User_ID', how='inner')

# Display the combined data
    # user_transaction_data['Amount'] = user_transaction_data['Amount'] * 1000
    transaction_features = user_transaction_data.groupby('User_ID').agg({
    'Transaction_Type': lambda x: x.value_counts().index[0],  # Most frequent transaction type
    'Amount': 'sum',  # Total transaction amount
    'Date': 'count'  # Number of transactions
    }).reset_index()
    transaction_features.rename(columns={'Date': 'No. of Transactions'}, inplace=True)
    transaction_features.rename(columns={'Amount': 'Total Amount of Transactions'}, inplace=True)
    transaction_features.rename(columns={'Transaction_Type': 'Most frequent transaction type'}, inplace=True)
    user_features = pd.merge(user_data, transaction_features, on='User_ID', how='inner')
    user_features['Risk_Tolerance'] = encoder_risk_tolerance.transform(user_features['Risk_Tolerance'])
    user_features['Most frequent transaction type'] = encoder_mft.transform(user_features['Most frequent transaction type'])
    X = user_features[['Age','Savings','Most frequent transaction type','Total Amount of Transactions','No. of Transactions']]
    y = rt_model.predict(X)
    if y == 1:
        risk = 'low'
    elif y == 2:
        risk = 'moderate'
    elif y ==0:
        risk = 'high'
    age = user_data['Age'].iloc[0]
    inc = user_data['Income'].iloc[0]
    save = user_data['Savings'].iloc[0]
    mft = user_features['Most frequent transaction type'].iloc[0]
    if mft == 0:
        mft = 'Crediting'
    else:
        mft = 'Debited'
    tta = user_features['Total Amount of Transactions'].iloc[0]
    nt = user_features['No. of Transactions'].iloc[0]
    session['risk'] = risk
    print("Age:",age,"Savings:",save,"Mft:",mft,"tta:",tta,"nt",nt)
    return render_template('user.html' , age = age,save =save,mft=mft,tta=tta,nt=nt ,data = records2 , rt = risk,n = user_name,m=user_mail,p =user_phone,inc =inc)
@app.route('/finance')
def finance():
    features = ['Open', 'High', 'Low', 'Close']
    original_df = pd.read_csv('dataset/stock.csv')
    rt = session.get('risk')
    original_df = original_df[original_df['RiskCategory'] == rt].head(25)
    print(original_df.shape)
    last_pred = []
    for index, row in original_df.iterrows():
        csv_path = row.iloc[3]  # Assuming the path is in the last column
        csv_data = pd.read_csv(csv_path)
        # Extract the last row and specific columns
        last_values = csv_data.loc[csv_data.index[-1], features] 
        original_df.loc[index, features] = last_values.values
        data1 = csv_data[features].values
        # print(data1)
        data1 = stock_scaler.transform(data1)
        # print(data1)
        X = []
        for i in range(len(data1) - 1):
            X.append(data1[i])
        X= np.array(X)
        # print('Before',X.shape)
        # Reshape the input data for LSTM (samples, time steps, features)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        # print(X.shape)
        # Example: Apply prediction logic to the data read from CSV
        prediction = stock_model.predict(X)
        prediction = stock_scaler.inverse_transform(prediction)  # Replace with your actual prediction logic
        # Add the prediction to the original DataFrame
        # print(prediction[-1])
        fprediction = prediction[-1]
        last_pred.append(fprediction.tolist())
        # print(j)
    pred_columns = ['Predicted Open', 'Predicted High', 'Predicted Low', 'Predicted Close']
    # Create a DataFrame from the list of predictions and set column names
    pred_df = pd.DataFrame(last_pred, columns=pred_columns)
    # Concatenate the original DataFrame and the predictions DataFrame
    result_df = pd.concat([original_df, pred_df], axis=1)
    user = session.get('username')
    df =  pd.read_csv('dataset/mutual.csv')
    df.replace('-', float('NaN'), inplace=True)
# Convert columns to numeric type
    numeric_columns = ['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino', 'alpha', 'sd', 'beta', 'sharpe', 'returns_1yr', 'returns_3yr', 'returns_5yr']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    # Drop rows with missing values
    df.dropna(inplace=True)
# Normalize the numerical columns
# Select relevant features and target variable
    features = ['expense_ratio', 'fund_size_cr', 'fund_age_yr', 'sortino', 'alpha', 'sd', 'beta', 'sharpe', 'returns_1yr', 'returns_3yr', 'returns_5yr']
    X = df[features]
    X = mut_scaler.transform(X)
    predictions = mut_model.predict(X)
    print(predictions)
    df['RF_Predictions'] = predictions.tolist()
    if rt == 'Low':
        rr = 1
    elif rt == 'Moderate':
        rr =2 
    else:
        rr = 3
    fdf = df[df['RF_Predictions'] == rr].drop(columns=['fund_manager','sortino','alpha','sd','beta','sharpe','risk_level','amc_name','rating','RF_Predictions'])
    # print(fdf)
    riskt = session.get('risk')
    connection = mysql.connector.connect(**db_config)
    user_id = session.get('user_id')
# SQL query to fetch records
    query = "SELECT * FROM User_Data where User_ID like %s"
    cursor = connection.cursor()
    cursor.execute(query,(user_id,))

# Fetch the records
    records1 = cursor.fetchall()

# Get column names from the cursor description
    columns = [column[0] for column in cursor.description]

# Create a DataFrame from the records and columns
    user_data = pd.DataFrame(records1, columns=columns)
    query = "SELECT * FROM transactions where User_Id like %s"
    cursor = connection.cursor()
    cursor.execute(query,(user_id,))

# Fetch the records
    records2 = cursor.fetchall()

# Get column names from the cursor description
    columns = [column[0] for column in cursor.description]

# Create a DataFrame from the records and columns
    transaction_data = pd.DataFrame(records2, columns=columns)
    transaction_data.rename(columns={'User_Id': 'User_ID'}, inplace=True)
    transaction_data
    # Combine user data with risk tolerance
    user_transaction_data = pd.merge(user_data, transaction_data, on='User_ID', how='inner')

# Display the combined data
    # user_transaction_data['Amount'] = user_transaction_data['Amount'] * 1000
    transaction_features = user_transaction_data.groupby('User_ID').agg({
    'Transaction_Type': lambda x: x.value_counts().index[0],  # Most frequent transaction type
    'Amount': 'sum',  # Total transaction amount
    'Date': 'count'  # Number of transactions
    }).reset_index()
    transaction_features.rename(columns={'Date': 'No. of Transactions'}, inplace=True)
    transaction_features.rename(columns={'Amount': 'Total Amount of Transactions'}, inplace=True)
    transaction_features.rename(columns={'Transaction_Type': 'Most frequent transaction type'}, inplace=True)
    user_features = pd.merge(user_data, transaction_features, on='User_ID', how='inner')
    user_features['Risk_Tolerance'] = encoder_risk_tolerance.transform(user_features['Risk_Tolerance'])
    user_features['Most frequent transaction type'] = encoder_mft.transform(user_features['Most frequent transaction type'])
    X_advice = user_features[['Age', 'Income', 'Savings', 'Debt', 'Risk_Tolerance']]
    y_advice = adv_model.predict(X_advice)
    y_advice = encoder_adv.inverse_transform(y_advice)
    print(y_advice)
    ad = y_advice[0]
    if  (ad == 1):
        advice = "According to your transaction history our advice is for you to diversify your portfolio."
    elif (ad==2):
        advice = "According to your transaction history our advice is for you to keep up with the stock market trends and invest when you have enough savings and find a good investment."
    else:
        advice = "According to your transaction history our advice is for you to make good risk assesments while investing."
    return render_template('finance.html',ad = advice,user = user,r = riskt,data2=fdf,data1=result_df.drop(columns=['NASDAQ Symbol','FilePath','RiskCategory','Symbol']).head(25).dropna())
@app.route('/guide' , methods = ['POST'])
def generate_guidance():
    user_input = request.form.to_dict()
    print(user_input)
    df = create_dataframe(user_input)
    EDA_columns_to_encode = ['GENDER',  'MARITALSTATUS','LIVINGARRANGEMENTS','PHYSICALHEALTHSTATUS','EXERCISEROUTINE','DIETARYHABITS','SOCIALINTERACTIONFREQUENCY','EMOTIONLABEL']
    encoded_categorical_columns = scalar1.transform(df[EDA_columns_to_encode])
    # Use the DataFrame 'df' as needed (e.g., store it, analyze it, etc.)
    encoded_categorical_df = pd.DataFrame(encoded_categorical_columns, columns=EDA_columns_to_encode)
    numerical_columns = df.select_dtypes(include=['int64'])
# Step 3: Combine numerical and encoded categorical data
    new_dataset = pd.concat([numerical_columns, encoded_categorical_df], axis=1)
    # context = "The user mentioned: " + user_input['user_input']
    selected_DataFrame = pd.DataFrame(new_dataset, columns=['GENDER',  'MARITALSTATUS','LIVINGARRANGEMENTS','PHYSICALHEALTHSTATUS','EXERCISEROUTINE','DIETARYHABITS','SOCIALINTERACTIONFREQUENCY'])
    columns_to_scale = ['GENDER',  'MARITALSTATUS','LIVINGARRANGEMENTS','PHYSICALHEALTHSTATUS','EXERCISEROUTINE','DIETARYHABITS','SOCIALINTERACTIONFREQUENCY']
    selected_DataFrame[columns_to_scale] = scalar2.transform(selected_DataFrame[columns_to_scale])
    X = selected_DataFrame
    emotion = emo_model.predict(X)
    guidance = generate_guidance(emotion)
    print(emotion)
    comedy_videos = get_comedy_videos()
    if emotion == 1:
        nearby_consultants = get_consultant_list()
        print(nearby_consultants)
        return render_template('well.html', guidance=guidance, comedy_videos=comedy_videos, consultants=nearby_consultants)
    elif emotion== 2:
        nearby_consultants = get_consultant_list()
        return render_template('well.html', guidance=guidance, comedy_videos=comedy_videos,consultants=nearby_consultants)
    else:
        return render_template('well.html', guidance=guidance, comedy_videos=comedy_videos)
def get_consultant_list():
        # Define a list of consultants with their information
        consultant_list = [
            {"name": "Dr. Jane Doe", "specialty": "Clinical Psychologist", "rate": "$150/hour", "location": "321 Oak St, City"},
            {"name": "Dr. John Smith", "specialty": "Marriage and Family Therapist", "rate": "$120/hour", "location": "456 Main St, Town"},
            {"name": "Dr. Emily Johnson", "specialty": "Life Coach", "rate": "$100/hour", "location": "789 Pine St, Village"},
            {"name": "Dr. Michael Brown", "specialty": "Career Counselor", "rate": "$130/hour", "location": "555 Elm St, Suburb"},
            # Add more consultants as needed
        ]

        return consultant_list
# def search_nearby_consultants():
#         try:
#             # Specify the type of place you're looking for, e.g., 'psychologist'
#             place_type = 'psychologist'
#             gmaps = googlemaps.Client(key='AIzaSyDZDNstAoeoNej3xtlCKZwT-Jl5IlK3wBE')
#             # Specify the radius (in meters) to search for places
#             radius = 5000  # Adjust this value based on your preference

#             # Get user's location (you can modify this based on your app's requirements)
#             user_location = get_user_location()

#             if user_location:
#                 # Perform a nearby search using the Google Places API
#                 places_result = gmaps.places_nearby(
#                     location=user_location,
#                     radius=radius,
#                     type=place_type
#                 )

#                 # Extract and format relevant information about nearby places
#                 nearby_consultants = [
#                     f"{place['name']} - {place['vicinity']}"
#                     for place in places_result.get('results', [])
#                 ]
#                 print(nearby_consultants)
#                 return nearby_consultants
#             else:
#                 return ["Unable to determine user's location."]

#         except Exception as e:
#             print(f"Error searching for nearby consultants: {e}")
#             return ["Error searching for nearby consultants."]

# def get_user_location():
#         # Get the user's location based on your app's logic
#         # For simplicity, returning a hardcoded location for the example
#         return {'lat': 37.7749, 'lng': -122.4194}
def generate_guidance(emotion):
        # Generate guidance based on the classified emotion and optional context
        if emotion == 0:
            guidance = "It seems like you're feeling positive! Enjoy some comedy videos:"
        elif emotion == 2:
            guidance = "I sense some negativity. Consider talking to a professional for support. Here are nearby consultants:"
            
        else:
            guidance = "Your emotions seem neutral. Enjoy some comedy videos and consider talking to nearby consultants:"
        
        return guidance
def get_comedy_videos():
        # Get comedy video links from the local folder
        video_files = os.listdir("static\comedy_videos")
        video_links = [f'/static/comedy_videos/{video}' for video in video_files]

        return video_links
def create_dataframe(user_input):
    # Create a DataFrame from the user input
    df = pd.DataFrame([user_input])
    df["EMOTIONLABEL"] = 'Sad'
    print(df)
    return df
@app.route("/well")
def emot():
    return render_template('well.html')
@app.route("/logout")
def logout():
    session.clear()
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
