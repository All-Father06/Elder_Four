# Elder_Four 
# TIAA Financial Advisory System 
Welcome to the TIAA Financial Advisory System! This Flask application provides a comprehensive financial advisory experience, leveraging machine learning models to assess risk tolerance, predict stock market trends, recommend mutual funds, and provide personalized guidance based on user emotions and lifestyle factors. 

## Getting Started 

### 1. Clone the Repository 
<pre>
git clone https://github.com/your-username/TIAA-Financial-Advisory.git
cd TIAA-Financial-Advisory
</pre>
### 2. Install Dependencies
<pre>
pip install -r requirements.txt
</pre>

### 3. Database Setup 
Ensure you have MySQL installed and running. 
Create a MySQL database named tiaa with the required tables. 
Update the db_config dictionary in the code with your MySQL credentials. 

### 4. Run the Application 
<pre>
python app.py
</pre>
Open your web browser and navigate to http://localhost:5000/ to access the application. 
Features User Authentication Users can securely log in with their credentials or register for a new account. 

Home Page The home page provides a snapshot of the user's latest transactions, offering a quick overview of their financial activity. 

User Profile Get a detailed view of the user's profile, including age, savings, transaction history, and a personalized risk tolerance assessment. 

1. Financial Insights Risk Tolerance Model 
   - **Objective**: Assess the user's risk tolerance level based on their financial profile.
     - **Model Type**: Random Forest Classifier.
       - **Implementation**: Predicts whether a user has low, moderate, or high risk tolerance, considering features such as age, savings, and transaction history.

2. Stock Market Prediction Model 
   - **Objective**: Predict future stock prices to provide insights into stock market trends.
     - **Model Type**: Long Short-Term Memory (LSTM) Neural Network.
       - **Implementation**: Utilizes historical stock data to train the LSTM model, which predicts future stock prices.

3. Mutual Fund Recommendation Model 
   - **Objective**: Recommend mutual funds based on the user's financial profile and risk tolerance.
     - **Model Type**: Random Forest Classifier.
       - **Implementation**: Predicts suitable mutual fund categories based on features like expense ratio, fund size, and historical performance.

4. Financial Guidance Model 
   - **Objective**: Provide personalized financial guidance based on the user's financial profile, including age, income, savings, debt, and risk tolerance.
     - **Model Type**: Random Forest Classifier.
       - **Implementation**: Utilizes a Random Forest Classifier to analyze the user's features and generate tailored financial guidance. Recommends actions such as diversifying the portfolio, assessing risk levels, and staying informed about market trends.

5. Emotional State Prediction and Guidance Model 
   - **Objective**: Predict the user's emotional state based on lifestyle factors.
     - **Model Type**: Random Forest Classifier.
       - **Implementation**: Analyzes user-provided inputs regarding gender, marital status, living arrangements, physical health status, exercise routine, dietary habits, and social interaction frequency. Predicts the emotional state as positive, negative, or neutral.

6. Emotional Guidance and Support 
   - **Positive Emotions**: Recommends enjoyable content such as comedy videos to enhance a positive mood.
   - **Negative Emotions**: Suggests seeking professional support, providing information on nearby therapists or counselors.
   - **Neutral Emotions**: Offers a balanced approach, providing both comedic relief and information on professional support.

These models collectively empower users to make informed decisions about their investments, considering their individual financial circumstances, risk preferences, and emotional well-being. The application aims to provide a holistic financial advisory experience tailored to each user's unique needs and goals. 

### 5. Dependencies 

- **Flask**
- **MySQL Connector**
- **pandas**
- **pickle**
- **scikit-learn**
- **TensorFlow**
- **joblib**
- **googlemaps** (for potential location-based features)

### Contributors 
- **Shabbir Talib**
- **Prachi Pathak**
- **Vaishnavi Chandgadkar**
- **Atharva Hirve**

### Acknowledgments 
Special thanks to TIAA Financial Services for inspiration and support. 

### License 

This project is licensed under the MIT License - see the LICENSE file for details.
