from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
import numpy as np
import pandas as pd

from helpers import apology

import joblib
import os

# ensure that Flask reads environment variables from my .env or .flaskenv
from dotenv import load_dotenv
load_dotenv()

# Configure application
app = Flask(__name__)

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Set the path to the `.joblib` file relative to the location of `app.py`
file_path_model_basic = os.path.join(os.path.dirname(__file__), 'body_fat', 'body_density_basic.joblib')
file_path_model_extra1 = os.path.join(os.path.dirname(__file__), 'body_fat', 'body_density_extra1.joblib')
file_path_model_extra2 = os.path.join(os.path.dirname(__file__), 'body_fat', 'body_density_extra2.joblib')
file_path_model_extra3 = os.path.join(os.path.dirname(__file__), 'body_fat', 'body_density_extra3.joblib')

file_path_trans_basic = os.path.join(os.path.dirname(__file__), 'body_fat', 'trans_basic.joblib')
file_path_trans_extra1 = os.path.join(os.path.dirname(__file__), 'body_fat', 'trans_extra1.joblib')
file_path_trans_extra2 = os.path.join(os.path.dirname(__file__), 'body_fat', 'trans_extra2.joblib')
file_path_trans_extra3 = os.path.join(os.path.dirname(__file__), 'body_fat', 'trans_extra3.joblib')

# Load pre-trained models
model_basic = joblib.load(file_path_model_basic)
model_extra1 = joblib.load(file_path_model_extra1)
model_extra2 = joblib.load(file_path_model_extra2)
model_extra3 = joblib.load(file_path_model_extra3)

# Load pre-trained power transformers
trans_basic = joblib.load(file_path_trans_basic)
trans_extra1 = joblib.load(file_path_trans_extra1)
trans_extra2 = joblib.load(file_path_trans_extra2)
trans_extra3 = joblib.load(file_path_trans_extra3)

@app.route("/", methods=["GET", "POST"])
def index():
    """Show form to fill out and information"""

    # List of required fields
    fields = ["age", "neck","knee", "ankle", "biceps","forearm", "wrist", "height", "weight", "abdomen","chest", "hip", "thigh"]
    
    if request.method == "POST":
        # Handle form submission

        # Clear any previously stored form data to ensure only current input is retained
        session.clear()

        # Store each form input in the session at the start of the POST request
        for field in fields:
            session[field] = request.form.get(field)

        # Check if units selected
        units = request.form.get("units") # get value of radio button with name "units"
        session["units"] = units # Store units in session
        if not units:
            return apology("Select units", 400)
        
        # Check if height, weight, and age are provided
        if not (session["height"] and session["weight"] and session["age"]):
            return apology("Height, Weight, and Age are obligatory", 400)
        
        # Access the input data from the form and add to a list of dictionaries
        data = {}

        for field in fields:
            value = session[field]

            # Proceed only if value is provided
            if value:
                try:
                    # Check if the value is a positive number
                    if float(value) > 0:
                        data[field] = float(value)  # Store as float for consistent numeric handling
                    else:
                        return apology("Provide a positive number", 400)
                except ValueError:
                    # If conversion fails, the input is not numeric
                    return apology("Provide a positive number", 400)
                
        # Convert units to be consisten with the trained models
        if units == "imperial":
            if "height" in data:
                # Convert height in inches to meters
                data["height"] = round((data["height"] * 0.0254), 2)
            if "weight" in data:
                # Convert weight in pounds to kilograms
                data["weight"] = round((data["weight"] * 0.454), 2)
            # Convert other length measurements in inches to cm
            for field in ["neck", "knee", "ankle", "biceps", "forearm", "wrist", "abdomen", "chest", "hip", "thigh"]:
                if field in data:
                    data[field] = round((data[field] * 2.54), 2)
        elif units == "metric" and "height" in data:
            # Convert height in cm to meters if it's in metric
            data["height"] = round((data["height"] / 100), 2)

        # calculate BMI
        height_sqr = data["height"] **2
        BMI = data["weight"]/height_sqr
        BMI = round(BMI, 1)

        # Store BMI to display in results page
        session["BMI"] = BMI

        # Calculate Abdomen to Chest ratio if input data provided
        if "abdomen" in data and "chest" in data:
            ACratio = data["abdomen"]/data["chest"]

        # Calculate Hip to Thigh ratio if input data provided
        if "hip" in data and "thigh" in data:
            HTratio = data["hip"]/data["thigh"]

        # Decide which model to use
        # If only height, weight and age provided -> show BMI and use basic model
        if "height" in data and "weight" in data and "age" in data and not("chest" in data and "abdomen" in data):

            # Create an input array with age and BMI as features
            input_basic = np.array([[data["age"], BMI]]) # 2D array format for one sample

            # Convert to DataFrame with the same feature names used in training
            input_basic_df = pd.DataFrame(input_basic, columns=['Age', 'BMI'])

            # Use the same power transformer as for the test data
            input_basic_df_transformed = pd.DataFrame(trans_basic.transform(input_basic_df), columns=input_basic_df.columns)

            # Get predictions from the basic model
            body_density_basic = model_basic.predict(input_basic_df_transformed)

            # Extract the first element from basic prediction
            body_density_basic_value = body_density_basic[0]  # Assuming it's a 1-element array
            
            # Calculate body fat percentage
            body_perc_basic = round((495 / body_density_basic_value -450), 1)

            # Store Body Fat Percentage to display on the result page
            session["BFP"] = body_perc_basic

            session["message1"] = "The data you provided was sufficient for a prediction using the Basic Model."
            session["message2"] = "The Basic Model is the least accurate, particularly when the predicted body fat percentage is very low or very high. To obtain a more reliable result, please provide your chest and abdomen circumferences."
            session["std"] = 5.5
            
            return render_template("result.html", BMI=BMI, BFP=body_perc_basic, message1=session["message1"], message2=session["message2"], std=session["std"])
        
        # If only height, weight, age, chest and abdomen are provided -> show BMI and predict extra1
        if "height" in data and "weight" in data and "age" in data and "chest" in data and "abdomen" in data and not("hip" in data and "thigh" in data):
            print("BMI: ", BMI)
            print("ACratio: ", ACratio)

            # Create an input array with age, BMI, and ACratio as features
            input_extra1 = np.array([[data["age"], BMI, ACratio]]) # 2D array format for one sample
            print("Input: ", input_extra1)

            # Convert to DataFrame with the same feature names used in training
            input_extra1_df = pd.DataFrame(input_extra1, columns=['Age', 'BMI', 'ACratio'])
            print("Input df: ", input_extra1_df)

            # Use the same power transformer as for the test data
            input_extra1_df_transformed = pd.DataFrame(trans_extra1.transform(input_extra1_df), columns=input_extra1_df.columns)
            print("Input df transformed: ", input_extra1_df_transformed)

            # Get predictions from the extra1 model
            body_density_extra1 = model_extra1.predict(input_extra1_df_transformed)
            print("Predicted body density (extra1): ", body_density_extra1)

            # Extract the first element from prediction_basic
            body_density_extra1_value = body_density_extra1[0]  # Assuming it's a 1-element array
            print(body_density_extra1_value)
            
            # Calculate body fat percentage
            body_perc_extra1 = round((495 / body_density_extra1_value -450), 1)
            print("Predicted (extra1) body %: ", body_perc_extra1)
            
            # Store Body Fat Percentage to display on the result page
            session["BFP"] = body_perc_extra1

            session["message1"] = "The data you provided was sufficient for a prediction using the Good Model."
            session["message2"] = "This model offers a significant improvement compared to the Basic Model. However, for an even more reliable result, please provide your hip and thigh circumferences."
            session["std"] = 4.9
            
            return render_template("result.html", BMI=BMI, BFP=body_perc_extra1, message1=session["message1"], message2=session["message2"], std=session["std"])
        
        # If only height, weight, age, chest, abdomen, hip, and thigh are provided -> show BMI and predict extra2
        if "height" in data and "weight" in data and "age" in data and "chest" in data and "abdomen" in data and "hip" in data and "thigh" in data and not("neck" in data and "knee" in data and "ankle" in data and "biceps" in data and "forearm" in data and "wrist" in data):
            print("BMI: ", BMI)
            print("ACratio: ", ACratio)
            print("HTratio: ", HTratio)

            # Create an input array with age, BMI, ACratio, and HTratio as features
            input_extra2 = np.array([[data["age"], BMI, ACratio, HTratio]]) # 2D array format for one sample
            print("Input: ", input_extra2)

            # Convert to DataFrame with the same feature names used in training
            input_extra2_df = pd.DataFrame(input_extra2, columns=['Age', 'BMI', 'ACratio', 'HTratio'])
            print("Input df: ", input_extra2_df)

            # Use the same power transformer as for the test data
            input_extra2_df_transformed = pd.DataFrame(trans_extra2.transform(input_extra2_df), columns=input_extra2_df.columns)
            print("Input df transformed: ", input_extra2_df_transformed)

            # Get predictions from the extra2 model
            body_density_extra2 = model_extra2.predict(input_extra2_df_transformed)
            print("Predicted body density (extra2): ", body_density_extra2)

            # Extract the first element from prediction_basic
            body_density_extra2_value = body_density_extra2[0]  # Assuming it's a 1-element array
            print(body_density_extra2_value)
            
            # Calculate body fat percentage
            body_perc_extra2 = round((495 / body_density_extra2_value -450), 1)
            print("Predicted (extra2) body %: ", body_perc_extra2)

            # Store Body Fat Percentage to display on the result page
            session["BFP"] = body_perc_extra2

            session["message1"] = "The data you provided was sufficient for a prediction using the Better Model."
            session["message2"] = "This model provides a fairly reliable result. However, for the most accurate and trustworthy outcome—especially if your BMI is very low or very high—please provide the missing data required for the Best Model."
            session["std"] = 4.9
            
            return render_template("result.html", BMI=BMI, BFP=body_perc_extra2, message1=session["message1"], message2=session["message2"], std=session["std"])
        
        # If all data are provided -> show BMI and predict extra3
        all_data_provided = all(data[key] for key in data)

        if all_data_provided:
            print("BMI: ", BMI)
            print("ACratio: ", ACratio)
            print("HTratio: ", HTratio)

            # Create an input array with age, neck, knee, ankle, biceps, forearm, wrist, BMI, ACratio, and HTratio as features
            input_extra3 = np.array([[data["age"], data["neck"], data["knee"], data["ankle"], data["biceps"], data["forearm"], data["wrist"], BMI, ACratio, HTratio]]) # 2D array format for one sample
            print("Input: ", input_extra3)

            # Convert to DataFrame with the same feature names used in training
            input_extra3_df = pd.DataFrame(input_extra3, columns=['Age', 'Neck', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist', 'BMI', 'ACratio', 'HTratio'])
            print("Input df: ", input_extra3_df)

            # Use the same power transformer as for the test data
            input_extra3_df_transformed = pd.DataFrame(trans_extra3.transform(input_extra3_df), columns=input_extra3_df.columns)
            print("Input df transformed: ", input_extra3_df_transformed)

            # Get predictions from the extra3 model
            body_density_extra3 = model_extra3.predict(input_extra3_df_transformed)
            print("Predicted body density (extra3): ", body_density_extra3)

            # Extract the first element from prediction_basic
            body_density_extra3_value = body_density_extra3[0]  # Assuming it's a 1-element array
            print(body_density_extra3_value)
            
            # Calculate body fat percentage
            body_perc_extra3 = round((495 / body_density_extra3_value -450), 1)
            print("Predicted (extra3) body %: ", body_perc_extra3)

            # Store Body Fat Percentage to display on the result page
            session["BFP"] = body_perc_extra3

            session["message1"] = "The data you provided was sufficient for a prediction using the Best Model."
            session["message2"] = ""
            session["std"] = 4.9
            
            return render_template("result.html", BMI=BMI, BFP=body_perc_extra3, message1=session["message1"], message2=session["message2"], std=session["std"])

        # Clear session data after successful form submission
        session.clear()

        return render_template("layout.html")
    
    # If accessed via GET request, load stored data into the form
    form_data = {field: session.get(field, '') for field in fields}
    form_data["units"] = session.get("units", '')
    return render_template("index.html", form_data=form_data)


@app.route("/result")
def result():
    """Show the results and some explanation"""

    if session['BMI'] and session['BFP']:
        return render_template("result.html", BMI=session['BMI'], BFP=session['BFP'], message1=session['message1'], message2=session['message2'], std=session["std"])
    
    return apology("Fill out the form", 400)

@app.route("/info")
def info():
    """Display html page with more information"""
    
    return render_template("info.html")