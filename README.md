# Body Fat Predictor and BMI Calculator (for men)
#### Video Demo: https://www.youtube.com/watch?v=zYxdDmawa38
#### Description:
##### https://bodyfatpredictor.eu.pythonanywhere.com/
**Body Fat Predictor** is a web application that predicts body fat percentage in men using machine learning models and calculates their BMI. The accuracy of the prediction depends on the level of information the user provides in the form on the main page. Each additional data point improves the accuracy of the prediction:
1. Height, Weight and Age: This information calculates the BMI and initiates a machine learning prediction with lower accuracy.
2. Chest and Abdomen Circumferences: These are the critical data points that significantly improve prediction accuracy.
3. Hip and Thigh Circumferences: Including these measurements further enhances the prediction.
4. Neck, Knee, Ankle, Biceps, Forearm, and Wrist Circumferences: Providing these additional measurements yields the most accurate prediction.

After submitting the form the user is redirected to a **Result** page where his results are displayed along with some basic information about its interpretation.

The **Info** page briefly describes the origin of the data, how the models were built, and the metrics of their performance.

The app was developed using VisualStudio Code. The project is built in Python using the Flask framework. It also uses HTML, CSS, and Bootstrap for the website creation and Python Jupyter Notebook  for data analysis and the training of machine learning models.

##### Technology and Files
###### Python Application/Flask
**Flask** - I have decided to use Flask as the web framework because it was used in Week9 of the CS50 course and my application is similar in design and functionality.

**app.py** - The application file that:
- imports necessary modules and helper functions;
- uses the `dotenv` library to load local environment variable files into the Flask application's environment;
- configures the Flask application and the session to store form data temporarily;
- with the `@app.after_request` decorator ensures that the responses are not cached, which prevents the browsers from displaying outdated content;
- loads pretrained models and their specific Power Transformers;
- defines the `@app.route("/")` that displays the form with `"GET"` method and handles the submission with `"POST"` method:
    - stores the form input data in a session;
    - checks for obligatory and valid input;
    - converts units;
    - calculates BMI;
    - checks for the data provided and decides which ML model to use;
    - redirects to **Result** page with the information stored in session: BMI, body fat percentage, model used, and some messages;
- defines the `@app.route("/result")` that displays the information stored in session or asks users to fill out the form if there is no data in session;
- defines the `@app.route("/info")` that displays basic information about the machine learning models.

**helpers.py** - with the implementation of **Apology**.

**templates** - HTML code for differnet pages, based on pages used in Week9 of C$50 Finance.

**static** - with styles.css, the favicon file, and the HTML I_heart_validator file.

**models'files** - body_density_basic.joblib, body_density_extra1.joblib, body_density_extra2.joblib and body_density_extra3.joblib. See more details below.

**transformers' files** - trans_basic.joblib, trans_extra1.joblib, trans_extra2.joblib and trans_extra3.joblib. See more details below.

##### Machine learning/Jupyter notebook

This part is based on the CS50 Machine Learning seminar about supervised machine learning models and the work on the data set done by Harshit Gupta on Kaggle. This was the first time I worked on machine learning exercises and I do realize the results are not optimal.

###### The data
The data is available on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset). It presents various body measurements, age, body density determined from underwater weighing, and percentage of body fat calculated from body density of 252 men. I chose this data set becasue it is a small set of high-quality clean data.

**bodyfat.csv** - Contains the data for the analysis.

**body_fat.ipynb** - This is the Jupyter Notebook where the data analysis and machine learning were performed:
- imports necessary libraries and the dataset;
- converst units;
- checks the quality of the data and removes outliers;
- performs data exploration to check for bias, noise and relationships between different features and the target;
- following Harshit Gupta, I introduced BMI, Abdomen-to-Chest ratio, and Hip-to-Thigh ratio columns to improve the model's accuracy;
- I decided to train models with body density as the target, because it was measured directly alongside other data points. Later I tried to predict body fat percentage directly, following the same procedure outlined here, as can be seen in **body_fat_direct.ipynb**. However the final machine learning metrics did not improve, and I decided to use the models trained here in the final web-based application;
- There are four models trained in the notebook: basic, extra1, extra2 and extra3. Each of them uses a different amount of data provided by the user. The first model I trained (now called extra3) takes all the data as present in the dataset. However, I thought that maybe not everybody would have enough patience to measure so many body points and decided to train models with less data points, even if it would mean lower accuracy. Each model follows the same steps:
    1. Drop data - starting from the model that takes all data points (model extra3), with every next model we drop the columns that are not going to be used;
    2. Train-test split - I used a 75/25 split, as the default in scikit-learn library, and set the random state to 42 to ensure the reproducibility of the split;
    3. Power Transform - to correct the normality of the data distribution. It is especially important when applying linear machine learning algorithms. The transformer is fit on the training data and later the same transformation is applied to the testing and user input data.
    4. Save Power Transformer - each model has its own transformer and here it is saved as a **.joblib** file, which is later used in **app.py** application.
    5. Test Machine Learning Models - following Harshit Gupta, I decided to use a set of models evaluated with the same metrics. The best model is further assessed for potential bias, which was observed in each model to varying extents. This is something to explore in more detail as I learn more about machine learning.
    6. Save the model - each model is saved as a **.joblib** file, which is later used in **app.py** application.

  After training the models it was clear that providing the minimal extra information compared to the Basic Model (that takes only height, weight and age, and is therefore highly correlated with BMI) significantly improves the model's performance. Providing more information than needed for the Good Model only slightly improved the performance; however, this additional information was especially important for the extreme values, where the models showed the highest bias;
- manually tests a few values - this is to test a few values from the dataset and see how they compare to the real values in the dataset, as well as to later compare the same values in the **app.py** application to confirm that the models were loaded and are performing correctly.

##### Hosting
The app is hosted at https://www.eu.pythonanywhere.com/. The files are uploaded via Git using GitHub and the machine learning Jupyter Notebooks are ignored using **.gitignore**, as they are not needed for the functionality of the website.