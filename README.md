# Optimization of NPK Analysis using Deep Learning

This project utilizes Streamlit to optimize NPK (Nitrogen, Phosphorus, Potassium) analysis using deep learning models. It provides an interactive interface for users to input NPK values along with environmental factors like temperature, humidity, and pH. Based on these inputs, the application predicts the most suitable crop using a combination of Random Forest and Simple RNN models.

## Features

- **Home Page**
  - Displays fundamental information about nutrients (Nitrogen, Phosphorus, Potassium), temperature, humidity, and pH for crop optimization.

- **NPK Analysis**
  - Allows users to input NPK values and environmental factors.
  - Provides a prediction on the most suitable crop based on the input data.
  - Visualizes NPK analysis results through bar charts.

- **Contact Us**
  - Lists project details including team members, project guide, and college address.

## Technologies Used

- **Streamlit**: Python framework for building interactive web applications.
- **Pandas, NumPy**: Data manipulation and numerical computing libraries.
- **TensorFlow, Scikit-learn**: Machine learning and deep learning frameworks.
- **Joblib**: For loading the Random Forest model.
- **LabelEncoder**: For encoding categorical labels.
- **HTML/CSS**: Custom styling for UI elements.

## Installation

To run this project locally, ensure you have Python installed along with the necessary libraries. Clone the repository and install dependencies using pip:

```bash
git clone <https://github.com/SivaShankar-Juthuka/Optimization-of-NPK-Anlaysis-Using-Deep-Learning>
cd <repository-directory>
pip install -r requirements.txt
```

## Usage

Run the Streamlit application using the following command:

```bash
streamlit run app.py
```
After running the command you are able to see the `localhost:port_number` in your terminal either you can click on that or else you can open your web browser and navigate to `http://localhost:port_number` to interact with the application.

## Contributors

- **J. Siva Shankar** - 20P31A0590@acet.ac.in
- **B. Divya Bala Tripura Sundari** - 20P31A0572@acet.ac.in
- **M. V. Sri Padma** - 20P31A05A2@acet.ac.in
- **N. Siya Sudiksha** - 20P31A05A6@acet.ac.in

## Project Guide

- **Dr. R. V. S. Lalitha, M.Tech Ph.D.** - Project Guide
