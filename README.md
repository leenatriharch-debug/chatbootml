# Customer Support Chatbot

## Overview

This is a Machine Learning based Customer Support Chatbot built using Python and Streamlit. The chatbot answers customer queries based on a predefined dataset of questions and answers.

## Features

* Interactive chat interface using Streamlit
* Machine Learning based response prediction
* TF-IDF Vectorization for text processing
* Logistic Regression for classification
* Fallback response for unknown questions

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn

## Project Structure

CustomerSupportBot/
│
├── app.py
├── customersupport.csv
├── requirements.txt
└── README.md

## Installation

1. Clone the repository:
   git clone <repository-url>

2. Move into the project folder:
   cd CustomerSupportBot

3. Install dependencies:
   pip install -r requirements.txt

4. Run the application:
   streamlit run app.py

## Dataset

The chatbot uses a CSV dataset containing:

* question
* answer

## How It Works

1. Loads customer support dataset.
2. Converts text into numerical vectors using TF-IDF.
3. Trains a Logistic Regression model.
4. Predicts the most relevant answer for user queries.
5. Shows a fallback message if confidence is low.

## Future Improvements

* OpenAI API integration
* Voice support
* Multi-language support
* Database connectivity
* Sentiment Analysis

## Author

Leena
BSc IT Final Year Project
