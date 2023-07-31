from flask import Flask, render_template, request,jsonify,session
import pandas as pd
from datacollection import scraping
from datetime import datetime, timedelta
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re
import joblib
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pycaret.regression import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pycaret.regression import load_model, predict_model,save_model

app = Flask(__name__)
app.secret_key = 'i_made_this'
def custom_tokenizer(text):
       tokens = word_tokenize(text)
       pos_tags = pos_tag(tokens)  # Perform POS tagging
       filtered_tokens = [token for token, pos in pos_tags if pos not in ['RB', 'JJ', 'VB']]  # Filter out adverbs, adjectives, and verbs
       return filtered_tokens
nltk.download('stopwords')
stopwords_list = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer,ngram_range=(1, 2),stop_words=stopwords_list)
# Standardize the features
scaler = StandardScaler()
pca = PCA(n_components=2)  # Set the number of components you want to retain

current_date = datetime.now().strftime("%Y-%m-%d")
data_folder_path = os.path.join(os.getcwd(), "data")
combined_folder_path = os.path.join(data_folder_path,'combined')
final_csv_filename = f"combined_{current_date}.csv"
file_path = os.path.join(combined_folder_path, final_csv_filename)

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/check_scraping_status', methods=['GET'])
def check_scraping_status():
    # Check the current status of the scraping process
    return jsonify({'status': session.get('scraping_status', 'not_started')})

@app.route('/dashboard', methods=['POST'])
def dashboard():
    user_location = request.form['location']
    

    # Load the data into a DataFrame for analysis
    
    df= scraping(user_location)

    

    # Perform statistical analysis on the data and pass the results to the dashboard template
    statistics = {
        'total_rows': len(df),
        'average_price': df['price'].mean(),
        'min_price': df['price'].min(),
        'max_price': df['price'].max()
        # Add more statistics as needed
    }
    room_types = df['type'].unique()
    locations = df['location'].unique()


    return render_template('dashboard.html', location=user_location, statistics=statistics, room_types=room_types, locations=locations)
@app.route('/pretrain', methods=['POST'])
def pretrain():
    nltk.download('averaged_perceptron_tagger')
    current_date = datetime.now().strftime("%Y-%m-%d")
    df = pd.read_csv(file_path)


    df['amenities'].fillna('', inplace=True)
    amenities_tfidf = vectorizer.fit_transform(df['amenities'])
    amenities_df = pd.DataFrame(amenities_tfidf.toarray(), columns=vectorizer.get_feature_names())
    from sklearn.preprocessing import LabelEncoder
    categorical_features = ['type', 'location']
    label_encoder = LabelEncoder()
    label_encoders = {}
    for feature in categorical_features:
       df[feature] = label_encoder.fit_transform(df[feature].str.lower())
       label_encoders[feature] = label_encoder
       joblib.dump(label_encoders[feature], f'{feature}_encoder.joblib')
    # Assigning the classes to the label encoders
    location_encoder = joblib.load('location_encoder.joblib')
    room_type_encoder = joblib.load('type_encoder.joblib')

    df_processed = pd.concat([df, amenities_df], axis=1)
    df_processed.drop(columns=['amenities','distance'], axis=1, inplace=True)
    df_processed.columns = [re.sub('[^A-Za-z]+', '', col) for col in df_processed.columns]
    df_processed.to_csv('preprocessed.csv', index=False)
    df_processed=df_processed.drop('ratings',axis=1)
    X = df_processed.drop('price', axis=1)  # Adjust the column name for your target variable
    y = df_processed['price']



    X_scaled = scaler.fit_transform(X)

    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

    # Concatenate the reduced dimensions with the target variable
    df_pca_target = pd.concat([df_pca, y], axis=1)

    data = df_pca_target

    regression_setup = setup(
        data,
        target='price',
        normalize=True,
        train_size=0.8,
        session_id=123,
        log_experiment=True,
        experiment_name='your_experiment_name'
        )
    best_models = compare_models()
    save_model(best_models,'saved_model')

    return jsonify({'status': 'Pretraining completed'})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        model=load_model('saved_model')
        user_location = request.form['location'].lower()
        user_room_type = request.form['room_type'].lower()
        user_amenities = request.form['amenities']

        # Load the label encoders for location and room type
        location_encoder = joblib.load('location_encoder.joblib')
        room_type_encoder = joblib.load('type_encoder.joblib')

        # Encode user inputs
        encoded_location = location_encoder.transform([user_location])[0]
        encoded_room_type = room_type_encoder.transform([user_room_type])[0]

        # Apply the same preprocessing to user amenities
        user_amenities_tfidf = vectorizer.transform([user_amenities])
        user_amenities_df = pd.DataFrame(user_amenities_tfidf.toarray(), columns=vectorizer.get_feature_names())

        # Create a dataframe with the user inputs
        user_input_df = pd.DataFrame({
            'location': [encoded_location],
            'type': [encoded_room_type]
        })
        user_processed = user_input_df.copy()
        user_processed.columns = [re.sub('[^A-Za-z]+', '', col) for col in user_processed.columns]
        user_processed = pd.concat([user_processed, user_amenities_df.iloc[:1]], axis=1)
        #user_processed.drop(columns=['amenities', 'distance'], axis=1, inplace=True)
    
        user_scaled = scaler.transform(user_processed)
        user_pca = pca.transform(user_scaled)
        user_pca_df = pd.DataFrame(data=user_pca, columns=['PC1', 'PC2'])
        predictions = predict_model(model, data=user_pca_df)
        predicted_price = predictions['prediction_label'].iloc[0]
        print(f"Predicted price for the given input: {predicted_price}")
        return jsonify({'predicted_price': predicted_price})


if __name__ == '__main__':
    app.run(debug=True)
