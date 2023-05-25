from flask import Flask, render_template, request, redirect
from sklearn.preprocessing import OneHotEncoder
import librosa
import numpy as np
from tensorflow import keras
import pandas as pd


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def emotion_prediction():
    prediction = ""

    if request.method == "POST":
        print(" FORM DATA RECEIVED")

        if "file" not  in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            df = pd.read_csv('./data_path.csv')

            encoder = OneHotEncoder()
            y = encoder.fit_transform(df[['label']])

            my_model = keras.models.load_model('./mymodel.h5')
            ans = []

            y, sr = librosa.load(file, duration=3, offset=0.5)
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

            ans.append(mfcc)
            ans = np.array(ans)

            prediction = my_model.predict(ans)
            prediction = encoder.inverse_transform(prediction)
            
    return render_template('index.html', predicted_emotion=prediction)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)