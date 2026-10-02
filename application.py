from flask import Flask, request, render_template
import json
import pickle 
import pandas as pd 
from custom_classes import CustomePreprocess

application = Flask(__name__)
app=application 

## import Random Forest regressor pickle file and feature names
model=pickle.load(open("housing_model.pkl", "rb"))
features=json.load(open("feature_names.json", "r"))


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict_data", methods=["GET", "POST"])
def predict_datapoint():
    if request.method=="POST":
        try:
            form_data=request.form
            data = {
    'type':                  request.form.get("type"),
    'squareMeters':          float(request.form.get("squareMeters")),
    'rooms':                 int(request.form.get("rooms")),
    'floor':                 int(request.form.get("floor")),
    'floorCount':            int(request.form.get("floorCount")),
    'buildYear':             int(request.form.get("buildYear")),
    'latitude':              float(request.form.get("latitude")),
    'longitude':             float(request.form.get("longitude")),
    'centreDistance':        float(request.form.get("centreDistance")),
    'poiCount':              int(request.form.get("poiCount")),
    'schoolDistance':        float(request.form.get("schoolDistance")),
    'clinicDistance':        float(request.form.get("clinicDistance")),
    'postOfficeDistance':    float(request.form.get("postOfficeDistance")),
    'kindergartenDistance':  float(request.form.get("kindergartenDistance")),
    'restaurantDistance':    float(request.form.get("restaurantDistance")),
    'collegeDistance':       float(request.form.get("collegeDistance")),
    'pharmacyDistance':      float(request.form.get("pharmacyDistance")),
    'ownership':             request.form.get("ownership"),
    'hasParkingSpace':       request.form.get("hasParkingSpace"),
    'hasBalcony':            request.form.get("hasBalcony"),
    'hasElevator':           request.form.get("hasElevator"),
    'hasSecurity':           request.form.get("hasSecurity"),
    'hasStorageRoom':        request.form.get("hasStorageRoom")}
            input_df = pd.DataFrame([data])
            result=model.predict(input_df)
            return render_template("home.html", results=round(result[0],2), form_data=form_data)
 
        except Exception as e:                         
            return render_template("home.html", 
                                   error=str(e),
                                   form_data=form_data)
 
 
    else:
        return render_template("home.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)



 