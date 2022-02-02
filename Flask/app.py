from flask import Flask,request,render_template
import numpy as np
import  pickle

#create flask app
app=Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,5)
    loaded_model = pickle.load(open("HIC1.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route("/predict",methods=["POST"])
def predict():
    predict_list=[float(x) for x in request.form.values()]
    result=ValuePredictor(predict_list)
    return render_template("home.html",prediction_text="Premium cost is Rs %f"%result)

if __name__=="__main__":
    app.run(debug=True)