from  flask import Flask,request,jsonify
from classifier import get_prediction

app = Flask(__name__)

@app.route("/")
def homepage() :
    return "welcome to my homepage"

@app.route("/predict-alphabet", methods = ["POST"])

def predict_data() :
    image = request.files.get("alphabet")
    prediction = get_prediction(image)
    return jsonify({
        "prediction of alphabet" : prediction
    })
if __name__ == "__main__" :
    app.run(debug = True)