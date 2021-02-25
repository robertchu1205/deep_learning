import os, sys, base64
import gpus
from io import BytesIO
from flask import Flask, jsonify, request
from importlib import reload
import predict
# from tensorflow.keras.preprocessing import image

def show_usage():
    print("Usage:")
    print("\tpython serving.py <port>\n")
    exit(-1)

app = Flask(__name__)

#curl -X POST http://localhost:8501/v1/models/<component>:predict -d @sample.json --header "Content-Type: application/json"
@app.route("/v1/models/<component>:predict", methods=["POST"])
def comp_predict(component="nocomp"):
    req_json = request.get_json(force=True)
    instances = req_json["instances"]

    totalnums = len(instances)
    all_ng_response = {"pred_class":"NG", "confidence": -2.0}
    all_ng_payload = {"predictions": [all_ng_response for i in range(totalnums)]}

    if component == "nocomp":
        return jsonify(all_ng_payload)
    else:
        if f"{component}.py" not in os.listdir("predict"):
            return jsonify(all_ng_payload)
        else:
            try:
                response = eval(f"predict.{component}.main({instances})")
                if response != None:
                    payload = {"predictions": response}
                    return jsonify(payload)
                else:
                    return jsonify(all_ng_payload)
            except:
                return jsonify(all_ng_payload)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_usage()
    os.environ["FLASK_ENV"] = "development"
    app.run(host="0.0.0.0", port=sys.argv[1], debug=False, use_reloader=True)