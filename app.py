from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    patient_id = None
    patient_name = None

    if request.method == "POST":
        patient_id = request.form["patient-name"]
        patient_name = request.form["password"]

    return render_template("index.html", patient_id=patient_id, patient_name=patient_name)

if __name__ == "__main__":
    app.run(debug=True)
