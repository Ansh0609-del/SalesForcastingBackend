from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# -------------------------------
# LOAD MODEL + ENCODERS (FIXED PATHS)
# -------------------------------
model = joblib.load("sales_model.pkl")

le_storetype = joblib.load("le_storetype.pkl")
le_assortment = joblib.load("le_assortment.pkl")
le_stateholiday = joblib.load("le_stateholiday.pkl")
le_promointerval = joblib.load("le_promointerval.pkl")

# LOAD STORE DATA
store_df = pd.read_csv("store.csv")


@app.route("/")
def home():
    return "XGBoost Sales Forecasting API Running 🚀"


# -------------------------------
# HELPER FUNCTION
# -------------------------------
def prepare_input(store, promo, holiday, school, date):

    store_info = store_df[store_df["Store"] == store]

    if store_info.empty:
        return None

    store_info = store_info.iloc[0]

    store_type = le_storetype.transform([str(store_info["StoreType"])])[0]
    assortment = le_assortment.transform([str(store_info["Assortment"])])[0]
    state_holiday = le_stateholiday.transform([holiday])[0]
    promo_interval = le_promointerval.transform(["0"])[0]

    day = date.day
    month = date.month
    year = date.year
    weekday = date.weekday()

    input_data = pd.DataFrame([{
        "Store": store,
        "DayOfWeek": weekday + 1,
        "Open": 1,
        "Promo": promo,
        "StateHoliday": state_holiday,
        "SchoolHoliday": school,
        "StoreType": store_type,
        "Assortment": assortment,
        "CompetitionDistance": store_info["CompetitionDistance"] if pd.notnull(store_info["CompetitionDistance"]) else 0,
        "CompetitionOpenSinceMonth": store_info["CompetitionOpenSinceMonth"] if pd.notnull(store_info["CompetitionOpenSinceMonth"]) else 0,
        "CompetitionOpenSinceYear": store_info["CompetitionOpenSinceYear"] if pd.notnull(store_info["CompetitionOpenSinceYear"]) else 0,
        "Promo2": store_info["Promo2"],
        "Promo2SinceWeek": store_info["Promo2SinceWeek"] if pd.notnull(store_info["Promo2SinceWeek"]) else 0,
        "Promo2SinceYear": store_info["Promo2SinceYear"] if pd.notnull(store_info["Promo2SinceYear"]) else 0,
        "PromoInterval": promo_interval,
        "day": day,
        "month": month,
        "year": year,
        "weekday": weekday
    }])

    return input_data


# -------------------------------
# SINGLE PREDICTION
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        input_data = prepare_input(
            store=int(data["Store"]),
            promo=int(data["Promo"]),
            holiday=str(data["Holiday"]),
            school=int(data["SchoolHoliday"]),
            date=pd.to_datetime(data["Date"])
        )

        if input_data is None:
            return jsonify({"error": "Invalid Store ID"})

        prediction = model.predict(input_data)[0]

        return jsonify({"Predicted Sales": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------------
# CSV BATCH PREDICTION
# -------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        predictions = []

        for _, row in df.iterrows():

            input_data = prepare_input(
                store=int(row["Store"]),
                promo=int(row["Promo"]),
                holiday=str(row["StateHoliday"]),
                school=int(row["SchoolHoliday"]),
                date=pd.to_datetime(row["Date"])
            )

            if input_data is None:
                continue

            pred = model.predict(input_data)[0]
            predictions.append(int(pred))

        return jsonify({
            "message": "Batch prediction done",
            "sample_output": predictions[:5]
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------------
# RUN APP (RENDER FIX)
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)