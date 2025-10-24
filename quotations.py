import pandas as pd
import pickle
import json
import psycopg2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# DATABASE CONFIG
# ------------------------------
DB_CFG = {
    "dbname": "logibrain_demo",
    "user": "gunguna",
    "password": "7Ctzh6di3n35",
    "host": "demo.logibrain.ai",
    "port": 6432
}

# ------------------------------
# FASTAPI APP
# ------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow Power BI or local access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------------------
# LABEL COLUMNS
# ------------------------------
LABEL_COLS = ['lob_name','shipment_type','cargo_type','movement_type']

# ------------------------------
# FETCH DATA FUNCTION
# ------------------------------
def get_last_month_data():
    """Fetch last 1 month of quotation data"""
    conn = psycopg2.connect(**DB_CFG)
    query = """
        WITH quotation_lob AS (
            SELECT 
                quotation_id,
                lob_name,
                lead,
                shipment_type,
                cargo_type,
                packages,
                gross_weight_in_kgs,
                volume_cbm,
                charge_weight_in_kgs,
                movement_type,
                carrier,
                transit_destination,
                transit_days,
                origin,
                place_of_receipt,
                port_of_loading,
                port_of_discharge,
                place_of_delivery,
                final_destination
            FROM quotation_lobs
        ),
        info AS (
            SELECT 
                quotation_date,
                quotation_id,
                party_name,
                party_branch,
                quoted_by,
                sales_coordinator
            FROM quotations
        )
        SELECT 
            ql.*,
            i.*
        FROM quotation_lob ql
        LEFT JOIN info i 
            ON ql.quotation_id = i.quotation_id
        WHERE i.quotation_date >= NOW() - INTERVAL '1 month'
    """
    df = pd.read_sql(query, conn)
    conn.close()
    print("Rows fetched from Postgres:", len(df))
    return df

# ------------------------------
# PREPROCESSING FUNCTION
# ------------------------------
def preprocess_new_data(df, label_encoders, target_encodings):
    """Preprocess new data before prediction"""
    # 1️⃣ Label encode
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # 2️⃣ Target encode
    for col, mapping in target_encodings.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: mapping.get(x, mapping.get("_global_", 0.5)))

    # 3️⃣ Fill numeric NaNs
    df = df.fillna(0)
    return df

# ------------------------------
# PREDICTION ENDPOINT
# ------------------------------

@app.get("/")
def home():
    return {"message": "Quotation Predictor API is running!"}
@app.get("/predict")
def predict():
    try:
        # Load model + artifacts
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("selected_features.json", "r") as f:
            selected_features = json.load(f)
        with open("label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        with open("target_encodings.pkl", "rb") as f:
            target_encodings = pickle.load(f)

        # Fetch data (ONLY last 1 month)
        df_new = get_last_month_data()
        fetched_count = len(df_new)
        print(f"🟡 Step 1: Rows fetched from PostgreSQL = {fetched_count}")

        if df_new.empty:
            return JSONResponse(content={"message": "No data for last 1 month."})

        # Preprocess
        df_preprocessed = preprocess_new_data(df_new.copy(), label_encoders, target_encodings)
        preprocessed_count = len(df_preprocessed)
        print(f"🟢 Step 2: Rows after preprocessing = {preprocessed_count}")

        # Ensure all required features exist
        for col in selected_features:
            if col not in df_preprocessed.columns:
                df_preprocessed[col] = 0

        X_new = df_preprocessed[selected_features]

        # Predict
        y_pred = model.predict(X_new)
        y_proba = model.predict_proba(X_new)[:, 1] if hasattr(model, "predict_proba") else [None] * len(y_pred)
        predicted_count = len(y_pred)
        print(f"🔵 Step 3: Rows predicted by model = {predicted_count}")

        # Attach predictions
        df_preprocessed['prediction'] = y_pred
        df_preprocessed['probability'] = y_proba

        # Select only required columns
        out_cols = ['quotation_date', 'quotation_id', 'party_name', 'lob_name', 'prediction', 'probability']
        df_out = df_preprocessed[out_cols]

        returned_count = len(df_out)
        print(f"🟣 Step 4: Rows returned by API = {returned_count}")

        # Final consistency check
        if not (fetched_count == preprocessed_count == predicted_count == returned_count):
            print("⚠️ WARNING: Row count mismatch detected!")
            print(f"Fetched: {fetched_count}, Preprocessed: {preprocessed_count}, Predicted: {predicted_count}, Returned: {returned_count}")

        return JSONResponse(content={
            
            "predictions": jsonable_encoder(df_out.to_dict(orient='records'))
        })

    except Exception as e:
        print("❌ Error in /predict:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)