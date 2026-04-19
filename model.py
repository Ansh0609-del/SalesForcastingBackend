import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib

# LOAD DATA
train = pd.read_csv("train.csv", low_memory=False)
store = pd.read_csv("store.csv")

# MERGE
df = pd.merge(train, store, on='Store')

# HANDLE MISSING VALUES
df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

categorical_cols = df.select_dtypes(include=['object', 'string']).columns
df[categorical_cols] = df[categorical_cols].fillna('0')

# FEATURE ENGINEERING
df['Date'] = pd.to_datetime(df['Date'])
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['weekday'] = df['Date'].dt.weekday
df.drop('Date', axis=1, inplace=True)

# ENCODING
le_storetype = LabelEncoder()
le_assortment = LabelEncoder()
le_stateholiday = LabelEncoder()
le_promointerval = LabelEncoder()

df['StoreType'] = le_storetype.fit_transform(df['StoreType'].astype(str))
df['Assortment'] = le_assortment.fit_transform(df['Assortment'].astype(str))
df['StateHoliday'] = le_stateholiday.fit_transform(df['StateHoliday'].astype(str))
df['PromoInterval'] = le_promointerval.fit_transform(df['PromoInterval'].astype(str))

# DROP
if 'Customers' in df.columns:
    df.drop('Customers', axis=1, inplace=True)

# FEATURES
X = df.drop('Sales', axis=1)
y = df['Sales']

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN XGBOOST 🚀
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# EVALUATE
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))

# SAVE (COMPRESSED)
joblib.dump(model, "sales_model.pkl", compress=3)
joblib.dump(le_storetype, "le_storetype.pkl")
joblib.dump(le_assortment, "le_assortment.pkl")
joblib.dump(le_stateholiday, "le_stateholiday.pkl")
joblib.dump(le_promointerval, "le_promointerval.pkl")

print("XGBoost model saved!")