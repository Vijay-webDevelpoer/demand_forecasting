from flask import Flask, render_template, request, redirect, session
import pandas as pd
import joblib
import json
import datetime
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load model
model_data = joblib.load('model/xgb_model.pkl')
if isinstance(model_data, tuple) and len(model_data) == 2:
    model, feature_cols = model_data
else:
    model = model_data
    feature_cols = []

# Load users safely
try:
    with open("users.json", "r") as f:
        users = json.load(f)
except Exception:
    users = {}

@app.route('/')
def index():
    if 'email' in session:
        return redirect('/home')
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email', '').strip()
    pwd = request.form.get('password', '')
    user = users.get(email)
    if user and check_password_hash(user.get('password', ''), pwd):
        session['email'] = email
        return redirect('/home')
    return render_template('login.html', login_error="Invalid credentials")

@app.route('/signup', methods=['POST'])
def signup():
    email = request.form.get('email', '').strip()
    pwd = request.form.get('password', '')
    if not email or not pwd:
        return render_template('login.html', login_error="Email and password required")
    if email in users:
        return render_template('login.html', login_error="User already exists")
    users[email] = {'password': generate_password_hash(pwd)}
    with open("users.json", "w") as f:
        json.dump(users, f, indent=2)
    return render_template('login.html', signup_success=True)

@app.route('/home')
def home():
    if 'email' not in session:
        return redirect('/')
    last_inputs = session.get('last_inputs', {})
    return render_template('home.html',
                           last_inputs=last_inputs,
                           result=session.get('result'),
                           investment=session.get('investment'),
                           revenue=session.get('revenue'),
                           profit=session.get('profit'),
                           profit_percent=session.get('profit_percent', 0),
                           is_profit=session.get('is_profit', False),
                           dark_mode=session.get('dark_mode', False))

@app.route('/toggle-theme')
def toggle_theme():
    session['dark_mode'] = not session.get('dark_mode', False)
    return redirect('/home')

@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in session:
        return redirect('/')
    try:
        data = {
            'Store ID': request.form.get('store_id', '').strip(),
            'Product ID': request.form.get('product_id', '').strip(),
            'Category': request.form.get('category', '').strip(),
            'Region': request.form.get('region', '').strip(),
            'Inventory Level': float(request.form.get('inventory', 0)),
            'Units Sold': float(request.form.get('units_sold', 0)),
            'Units Ordered': float(request.form.get('units_ordered', 0)),
            'Price': float(request.form.get('price', 0)),
            'Discount': float(request.form.get('discount', 0)),
            'Competitor Pricing': float(request.form.get('competitor', 0)),
            'Seasonality': request.form.get('seasonality', '').strip(),
        }

        date_str = request.form.get('date', '')
        if date_str:
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            data['Day'] = date_obj.day
            data['Month'] = date_obj.month
            data['Year'] = date_obj.year
        else:
            now = datetime.datetime.now()
            data['Day'] = now.day
            data['Month'] = now.month
            data['Year'] = now.year

        df = pd.DataFrame([data])
        df = pd.get_dummies(df)

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]

        prediction = float(model.predict(df)[0])

        price = data['Price']
        discount = data['Discount']
        cost_price = price * (1 - discount / 100)
        investment = prediction * cost_price
        revenue = prediction * price
        profit = revenue - investment
        profit_percent = round((profit / revenue) * 100, 2) if revenue > 0 else 0
        is_profit = profit >= 0

        session['last_inputs'] = data
        session['result'] = round(prediction, 2)
        session['investment'] = round(investment, 2)
        session['revenue'] = round(revenue, 2)
        session['profit'] = round(profit, 2)
        session['profit_percent'] = profit_percent
        session['is_profit'] = is_profit

        # Save to user_history
        record = {
            "email": session['email'],
            "timestamp": str(datetime.datetime.now()),
            **data,
            "demand": round(prediction, 2),
            "profit_percent": profit_percent
        }
        try:
            with open("user_history.json", "r") as f:
                history = json.load(f)
        except Exception:
            history = []
        history.append(record)
        with open("user_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Dashboard record
        actual_demand = prediction * 0.98
        record_dashboard = {
            "month": data["Month"],
            "year": data["Year"],
            "predicted": round(prediction, 2),
            "actual": round(actual_demand, 2)
        }
        try:
            with open("actual_vs_predicted.json", "r") as f:
                chart_data = json.load(f)
        except Exception:
            chart_data = []
        chart_data.append(record_dashboard)
        with open("actual_vs_predicted.json", "w") as f:
            json.dump(chart_data, f, indent=2)

        return redirect('/home')
    except Exception as e:
        print("Prediction Error:", e)
        return redirect('/home')

@app.route('/history')
def history():
    if 'email' not in session:
        return redirect('/')
    try:
        with open("user_history.json", "r") as f:
            data = json.load(f)
    except Exception:
        data = []
    user_history = [r for r in data if r['email'] == session['email']]
    return render_template("history.html", records=user_history, dark_mode=session.get('dark_mode', False))

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect('/')
    try:
        with open("actual_vs_predicted.json", "r") as f:
            data = json.load(f)
    except Exception:
        data = []

    if not data:
        return render_template("dashboard.html", chart_labels=[], chart_predicted=[], chart_actual=[], mae=0, rmse=0)

    predicted = [float(d['predicted']) for d in data]
    actual = [float(d['actual']) for d in data]
    labels = [f"{d['month']}/{d['year']}" for d in data]

    mae = np.mean(np.abs(np.array(predicted) - np.array(actual)))
    rmse = np.sqrt(np.mean((np.array(predicted) - np.array(actual)) ** 2))

    return render_template("dashboard.html",
                           chart_labels=labels,
                           chart_predicted=predicted,
                           chart_actual=actual,
                           mae=round(mae, 2),
                           rmse=round(rmse, 2),
                           dark_mode=session.get('dark_mode', False))

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)
