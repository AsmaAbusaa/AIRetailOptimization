from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from xgboost import XGBRegressor
import os

app = Flask(__name__)
api = Api(app)
# Placeholder for storing the processed data
processed_data = None
daily_orders_df=None
filtered_items=None
items_features=[]
target=None
item_models = {}
processed_orders=False
weekly_item_sales=pd.read_csv('seed_weekly.csv')
weekly_item_sales.drop('Unnamed: 0',axis=1,inplace=True)
df = pd.read_csv('file_out2.csv')
df.drop(["Unnamed: 0","TotalSales","Discount","CustomerID"], axis=1, inplace = True)
df['Date']=pd.to_datetime(df['Date'])
df=df.drop( df.query(" `Quantity`==0 ").index)    
processed_data = df

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
REQUIRED_COLUMNS = {'InvoiceID', 'ProductID', 'Date', 'Quantity'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class UploadCSV(Resource):
    def post(self):
        if 'file' not in request.files:
            return {'message': 'No file part'}, 400

        file = request.files['file']

        if file.filename == '':
            return {'message': 'No selected file'}, 400

        if file and allowed_file(file.filename):
            # Create the uploads folder if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            df_ = pd.read_csv(filepath)

            # Check if the DataFrame contains required columns
            if not set(REQUIRED_COLUMNS).issubset(df.columns):
                return {'message': f'File must contain columns: {", ".join(REQUIRED_COLUMNS)}'}, 400

            return {'message': 'File uploaded and saved successfully', 'data': df.to_dict()}, 200
        else:
            return {'message': 'Invalid file type'}, 400

api.add_resource(UploadCSV, '/upload')

@app.route('/processed', methods=['GET'])
def preprocessing_file():
    global processed_data,daily_orders_df,items_features,target,processed_orders
    if processed_data.notnull:
        #Daily orders processing data for the next day
        daily_orders_df=df.set_index('Date')
        
        daily_orders_df = daily_orders_df['Quantity'].resample('D').count() 
        daily_orders_df=pd.DataFrame({'Date':daily_orders_df.index,'Orders_Count':daily_orders_df.values})
        daily_orders_df.fillna(0,inplace=True)
        daily_orders_df=daily_orders_df.set_index('Date')
        daily_orders_df=create_time_features(daily_orders_df)
        daily_orders_df=add_lags(daily_orders_df)
        processed_orders=True
        items_features=["Month", "Year", "Week", "Quantity_Lag_1", "Quantity_Lag_2", "Monthly_Sum", "Monthly_Avg", "min", "max"]
        target='Quantity'
        return jsonify({'message': 'File uploaded and processed successfully'})

    return jsonify({'error': 'Invalid file format'})

@app.route('/products', methods=['GET']) # /Products
def mine_association_rules():
    global processed_data,filtered_items
    basket = (processed_data 
          .groupby(['InvoiceID', 'ProductID'])['Quantity'] 
          .sum().unstack().reset_index().fillna(0) 
          .set_index('InvoiceID'))
    frq_items = apriori(basket.astype(bool), min_support=.001, use_colnames=True)
    rules = association_rules(frq_items, metric="lift", min_threshold=1)
    # Step 1: Filter based on Support
    min_support_threshold = 0.001  # Set your desired support threshold
    filtered_rules_support = rules[rules['support'] > min_support_threshold]

    # Step 2: Filter based on Confidence
    min_confidence_threshold = 0.1  # Set your desired confidence threshold
    filtered_rules_confidence = filtered_rules_support[filtered_rules_support['confidence'] > min_confidence_threshold]
    # Extract unique items from the 'antecedents' and 'consequents' columns
    all_items = set(item for rule_items in filtered_rules_confidence['antecedents'].tolist() + filtered_rules_confidence['consequents'].tolist() for item in rule_items)
    filtered_items= list(all_items)
    
    return jsonify({'itemsets':  filtered_items})

@app.route('/orders', methods=['GET'])
def predict_orders():
    global daily_orders_df
    
    if not processed_orders:
        return jsonify({'error': 'you must process dataset before'})
    
    future=create_future_df(daily_orders_df)
    future_X=future.drop(['Orders_Count','isFuture'], axis=1)
   
    model,X_test,y_test=build_orders_model(daily_orders_df)
    future['Orders_Count']=model.predict(future_X)
    X_test['Orders_Count']=model.predict(X_test)
    predicted_df=pd.concat([X_test,future])

    return plot_actual_vs_predicted(y_test.index,y_test,predicted_df.index,predicted_df['Orders_Count'],0,0)

@app.route('/predict/<int:item_id>', methods=['GET'])
def predict_for_item(item_id):
    global items_features,item_models
    processed_item=pd.DataFrame()
    max=df['Date'].max()
    min=df['Date'].min()

    item=df.loc[df['ProductID']==item_id]

    item=rebuild_weekly_dataset_for_item(item,min,max)
    item['Date']=item['Date'].dt.to_timestamp()
    item=item.set_index('Date')
    processed_item=engineer_weekly_item_sales_features_for_product(item)
    processed_item=processed_item.set_index('Date')

    if item_id not in item_models:
        model = train_model_for_item(item_id,processed_item)
    else :
        model = item_models[item_id]

    X_train, X_test, y_train, y_test=split_data_for_item_predcition(processed_item,items_features,target)
    
    future = create_future_item(processed_item)
    future_X=future[items_features]
    feature_df=pd.concat([X_test,future_X])
    prediction = model.predict(feature_df)
    prediction=list(prediction)
    return plot_actual_vs_predicted(y_test.index,y_test,feature_df.index,prediction,item_id,1)

def build_orders_model(data):
    X = data.drop('Orders_Count', axis=1)# features 
    y = data['Orders_Count'] # target
    split_index = int(0.9 * len(y))

    X_train, X_test = X.iloc[:split_index, :], X.iloc[split_index:, :]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    daily_count_model =XGBRegressor( objective='reg:squarederror',
    learning_rate=0.1,
    max_depth=3,
    n_estimators=50,
    subsample=0.8,
    colsample_bytree=0.9)

    daily_count_model.fit(X_train, y_train)
    daily_count_model.save_model('orders_model.json')
    return daily_count_model,X_test,y_test

def create_time_features(df):
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    return df

def add_lags(df):
    df['Prev_Day_Orders'] = df['Orders_Count'].shift(1)
    df['Prev_Week_Orders'] = df['Orders_Count'].shift(7)
    df['Prev_Month_Orders'] = df['Orders_Count'].shift(30)

    window_size = 7  # Define the window size for rolling statistics
    df['Rolling_Avg_Quantity'] = df['Orders_Count'].rolling(window=window_size).mean()
    df['last_week_sum'] = df['Orders_Count'].rolling(window='7D').sum()

    return df

def rebuild_weekly_dataset_for_item(item_df,min_date,max_date):
    grouped_data = item_df.groupby(['ProductID', item_df['Date'].dt.to_period('W')])['Quantity'].sum().reset_index()
    product_ids = item_df['ProductID'].unique()
    
    start_week = min_date.to_period('W')
    end_week = max_date.to_period('W')
    Date = pd.period_range(start_week, end_week, freq='W')

    all_combinations = []
    for product_id in product_ids:
        for week in Date:
            all_combinations.append({'ProductID': product_id, 'Date': week})
   
    all_combinations_df = pd.DataFrame(all_combinations)
    
    items_sales = pd.merge(all_combinations_df, grouped_data, on=['ProductID', 'Date'], how='left').fillna(0)
    
    return items_sales

def engineer_weekly_item_sales_features_for_product(product_data):
    global weekly_item_sales
    date=product_data.index
    product_data['Week'] = product_data.index.isocalendar ().week
    product_data['Month'] = product_data.index.month
    product_data['Year'] = product_data.index.year

    # Creating lag features
    product_data['Quantity_Lag_1'] = product_data['Quantity'].shift(1).fillna(0)
    product_data['Quantity_Lag_2'] = product_data['Quantity'].shift(2).fillna(0)

    # Monthly aggregations
    monthly_sum = product_data.groupby(['Year', 'Month'])['Quantity'].sum().reset_index(name='Monthly_Sum')
    monthly_avg = product_data.groupby(['Year', 'Month'])['Quantity'].mean().reset_index(name='Monthly_Avg')
    monthly_stats = product_data.groupby(['Year', 'Month'])['Quantity'].agg(['min', 'max']).reset_index()

    # Merge the monthly aggregations back to the original dataframe
    product_data = pd.merge(product_data, monthly_sum, on=['Year', 'Month'], how='left')
    product_data = pd.merge(product_data, monthly_avg, on=['Year', 'Month'], how='left')
    product_data = pd.merge(product_data, monthly_stats, on=['Year', 'Month'], how='left')
    product_data['min'] = product_data['min'].tolist()
    product_data['max'] = product_data['max'].tolist()
    product_data['Date']=date
    weekly_item_sales=pd.concat([weekly_item_sales,product_data],axis=0)
    
    return product_data

def split_data_for_item_predcition(product_data, features, target):
         
    if not product_data.empty:
         print(product_data.head())
    else:
        print(f'No data found for this Product.')
    
    product_data = product_data.sort_index()

    X = product_data[features]
    y = product_data[target] 

    split_index = int(0.8 * len(y))
    X_train, X_test= X.iloc[:split_index, :], X.iloc[split_index:, :]
    y_train, y_test= y.iloc[:split_index], y.iloc[split_index:]
    return X_train, X_test, y_train, y_test

def train_model_for_item(item_id,product_data):
    global items_features,target

    model = XGBRegressor( objective='reg:squarederror',
    learning_rate=0.1,
    max_depth=3,
    n_estimators=50,
    subsample=0.8,
    colsample_bytree=0.9)
    
    X_train, X_test, y_train, y_test=split_data_for_item_predcition(product_data,items_features,target)
    model.fit(X_train, y_train)

    item_models[item_id] = model

    return model

def create_future_item(df):
    Date = pd.date_range('2023-03-25','2023-04-3', freq='W-SUN')
    future = pd.DataFrame(index=Date)
    future['isFuture'] = True

    df_copy = df.loc[:, ['ProductID', 'Quantity', 'Week', 'Month', 'Year']].copy()
    df_copy['isFuture'] = False

    df_and_future = pd.concat([df_copy, future],axis=0)
    date=df_and_future.index

    df_and_future = engineer_weekly_item_sales_features_for_product(df_and_future)
    df_and_future.fillna(0, inplace=True)
    df_and_future.set_index(date,inplace=True)
    future_df = df_and_future[df_and_future['isFuture']].copy()
    future_df.drop(['ProductID',	'Quantity','isFuture'],axis=1,inplace=True)

    return future_df

def create_future_df(df):
    df_copy=df.copy()
    Date = pd.to_datetime(['2023-03-26 00:00:00','2023-03-27 00:00:00'])
    future = pd.DataFrame(index=Date)
    future['isFuture'] = True
    df_copy['isFuture'] = False
    df_and_future = pd.concat([df_copy, future])
    df_and_future = create_time_features(df_and_future)
    df_and_future = add_lags(df_and_future)
    df_and_future.fillna(0,inplace=True)
    future_df= df_and_future.query('isFuture').copy()
    return future_df

import plotly.graph_objs as go

def plot_actual_vs_predicted(actual_x,actual_y, prediction_x,prediction_y,product_id,flag):
    # Plotting Actual vs Predicted
    if flag:
        title = title=f'Actual vs. Predicted for Product {product_id}'
    else: title='Actual vs. Predicted Orders count'
    
    trace_actual = go.Scatter(
        x=actual_x,
        y=actual_y,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    )

    trace_predicted = go.Scatter(
        x=prediction_x,
        y=prediction_y,
        mode='lines',
        name='Predicted',
        line=dict(color='red')
    )

    
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Index'),
        yaxis=dict(title='Quantity'),
        showlegend=True,
    )
    import plotly.io as pio
    # Create two subplots, one for Actual vs Predicted and another for Future Predictions
    fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
   
    html_output = pio.to_html(fig, full_html=False)
    return html_output

if __name__ == '__main__':
    app.run(debug=True)
