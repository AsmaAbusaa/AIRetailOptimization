from flask import Flask,jsonify,session
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from xgboost import XGBRegressor
#from sklearn.metrics import mean_squared_error
from werkzeug.exceptions import BadRequest
import os

app = Flask(__name__)
app.secret_key='53f57a96faf2d75f32d5c4e9ce2e8754163a17cd7874b356'

items_features=["Month", "Year", "Week", "Quantity_Lag_1", "Quantity_Lag_2", "Monthly_Sum", "Monthly_Avg", "min", "max"]
target ='Quantity'

#@app.route('/processed', methods=['GET'])
@app.errorhandler(BadRequest)
def handle_bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(404)
def handle_not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.route('/process/<file_id>', methods=['POST']) # take file id / name
def process_file(file_id):
    try:
       file_path = os.path.join('uploads', f'{file_id}.csv')
       df = pd.read_csv(file_path)

    except Exception as e:
        return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
    
    df.drop(["Unnamed: 0","TotalSales","Discount","CustomerID"], axis=1, inplace = True)
    df['Date']=pd.to_datetime(df['Date'])
    processed_data=df

    session['file_path']= file_path

    basket = (processed_data 
            .groupby(['InvoiceID', 'ProductID'])['Quantity'] 
            .sum().unstack().reset_index().fillna(0) 
            .set_index('InvoiceID'))
        
    frq_items = apriori(basket.astype(bool), min_support=.002, use_colnames=True)
    rules = association_rules(frq_items, metric="lift", min_threshold=1)
    rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])

    # Extract unique items from the 'antecedents' and 'consequents' columns
    all_items_list = []
    for rule_items in rules['antecedents'].tolist() + rules['consequents'].tolist():
        for item in rule_items:
            if item not in all_items_list:
                all_items_list.append(item)

    session['sorted_products']=all_items_list
    
    return jsonify({'message': 'File uploaded and processed successfully'}), 200

@app.route('/products', methods=['GET']) # /Products
def mine_association_rules():
    try:
        product_ids = session.get('sorted_products')

        if product_ids is None:
            return jsonify({'error': 'No processed data available'}), 404
        return jsonify({'itemsets':  product_ids})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/orders', methods=['GET'])
def predict_orders():
    try:
        file_path=session.get('file_path')
        df = pd.read_csv(file_path)

        if file_path is None:
            raise BadRequest('No processed data available for daily orders')
        
        df.drop(["Unnamed: 0","TotalSales","Discount","CustomerID"], axis=1, inplace = True)
        df['Date']=pd.to_datetime(df['Date'])
        
        daily_orders_df=df.set_index('Date')
        daily_orders_df = daily_orders_df['InvoiceID'].resample('D').count() 
        daily_orders_df=pd.DataFrame({'Date':daily_orders_df.index,'Orders_Count':daily_orders_df.values})
        daily_orders_df.fillna(0,inplace=True)
        daily_orders_df=daily_orders_df.set_index('Date')
        daily_orders_df=create_time_features(daily_orders_df)
        daily_orders_df=add_lags(daily_orders_df)

        future=create_future_df(daily_orders_df)
        future_X=future.drop(['Orders_Count','isFuture'], axis=1)
    
        model,X_test,y_test=build_orders_model(daily_orders_df)

        future['Orders_Count']=model.predict(future_X)
        X_test['Orders_Count']=model.predict(X_test)
        predicted_df=pd.concat([X_test,future])

        result = {
            'actual_dates': list(y_test.index),
            'actual_orders': list(y_test),
            'predicted_dates': list(predicted_df.index),
            'predicted_orders': list(predicted_df['Orders_Count'])
        }
        #mae = mean_absolute_error(y_test, X_test['Orders_Count'])
        return jsonify(result)
    
    except BadRequest as e:
        return handle_bad_request(e)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<int:item_id>', methods=['GET'])
def predict_for_item(item_id):
    global items_features,target
    try:
        file_path=session.get('file_path')
        df = pd.read_csv(file_path)

        if file_path is None:
            raise BadRequest('No processed data available for daily orders')
        
        df.drop(["Unnamed: 0","TotalSales","Discount","CustomerID"], axis=1, inplace = True)
        df['Date']=pd.to_datetime(df['Date'])
        processed_data=df

        processed_item=pd.DataFrame()
        max=processed_data['Date'].max()
        min=processed_data['Date'].min()

        item=processed_data.loc[processed_data['ProductID']==item_id]

        item=rebuild_weekly_dataset_for_item(item,min,max)
        item['Date']=item['Date'].dt.to_timestamp()
        item=item.set_index('Date')
        processed_item=engineer_weekly_item_sales_features_for_product(item)
        processed_item=processed_item.set_index('Date')

        model = train_model_for_item(item_id,processed_item)

        X_train, X_test, y_train, y_test=split_data_for_item_predcition(processed_item,items_features,target)
        
        future = create_future_item(processed_item)
        future_X=future[items_features]

        future['Quantity']=model.predict(future_X)
        X_test['Quantity']=model.predict(X_test)

        predicted_df=pd.concat([X_test,future])

        result = {
            'actual_dates': list(y_test.index),
            'actual_orders': list(y_test),
            'predicted_dates': list(predicted_df.index),
            'predicted_orders': list(predicted_df['Quantity'])
        }
        rmse_ = mean_squared_error(y_test, X_test['Quantity'], squared=False)
        return jsonify(result)
    
    except BadRequest as e:
        return handle_bad_request(e)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
#    return plot_actual_vs_predicted(y_test.index,y_test,predicted_df.index,predicted_df['Quantity'],item_id,1)

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
    #global weekly_item_sales
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
    #weekly_item_sales=pd.concat([weekly_item_sales,product_data],axis=0)
    
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

    return model

def create_future_item(df):
    min=df.index.max()
    min_date= min+pd.Timedelta(days=1)
    max_date= min_date+pd.Timedelta(days=14)
    Date = pd.date_range(min_date,max_date, freq='W')

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
    
    min=df_copy.index.max()
    min_date= min+pd.Timedelta(days=1)
    max_date= min_date+pd.Timedelta(days=1)
    Date = pd.date_range(min_date,max_date, freq='D')

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
