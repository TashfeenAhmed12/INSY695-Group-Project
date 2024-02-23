import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import joblib

# Your preprocessing functions here
#adding a feature: average days_since_prior_order for each user_id

def avg_days_since_prior_order(data):
    days_since_prior_order = data.groupby('user_id')['days_since_prior_order'].mean().reset_index()
    data = data.merge(days_since_prior_order, on='user_id', how='left')
    data = data.rename(columns={'days_since_prior_order_x':'days_since_prior_order', 'days_since_prior_order_y':'avg_days_since_prior_order'})
    return data


#most_common order_dow for each user_id : mode of order_dow for each user_id
def most_common_order_dow(data):
    order_dow = data.groupby('user_id')['order_dow'].agg(lambda x:x.value_counts().index[0]).reset_index()
    data = data.merge(order_dow, on='user_id', how='left')
    data = data.rename(columns={'order_dow_x':'order_dow', 'order_dow_y':'most_common_order_dow'})
    return data


#spliting order_hour_of_day into 4 categories
def split_order_hour_of_day(data):
    data['order_hour_of_day'] = pd.cut(data['order_hour_of_day'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])
    return data


#total number of orders for each user_id
def total_orders(data):
    total_orders = data.groupby('user_id')['order_number'].max().reset_index()
    data = data.merge(total_orders, on='user_id', how='left')
    data = data.rename(columns={'order_number_x':'order_number', 'order_number_y':'total_orders'})
    return data

#average number of products in each order for each user_id
def avg_products(data):
    avg_products = data.groupby('user_id')['add_to_cart_order'].mean().reset_index()
    data = data.merge(avg_products, on='user_id', how='left')
    data = data.rename(columns={'add_to_cart_order_x':'add_to_cart_order', 'add_to_cart_order_y':'avg_products'})
    return data

#fill na values in days_since_prior_order with 0
def fill_na(data):
    data['days_since_prior_order'] = data['days_since_prior_order'].fillna(0)
    return data

##imputing order_hour_of_day with most common order_hour_of_day
def impute_order_hour_of_day(data):
    data['order_hour_of_day'] = data['order_hour_of_day'].fillna(data['order_hour_of_day'].mode()[0])
    return data

# Function to preprocess the data
def preprocess_data(df):
    # Apply your preprocessing functions
    df = avg_days_since_prior_order(df)
    df = most_common_order_dow(df)
    df = split_order_hour_of_day(df)
    df = total_orders(df)
    df = avg_products(df)
    df = fill_na(df)
    df = impute_order_hour_of_day(df)
    
    # Convert categorical columns using one-hot encoding
    categorical_features = ['order_dow', 'order_hour_of_day', 'aisle', 'department', 'most_common_order_dow']
    df = pd.get_dummies(df, columns=categorical_features)
    
    # Set product_id and user_id as index if not already
    if 'product_id' in df.columns and 'user_id' in df.columns:
        df.set_index(['product_id', 'user_id'], inplace=True)
    
    # Select the relevant features for the model
    selected_features = ['aisle_fresh fruits', 'aisle_packaged vegetables fruits', 'avg_days_since_prior_order', 'avg_products', 'days_since_prior_order', 'department_dairy eggs', 'department_pantry', 'department_produce', 'department_snacks', 'most_common_order_dow_0', 'most_common_order_dow_1', 'most_common_order_dow_5', 'order_dow_0', 'order_dow_1', 'order_dow_3', 'order_dow_4', 'order_dow_6', 'order_hour_of_day_afternoon', 'order_hour_of_day_evening','total_orders']
    return df[selected_features]

# Function to load the trained model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Title of the app
st.title('Product Reorder Prediction')

# Upload the dataset
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Read the dataset
    master_dataset = pd.read_csv(uploaded_file)
    
    # Preprocess the dataset
    preprocessed_data = preprocess_data(master_dataset)
    
    # Load the model (provide the correct path to your model)
    model = load_model('catboost_model.pkl')
    
    # Predict reorders
    predictions = model.predict(preprocessed_data)
    preprocessed_data['reordered'] = predictions  # Add predictions to the dataframe
    
    # Display the predictions
    st.write("Predictions:")
    display_df = preprocessed_data.reset_index()[['product_id', 'user_id', 'reordered']]  # Reset index to show product_id and user_id
    st.dataframe(display_df)
 
# Add some instructions and information about the app
st.sidebar.header('User Instructions')
st.sidebar.text('Upload your dataset in CSV format.')
st.sidebar.text('Ensure the dataset has the required columns.')