import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from implicit.cpu.lmf import LogisticMatrixFactorization


available_full = pd.read_pickle('available_only.pkl')
train_pivot_list = pd.read_pickle('train_pivot_list_implicit.pkl')
with open('train_pivot_sparse_implicit.pkl', 'rb') as pkl_file:
    train_pivot_sparse = pickle.load(pkl_file)
with open('best_model_implicit.pkl', 'rb') as pkl_file:
    best_model_implicit = pickle.load(pkl_file)
with open('unique_items.pkl', 'rb') as pkl_file:
    unique_items = pickle.load(pkl_file)
with open('events_impl.pkl', 'rb') as pkl_file:
    events_impl = pickle.load(pkl_file)
    
# available_full.timestamp = available_full.timestamp.
max_date = available_full.timestamp.max()
min_date = available_full.timestamp.min()

# Функция для обработки некорректных дат
def parse_dates(date):
    try:
        return pd.to_datetime(date, format='%Y-%m-%d')
    except ValueError:
        return pd.NaT


def top_n_month(date, n=3):
    events = events_impl[events_impl['available'] == 1]
    start_mday = date - pd.DateOffset(months=1)
    events = events[(events['timestamp'] > start_mday)&(events['timestamp'] < date)]
    top_n = events[events['event'] == 'transaction'].groupby('itemid', as_index=False)['event'].count().sort_values('event', ascending=False)
    top_n = top_n.itemid.to_list()[:n]
    return top_n


app = Flask(__name__)


@app.errorhandler(415)
def unsupported_media_type(error):
    response = jsonify(f'Error: Please enter a valid query. Content type must be application/json.')
    response.status_code = 415
    return response


@app.route('/')
def index():
    return "Test message. Server started!"


@app.route('/recommend', methods=['POST'])
def predict():
    # n - количество выдаваемых рекомендаций
    n = 3

    if request.content_type != 'application/json':
        return unsupported_media_type(415)

    if request.json is None:
        return jsonify(f'Please enter correct date or visitorid')
    if isinstance(request.json, list) and len(request.json) == 2:
        df = pd.DataFrame({'date': [request.json[0]], 'visitorid': [request.json[1]]})
    elif isinstance(request.json, dict) and 'date' in request.json and 'visitorid' in request.json:

        df = pd.DataFrame(request.json, index=[0], columns=['date','visitorid'])
    else:
        df = pd.read_json(request.json, orient='table')


    df['date'] = df['date'].apply(parse_dates)
    if df['date'].isna().any()  or df['visitorid'].isna().any():
        return jsonify(f'Please enter correct date or visitorid')
    if not df['visitorid'].str.isdigit().all():
        return jsonify(f'Please enter correct visitorid')
    if (df['date'][0] > max_date) or (df['date'][0] < min_date):
        return jsonify(f'Please enter correct date. Date must be between {min_date.strftime("%Y-%m-%d")} and {max_date.strftime("%Y-%m-%d")}')

    user_id = df['visitorid'][0]
    file_name = f"my_volume/recom_user_{user_id}_date_{df['date'][0].strftime('%Y-%m-%d')}.csv"
    available = available_full[(available_full['timestamp'] == df['date'][0])]['itemid'].tolist()
    recommendations_df = pd.DataFrame(columns=['date', 'visitorid', 'recomendation'])


    if user_id in train_pivot_list:
        # Получение индекса элемента
        index_user = train_pivot_list.index(user_id)
        recomendations_ids, _ = best_model_implicit.recommend(index_user, train_pivot_sparse[index_user], N=10)

        recommendations = unique_items[recomendations_ids].tolist()
        filtered_recommendations = [rec for rec in recommendations if rec in available][:n]


        num_to_add = 3 - len(filtered_recommendations)
        if num_to_add > 0:
            filtered_recommendations.extend(top_n_month(df['date'][0], n=n)[:num_to_add])
            recommendations = filtered_recommendations
        else:
           recommendations = top_n_month(df['date'][0], n=n)

    else:
        recommendations = top_n_month(df['date'][0], n=n)


    recommendations_df.recomendation = recommendations
    recommendations_df['date'] = df['date'][0]
    recommendations_df['visitorid'] = user_id
    recommendations_df.to_csv(file_name, index=False)

    return jsonify(f'Recommendations for user {user_id}: {recommendations}')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
