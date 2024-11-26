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


def top_n_month(date, n=3):
    events = events_impl[events_impl['available'] == 1]
    start_mday = date - pd.DateOffset(months=1)
    events = events[(events['timestamp'] > start_mday)&(events['timestamp'] < date)]
    top_n = events[events['event'] == 'transaction'].groupby('itemid', as_index=False)['event'].count().sort_values('event', ascending=False)
    top_n = top_n.itemid.to_list()[:n]
    return top_n


app = Flask(__name__)


@app.route('/')
def index():
    return "Test message. Server started!"


@app.route('/recommend', methods=['POST'])
def predict():
    # n - количество выдаваемых рекомендаций
    n = 3
    df = pd.read_json(request.json, orient='table')
    df['date'] = pd.to_datetime(df['date'])
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
        if len(filtered_recommendations) != 0:

            # Сохранение рекомендаций в CSV файл с номером пользователя в имени
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
