import sys

sys.path.append('lightfm/')

from collections import defaultdict

import pandas as pd
from func_for_lightfm import *


def part_day(hour):
    if hour >= 5 and hour < 11:
        return 'Morning'
    elif hour >= 11 and hour < 17:
        return 'Afternoon'
    elif hour >= 17 and hour < 23:
        return 'Evening'
    else:
        return 'Night'


def get_time_periods(hour):
    if hour >= 3 and hour < 7:
        return 'Dawn'
    elif hour >= 7 and hour < 12:
        return 'Morning'
    elif hour >= 12 and hour < 16:
        return 'Afternoon'
    elif hour >= 16 and hour < 22:
        return 'Evening'
    else:
        return 'Night'


# Функция для дополнения недостающих дат - воскресение
def add_missing_sundays(user_data, start_date, end_date):
    """
    добавляет пропущенные даты (воскресенье) для каждого товара от начальной даты до конечной даты
    в датесете должны быть столбцы 'timestamp' и 'itemid'
    пример использования
    available_full = available.groupby('itemid').apply(add_missing_sundays,
                                            start_date=start_date,
                                            end_date=end_date).ffill().bfill().reset_index(drop=True)

    """
    all_sundays = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
    full_data = pd.DataFrame({'timestamp':all_sundays})
    user_data = user_data.merge(full_data, on='timestamp', how='right')
    user_data['itemid'] = user_data['itemid'].ffill().bfill()
    return user_data


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        for i in range(len(user_ratings)):
            user_ratings[i] = user_ratings[i][0]
        top_n[uid] = user_ratings[:n]

    return top_n


def surprise_precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


# Функция для получения топ-N рекомендаций
def get_top_n_recommendations(model, user_id, n=3):
    # Получение списка всех items
    item_ids = np.arange(len(dataset.mapping()[2]))

    # Предсказание для всех items
    scores = model.predict(user_id, item_ids)

    # Получение топ-N рекомендаций
    top_items = np.argsort(-scores)[:n]

    return top_items
