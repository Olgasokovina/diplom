import pandas as pd
from collections import defaultdict

# Функция для дополнения недостающих дат
def add_missing_days(user_data,start_date,end_date):
    '''
    добавляет пропущенные даты для каждого товара от начальной даты до конечной даты
    в датесете должны быть столбцы 'timestamp' и 'itemid'
    пример использования
    available_full = available.groupby('itemid').apply(add_missing_days,start_date=start_date,end_date=end_date).ffill().bfill().reset_index(drop=True)
    '''

    all_days = pd.date_range(start=start_date, end=end_date, freq='d')
    full_data = pd.DataFrame({'timestamp': all_days})
    user_data = user_data.merge(full_data, on='timestamp', how='right')
    user_data['itemid'] = user_data['itemid'].ffill().bfill()

    return user_data