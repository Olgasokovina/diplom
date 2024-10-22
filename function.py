import pandas as pd

def part_day(hour):
    if hour >=5 and hour < 11:
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
def add_missing_sundays(user_data,start_date,end_date):
    """
    добавляет пропущенные даты (воскресенье) для каждого товара от начальной даты до конечной даты
    в датесете должны быть столбцы 'timestamp' и 'itemid'
    пример использования
    available_full = available.groupby('itemid').apply(add_missing_sundays,start_date=start_date,end_date=end_date).ffill().bfill().reset_index(drop=True)

    """
    all_sundays = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
    full_data = pd.DataFrame({'timestamp': all_sundays})
    user_data = user_data.merge(full_data, on='timestamp', how='right')
    user_data['itemid'] = user_data['itemid'].ffill().bfill()
    return user_data



# Функция для дополнения недостающих дат
def add_missing_days(user_data,start_date,end_date):
    '''
    добавляет пропущенные даты для каждого товара от начальной даты до конечной даты
    в датесете должны быть столбцы 'timestamp' и 'itemid'
    пример использования
    available_full = available.groupby('itemid').apply(add_missing_sundays,start_date=start_date,end_date=end_date).ffill().bfill().reset_index(drop=True)
    '''

    all_days = pd.date_range(start=start_date, end=end_date, freq='d')
    full_data = pd.DataFrame({'timestamp': all_days})
    user_data = user_data.merge(full_data, on='timestamp', how='right')
    user_data['itemid'] = user_data['itemid'].ffill().bfill()

    return user_data