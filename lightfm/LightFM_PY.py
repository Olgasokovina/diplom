import pickle
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score
# from scipy.sparse import csr_matrix
import multiprocessing

random_state = 42

num_threads = multiprocessing.cpu_count()
print(f'num_threads = {num_threads}')


events = pd.read_pickle('events_1.pkl')
event_type = {
            'view': 0,
            'addtocart':0,
            'transaction': 1,
            }

events['event'] = events['event'].map(event_type)

split_date = '2015-07-30' # Почему именно такая дата: чтобы разбика примерно была  в соотношении как в эталонном примере 70:30
train = events[events['timestamp'].dt.strftime('%Y-%m-%d') < split_date]
test = events[events['timestamp'].dt.strftime('%Y-%m-%d') >= split_date]

# test = test.sample(frac=0.5, random_state=random_state)

print('File opened')

# Преобразование колонок в серии pandas и получение уникальных значений
all_itemid = pd.concat([pd.Series(events['itemid']), pd.Series(train['itemid']), pd.Series(test['itemid'])]).unique()
all_visitors = pd.concat([pd.Series(events['visitorid']), pd.Series(train['visitorid']), pd.Series(test['visitorid'])]).unique()
# all_parents = pd.concat([pd.Series(events['parentid']), pd.Series(train['parentid']), pd.Series(test['parentid'])]).unique()
all_available = pd.concat([pd.Series(events['available']), pd.Series(train['available']), pd.Series(test['available'])]).unique()
# all_categoryid = pd.concat([pd.Series(events['categoryid']), pd.Series(train['categoryid']), pd.Series(test['categoryid'])]).unique()
# all_day_of_week = pd.concat([pd.Series(events['day_of_week']), pd.Series(train['day_of_week']), pd.Series(test['day_of_week'])]).unique()
# all_year = pd.concat([pd.Series(events['year']), pd.Series(train['year']), pd.Series(test['year'])]).unique()
# all_month = pd.concat([pd.Series(events['month']), pd.Series(train['month']), pd.Series(test['month'])]).unique()
all_day = pd.concat([pd.Series(events['day']), pd.Series(train['day']), pd.Series(test['day'])]).unique()
all_hour = pd.concat([pd.Series(events['hour']), pd.Series(train['hour']), pd.Series(test['hour'])]).unique()
all_minute = pd.concat([pd.Series(events['minute']), pd.Series(train['minute']), pd.Series(test['minute'])]).unique()
# all_day_period = pd.concat([pd.Series(events['Day Period']), pd.Series(train['Day Period']), pd.Series(test['Day Period'])]).unique()

# Создание объекта Dataset
dataset = Dataset()
dataset.fit(
    all_visitors,
    all_itemid,
    user_features=all_visitors,
    item_features=pd.concat([
                            # pd.Series(all_parents),
                             pd.Series(all_available),
                            #  pd.Series(all_categoryid),
                            #  pd.Series(all_day_of_week),
                            #  pd.Series(all_year),
                            #  pd.Series(all_month),
                             pd.Series(all_day),
                             pd.Series(all_hour),
                             pd.Series(all_minute),
                            #  pd.Series(all_day_period)
                             ]))

# Построение матрицы признаков для элементов
item_features = dataset.build_item_features(
    [(itemid, [
                # parentid,
               available,
            #    categoryid,
            #    day_of_week,
            #    year,
            #    month,
               day,
               hour,
               minute,
            #    day_period
               ])
     for    itemid,
            # parentid,
            available,
            # categoryid,
            # day_of_week,
            # year,
            # month,
            day,
            hour,
            minute,
            # day_period
     in zip(events['itemid'],
            # events['parentid'],
            events['available'],
            # events['categoryid'],
            # events['day_of_week'],
            # events['year'],
            # events['month'],
            events['day'],
            events['hour'],
            events['minute'],
            # events['Day Period']
            )])

# Построение разреженной матрицы взаимодействий
(interactions, weights) = dataset.build_interactions(
    [(row['visitorid'], row['itemid'], row['event']) for index, row in train.iterrows()])

# Преобразование тестового набора
test_interactions = dataset.build_interactions(
    [(row['visitorid'], row['itemid'], row['event']) for index, row in test.iterrows()])[0]

# Обучение модели
model_lfm = LightFM(loss='warp',  # Определяем функцию потерь
                    random_state=random_state,  # Фиксируем случайное разбиение
                    learning_rate=0.05,  # Темп обучения
                    no_components=100  # Размерность вектора для представления данных в модели
                    )

print('model created')

model_lfm.fit(interactions,  # Обучающая выборка
              epochs=50,  # Количество эпох
              item_features=item_features,
              verbose=1  # Отображение обучения
              )

print('model fitted')

# Производим сериализацию и записываем результат в файл формата pkl
with open('my_volume/model_lfm.pkl', 'wb') as output:
    pickle.dump(model_lfm, output)

print('model save')

# Оценка модели на тестовой выборке
prec_score = precision_at_k(model_lfm, test_interactions, num_threads=num_threads, item_features=item_features).mean()
print(prec_score)
auc_score_ = auc_score(model_lfm, test_interactions, num_threads=num_threads, item_features=item_features).mean()
print(auc_score_)

with open("my_volume/Output_best.txt", "w+") as text_file:
    text_file.write('precision_at_k = ' + str(prec_score) + '\n' + 'auc_score = ' + str(auc_score_))