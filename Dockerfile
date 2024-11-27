# Соберём образ контейнера и назовём его server_image_simple:
    #      docker build -t diplom_image .

# Затем запустим контейнер:
    # docker run -it --rm --name=server_container -p=5000:5000 diplom_image

    # D:\olga2\Downloads\recomend

    # Ключ -v требует указания путей, которые записываются в формате <путь из папки запуска терминала>:<путь в контейнере>:
        # $ docker run -it --rm -v <путь>:/my_volume  --name server_container -p=5000:5000 diplom_image
        # В моем случае
    #     $ docker run -it --rm -v D:/olga2/Diplom/diplom/recomendation:/my_volume  --name server_container -p=5000:5000 diplom_image



# Задаём базовый образ
FROM python:3.11

# # Копируем  вспомогательные файлы в рабочую директорию контейнера
VOLUME /my_volume

# RUN pip install --upgrade pip
COPY ./app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./data/events_impl.pkl ./
COPY ./data/train_pivot_sparse_implicit.pkl ./
COPY ./data/train_pivot_list_implicit.pkl ./
COPY ./models/best_model_implicit.pkl ./
COPY ./data/available_only.pkl ./
COPY ./data/unique_items.pkl ./


COPY ./app/server.py ./
CMD [ "python", "./server.py" ]