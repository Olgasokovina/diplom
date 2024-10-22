# Пререписать!!!


# Соберём образ контейнера и назовём его server_image_simple:
    #      docker build -t sur_1_image .

# Затем запустим контейнер:
    # docker run -it --rm --name=server_container -p=5000:5000 sur_1_image

    # D:\olga2\Downloads\recomend

    # Ключ -v требует указания путей, которые записываются в формате <путь из папки запуска терминала>:<путь в контейнере>:

    #     $ docker run -it --rm -v my_volume:/my_volume  --name sur_1_container sur_1_image



# Задаём базовый образ
FROM python:3.11.10

# # Копируем  вспомогательные файлы в рабочую директорию контейнера
VOLUME /my_volume

# надо другой путь!!!
#   D:\olga2\Diplom\diplom\data\test_time.pkl
COPY ././diplom/data/train_time.pkl ./   
# COPY ../data/test_time.pkl ./


# # RUN pip install --upgrade pip
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./sur_1.py ./
CMD [ "python", "sur_1.py" ]