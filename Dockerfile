Пререписать!!!


# Соберём образ контейнера и назовём его server_image_simple:
    #      docker build -t diplom_image .

# Затем запустим контейнер:
    # docker run -it --rm --name=server_container -p=5000:5000 diplom_image

    # D:\olga2\Downloads\recomend

    # Ключ -v требует указания путей, которые записываются в формате <путь из папки запуска терминала>:<путь в контейнере>:

    #     $ docker run -it --rm -v D:/olga2/Downloads/recomend/my_volume:/my_volume  --name lightfm_container diplom_image



# Задаём базовый образ
FROM python:3.11.10

# # Копируем  вспомогательные файлы в рабочую директорию контейнера
# VOLUME /my_volume

# COPY ./ratings.csv ./
# COPY ./books.csv ./
# COPY ./tags.csv ./
# COPY ./book_tags.csv ./

# # RUN pip install --upgrade pip
# COPY ./requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# COPY ./learning_lightfm.py ./
# CMD [ "python", "./learning_lightfm.py" ]