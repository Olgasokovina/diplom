
# Соберём образ контейнера и назовём его server_image_simple:
    #      docker build -t lightfm_image .


# Ключ -v требует указания путей, которые записываются в формате <путь из папки запуска терминала>:<путь в контейнере>:
    #     $ docker run -it --rm -v D:/olga2/Diplom/diplom/lightfm/my_volume:/my_volume  --name lightfm_container lightfm_image



# Задаём базовый образ
FROM python:3.11.10

# Копируем  вспомогательные файлы в рабочую директорию контейнера
VOLUME /my_volume

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY ./data/events_1.pkl ./

COPY ./LightFM_PY.py ./

CMD [ "python", "./LightFM_PY.py" ]