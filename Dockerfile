# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Без буферизации вывода — логи видно сразу через `docker compose logs -f`.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# git — чтобы кнопка «Обновить с GitHub» в боте могла сделать git pull прямо внутри контейнера
# (см. docker-compose.yml: весь репозиторий смонтирован в /app, а не только скопирован при сборке).
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    # /app смонтирован с хоста и принадлежит хостовому пользователю, а не root внутри контейнера —
    # без этого git отказывается работать с репозиторием («detected dubious ownership»).
    && git config --system --add safe.directory /app

# Только исходники пакета — этого достаточно для запуска, слой с зависимостями кэшируется
# и пересобирается лишь при изменении pyproject.toml или src/.
COPY pyproject.toml ./
COPY src ./src
# editable — пакет ссылается прямо на /app/src, а не на копию внутри образа. Так «Обновить с GitHub»
# (git pull в смонтированном /app + перезапуск процесса) подхватывает новый код без пересборки образа.
# Пересборка (docker compose up -d --build) нужна только когда меняются сами зависимости.
RUN pip install --no-cache-dir -e .

# База лежит здесь (см. DATABASE_URL в .env) — монтируйте volume поверх для сохранности данных.
RUN mkdir -p /app/data
VOLUME ["/app/data"]

CMD ["python", "-m", "lalascaner", "run"]
