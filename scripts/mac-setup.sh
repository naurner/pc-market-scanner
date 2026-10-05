#!/bin/bash
# Установка и запуск Lalascaner на Mac одной командой:  bash scripts/mac-setup.sh
# Можно запускать повторно: скрипт обновляет код, донастраивает .env и перезапускает бота.

cd "$(dirname "$0")/.." || exit 1

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }
fail() { printf '\n\033[31m✗ %s\033[0m\n' "$1"; exit 1; }

# set_env КЛЮЧ ЗНАЧЕНИЕ — заменить строку в .env (или добавить, если её нет)
set_env() {
  if grep -q "^$1=" .env; then
    sed "s|^$1=.*|$1=$2|" .env > .env.tmp && mv .env.tmp .env
  else
    echo "$1=$2" >> .env
  fi
}

say "1/6  Обновляю код"
git pull --ff-only || echo "(не удалось обновить код — продолжаю с тем, что есть)"

say "2/6  Настраиваю .env"
[ -f .env ] || cp .env.example .env
TOKEN=$(grep '^BOT_TOKEN=' .env | cut -d= -f2-)
if [ -z "$TOKEN" ]; then
  printf 'Вставьте токен бота от @BotFather и нажмите Enter: '
  read -r TOKEN
  [ -n "$TOKEN" ] || fail "Без токена бот не запустится"
  set_env BOT_TOKEN "$TOKEN"
fi

set_env TELEGRAM_ENABLED true
set_env AI_ENABLED true
set_env AI_PROVIDER ollama
set_env AI_BASE_URL ""
set_env AI_BATCH_SIZE ""

say "3/6  Ollama (бесплатный ИИ)"
if ! command -v ollama >/dev/null 2>&1; then
  command -v brew >/dev/null 2>&1 || fail "Не найден Homebrew. Установите Ollama вручную с https://ollama.com/download и запустите скрипт снова"
  brew install ollama || fail "Не удалось установить Ollama"
fi
if ! curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  brew services start ollama >/dev/null 2>&1
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 2
  done
fi
if ! curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Служба не поднялась, запускаю Ollama напрямую..."
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  sleep 5
fi
curl -s -m 3 http://localhost:11434/api/tags >/dev/null 2>&1 || fail "Ollama не запускается. Установите приложение с https://ollama.com/download, откройте его и запустите скрипт снова"
echo "Ollama работает."

say "4/6  Модель ИИ"
# Хотите конкретную модель (пропускает проверку размера): MODEL=qwen2.5:7b bash scripts/mac-setup.sh
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
# Модель делит память с macOS и Docker — берём с запасом, а не всё, что есть.
if [ "$MEM_GB" -ge 16 ]; then SAFE_MB=8192; DEFAULT_MODEL=qwen2.5:7b; else SAFE_MB=2200; DEFAULT_MODEL=qwen2.5:3b; fi

# Размер уже скачанной модели по имени из `ollama list` (в МБ), иначе пусто.
model_size_mb() {
  ollama list 2>/dev/null | awk -v n="$1" -v n2="$1:latest" '
    NR>1 && ($1==n || $1==n2) { v=$3; u=$4; if (u=="GB") printf "%d", v*1024+0.5; else printf "%d", v+0.5; exit }'
}

INSTALLED=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}')
if [ -z "${MODEL:-}" ]; then
  BEST=""; BEST_MB=999999; QWEN=""
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    case "$name" in *bge*|*embed*|*minilm*) continue ;; esac  # модели векторов не умеют разбирать объявления
    mb=$(model_size_mb "$name")
    [ -n "$mb" ] || continue
    [ "$mb" -le "$SAFE_MB" ] || continue  # не помещается в память Mac — пропускаем
    case "$name" in qwen2.5:*) [ -z "$QWEN" ] && QWEN="$name" ;; esac
    [ "$mb" -lt "$BEST_MB" ] && { BEST="$name"; BEST_MB="$mb"; }
  done <<< "$INSTALLED"
  if [ -n "$QWEN" ]; then
    MODEL="$QWEN"
  elif [ -n "$BEST" ]; then
    MODEL="$BEST"
  else
    MODEL="$DEFAULT_MODEL"
    if [ -n "$INSTALLED" ]; then
      echo "Памяти на Mac ${MEM_GB} ГБ — скачанные модели для неё великоваты, беру более лёгкую $MODEL"
    else
      echo "Моделей ещё нет. Памяти на Mac: ${MEM_GB} ГБ → беру $MODEL"
    fi
  fi
fi

if echo "$INSTALLED" | grep -qx "$MODEL" || echo "$INSTALLED" | grep -qx "$MODEL:latest"; then
  echo "Использую уже скачанную модель: $MODEL"
else
  echo "Скачиваю $MODEL (в первый раз может быть несколько гигабайт, подождите)..."
  ollama pull "$MODEL" || fail "Не удалось скачать модель $MODEL"
fi
set_env AI_MODEL "$MODEL"

EMBED_MODEL=$(grep -E '^EMBED_MODEL=' .env 2>/dev/null | cut -d= -f2)
EMBED_MODEL=${EMBED_MODEL:-bge-m3}
if echo "$INSTALLED" | grep -qx "$EMBED_MODEL" || echo "$INSTALLED" | grep -qx "$EMBED_MODEL:latest"; then
  echo "Модель векторов уже скачана: $EMBED_MODEL"
else
  echo "Скачиваю модель векторов $EMBED_MODEL (~1.2 ГБ) — для похожих объявлений и «за сколько продать»..."
  ollama pull "$EMBED_MODEL" || echo "(не удалось — бот скачает её сам при запуске)"
fi

say "5/6  Запускаю бота в Docker"
docker info >/dev/null 2>&1 || fail "Docker не запущен. Откройте приложение Docker Desktop, дождитесь зелёного значка и запустите скрипт снова"
docker compose up -d --build || fail "Не удалось запустить контейнер"

say "6/6  Проверяю, что ИИ подключён (первый раз до пары минут — модель загружается в память)"
sleep 5
docker compose exec -T lalascaner python -m lalascaner ai-check

printf '\n\033[32m✓ Готово.\033[0m Напишите боту @lala_sccaner_bot команду /start.\n'
echo "Смотреть, что делает бот:  docker compose logs -f"
