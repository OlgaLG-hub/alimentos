#!/bin/bash
set -e

cd "$(dirname "$0")"

VENV_BIN="$(pwd)/Aliment.venv/bin/python"
APP_URL="http://localhost:8501"

if [ ! -x "$VENV_BIN" ]; then
  echo "No se encontró el entorno virtual en: $VENV_BIN"
  exit 1
fi

if ! command -v open >/dev/null 2>&1; then
  echo "No se pudo abrir el navegador automáticamente porque 'open' no está disponible en este sistema."
  echo "Abre esta URL manualmente: $APP_URL"
  exit 1
fi

"$VENV_BIN" -m streamlit run olga_d_project.py --server.headless true --browser.gatherUsageStats false &
APP_PID=$!

for i in {1..30}; do
  if curl -fsS "$APP_URL" >/dev/null 2>&1; then
    open "$APP_URL"
    echo "La app se abrió en el navegador."
    echo "PID del proceso de Streamlit: $APP_PID"
    wait "$APP_PID"
    exit 0
  fi
  sleep 1
done

echo "La app no respondió en $APP_URL dentro del tiempo esperado."
echo "Comprueba si el CSV y el entorno virtual están disponibles."
kill "$APP_PID" 2>/dev/null || true
exit 1
