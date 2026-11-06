# Подключение сервисов

## Локальный запуск без инфраструктуры

```bash
uv sync --frozen
uv run --frozen python -m examples.checkpoint_demo
uv run --frozen python -m benchmarks.evidence_payload
uv run --frozen python -m pytest
```

Tracing по умолчанию отключён. Для этих команд не нужен `.env`.

## PostgreSQL

```bash
cp .env.example .env
docker compose -f compose.persistence.yml up -d
uv run --env-file .env python scripts/checkpoint_db.py setup
uv run --env-file .env python scripts/protocol_db.py setup
```

Убедитесь, что порт `55432` свободен. Пароль в шаблоне предназначен только для локальной development БД. Интеграционные тесты создают записи в указанной БД: используйте отдельную тестовую базу, не production.

```bash
uv run --env-file .env python -m pytest
```

Docker daemon в окружении подготовки был недоступен: команды PostgreSQL здесь документированы, но их успешное выполнение не заявляется.

## Backend tools

Клиент `BusinessToolClient` получает `COMMERCE_AGENT_TOOLS_API_URL` и `COMMERCE_AGENT_TOOLS_SERVICE_TOKEN` через environment. Он проверяет схему ответа, ownership и ограничения payload. Токен не должен попадать в `CommerceState` или публичные outputs.

Нужен совместимый business-tools API, включая версии доменных контрактов и artifact references. Этот репозиторий не содержит его SQL-реализацию. Публичный snapshot `ecommerce-ai-backend` не содержит требуемые маршруты `/internal/v1/agent-tools/*`. Для живой интеграции нужен совместимый business-tools service и отдельная проверка parity.

## Internal API и frontend

`commerce_agent.protocol_read_api:app` содержит internal API. Его запускают на закрытом интерфейсе с отдельным service token и доверенным BFF context. Нельзя подменять полноценный BFF передачей tenant/user headers напрямую из браузера.

```bash
uv run --env-file .env uvicorn commerce_agent.protocol_read_api:app --host 127.0.0.1 --port 8124
```

`frontend-probe` содержит transport и контрактные проверки SDK, а не готовый продуктовый UI:

```bash
cd frontend-probe
npm ci --ignore-scripts
npm run typecheck
npm test
```

Live probes требуют отдельно запущенного BFF/runtime и тестового bearer token. Без них они пропускаются.

## Observability

`.env.example` перечисляет переменные tracing. LangSmith EU и Langfuse включаются независимо. Private scripts `check_langsmith_*` / `check_langfuse_*` требуют ключей и могут создавать синтетические traces; они не входят в запуск локального примера.

Default export masking и ingress validation сохраняются. Development debug — отдельный opt-in режим; он не является разрешением публиковать клиентские данные.
