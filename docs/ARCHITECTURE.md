# Архитектура и контракт миграции

`commerce_agent` — стабильный graph ID. `CommerceState` — типизированная граница между graph nodes и клиентскими проекциями. Вход содержит версию схемы и сообщения; tenant/user/roles устанавливает доверенный BFF, а не браузер.

## Владение слоями

| Слой | Ответственность |
| --- | --- |
| LangGraph | Порядок выполнения, conditional edges, reducers, checkpoints, interrupts |
| Domain subgraphs | Sales, Marketing, Supply, Price, Finance и принадлежащие им результаты |
| Business core | SQL, финансовые формулы, симуляции, audit, outbox, разрешение и исполнение действий |
| BFF | Авторизация, владельцы threads, ограничения API, маршруты `/v2/agent/*` |
| React transport | Команды, streaming, reconnect и отображение проекций |
| Observability | Контролируемый экспорт LangSmith EU / Langfuse; не источник истины |

Зависимости доменов внедряются через `SalesDependencies`, `MarketingDependencies`, `SupplyDependencies`, `PriceDependencies`, `FinanceDependencies`. Если они отсутствуют, соответствующая ветка явно возвращает `not_implemented`.

## Стабильное состояние

```text
schema_version, messages, tenant_id, user_id, request_id,
query_plan, agent_statuses,
sales_analysis, marketing_analysis, supply_analysis, price_analysis, finance_analysis,
research_result, detected_incidents, evidence_matrix, data_gaps,
action_tasks, confidence, manager_decision, pending_interrupt,
execution_receipts, errors
```

Владение полями доменов и reducers проверяется тестами. Сериализуемое состояние не должно содержать credentials. Ресурсы клиентов, подключения и модельные зависимости передаются отдельно.

## Что уже есть в коде

- Opt-in доменные subgraphs, типизированные adapters и проверка исходных фактов.
- Ingress guards, сериализация checkpoints и private journals.
- PostgreSQL protocol storage, commands, streaming и worker recovery paths.
- Bounded research, cross-domain feedback для отдельных режимов Marketing.
- Два независимых контролируемых tracing exporter.

## Граница завершённости

Наличие этих модулей не означает завершения общего workflow. В `nodes/core.py` сохраняются stub-реализации общего планирования, synthesis, gap resolution и action path. Deep Research не включён. Browser BFF и SQL executor принадлежат отдельному backend.

Production переключается только после проверки parity, isolation, recovery, latency/cost и rollback. Исходная миграция предполагает big-bang cutover после staging acceptance. В рамках экспорта ни переключение, ни удаление legacy runtime не выполнялись.

Rollback портфолио не затрагивает исходный сервис: репозиторий является отдельной копией. Для будущей интеграции rollback выключает opt-in для новых runs; существующие versioned checkpoints сохраняются для совместимого reader или read-only просмотра.
