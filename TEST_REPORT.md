# Проверка проекта

Выполнены локальные проверки:

## 1. Проверка синтаксиса

```bash
python -m compileall .
```

Результат: проект компилируется без синтаксических ошибок.

## 2. Smoke test

```bash
python tests/smoke_test.py
```

Что проверяется:

- индексация тестовой библиотеки;
- поиск в `Lite / слабое железо`;
- поиск в `Full / LLM + агент`;
- preview данных для адаптации embedding-модели и выбор стратегии LoRA/full.

Использован `stub`-режим локальной LLM для воспроизводимого теста без загрузки внешних весов.

Ожидаемый успешный вывод:

```text
[OK] lite mode search
[OK] full mode search
[OK] adaptation preview
[OK] smoke test finished
```
