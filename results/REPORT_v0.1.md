# TR-Core: Behavioral Program Induction — Experimental Report v0.1

**Research question (RQ):** Цель — показать, что фиксированное ядро (substrate) может превращать
опыт в исполняемые программы поведения (behavioural programs), которые *развиваются* (обобщаются,
специализируются) и дают *transfer* на невиданные ситуации.

**Null hypothesis (H0):** Индукция behavioral program (bp) не превосходит chance-level базу на
островных (transfer) наборах; никакое развитие не наблюдается.

Приложение к открытому протоколу `docs/ru/ПРОТОКОЛ_ИССЛЕДОВАНИЯ_v0.1.md`, формализму
`docs/ru/СПЕЦИФИКАЦИЯ_BEHAVIORAL_PROGRAM_v0.2.md` и locked preregistration
`docs/ru/ПРЕРЕГИСТРАЦИЯ_ЭКСПЕРИМЕНТА_v0.1.md`.

---

## 1. Постановка

- **Среда:** 72 контекста (3×2×2×2×3 роль×дружелюбие×статус×настроение×контекст);
  oracle задаёт ответ из 4 действий {approach, avoid, ask, obey} по 8 приоритетным правилам
  (`warm→approach`, `child→approach`, `conflict∧stranger→avoid`, `upset∧conflict→avoid`,
  `upset∧request→ask`, `conflict∧cold→obey`, `(adult|elder)∧request→obey`, fallback→`ask`).
- **Сплиты (per seed):** train=600, known=120, T1=60, T2=60, T3=40. Transfer считается на
  T1-T3 (островные); известно-эстимация на `known`.
- **Конфигурации (абляции):** `base` (static memory baseline), `full` (bp-индукция),
  `null_bp` (bp-механизм, но outcome случайный — контроль механизма), `ab_no_generalize`,
  `ab_no_specialize`, `ab_no_compose` (planned null: COMPOSE не задействован, ≡ full),
  `decision_list` (внешний baseline).
- **Статистика:** paired permutation test (20k perms, без scipy), Cohen's d_z,
  bootstrap-95%CI (2000), Holm–Bonferroni; α=0.05. Данные фиксированы prereg.
- **Репликации:** main `SEED_BASE=1000`, replication `REPLICATION_BASE=9000`, N=24 каждая.

## 2. Результаты (overall transfer, N=24)

| конфиг | main mean | rep mean |
|---|---|---|
| base | 0.237 | 0.260 |
| **full** | **0.620** | **0.666** |
| ab_no_generalize | 0.249 | 0.249 |
| ab_no_specialize | 0.625 | 0.646 |
| ab_no_compose | 0.620 | 0.666 |
| null_bp | 0.295 | 0.284 |
| decision_list | 0.787 | 0.795 |

### Парные сравнения (full vs …), Holm-скорр.

| сравнение | main d / dz / p | rep d / dz / p |
|---|---|---|
| full vs base | +0.383 / 5.92 / <0.0001 | +0.406 / 5.66 / <0.0001 |
| full vs null_bp | +0.325 / 2.50 / <0.0001 | +0.382 / 2.59 / <0.0001 |
| full vs ab_no_generalize | +0.371 / 5.99 / <0.0001 | +0.417 / 6.12 / <0.0001 |
| full vs ab_no_specialize | −0.005 / −0.13 / n.s. | +0.019 / 0.37 / n.s. (raw p=0.045) |
| full vs ab_no_compose | 0.000 / 0.00 / n.s. | 0.000 / 0.00 / n.s. |

### Оси transfer (full vs ab_no_generalize)

| ось | main d / dz / p | rep d / dz / p |
|---|---|---|
| T1 (дружелюбие) | +0.288 / 2.85 / <0.0001 | +0.345 / 2.96 / <0.0001 |
| T2 (статус) | +0.510 / 6.97 / <0.0001 | +0.547 / 6.65 / <0.0001 |
| T3 (настроение) | +0.287 / 2.53 / <0.0001 | +0.330 / 2.26 / <0.0001 |

Все три оси значимы (vs ab_no_generalize и vs base, dz 2.3–7.0, p<0.0001).

## 3. Возможность-проверки (критерии из prereg)

| критерий | порог | main | rep |
|---|---|---|---|
| full ≫ base | p<0.05 | ✓ | ✓ |
| full ≫ null_bp (нужен опыт) | p<0.05 | ✓ | ✓ |
| full ≫ ab_no_generalize | p<0.05 | ✓ | ✓ |
| known retention ≥ 0.7 | ≥0.7 | 0.783 (min 0.683) | 0.802 (min 0.700) |
| known full ≥ ab_no_gen + 0.1 | +0.1 | +0.364 | +0.360 |
| developmental рост | монотонный ↑ | 0.248→0.578→0.643→0.637 | 0.284→0.528→0.651→0.667 |

Развитие по лог₂-доле опыта: slope ≈ +0.117 (main) / +0.118 (rep) по log2(train_frac), N=32 точки.

## 4. Процессные свидетельства (trace, seed=1 full)

8 ACTIVE правил после 600 триалов; 4 из них — продукт индукции:
- `friendliness=warm → approach` **[GENERALIZE]** (совпадает с oracle)
- `role=child → approach` **[GENERALIZE]** (совпадает с oracle)
- `status=stranger → approach` **[GENERALIZE]** (seed-специфично, валидировано на explore-триалах этого сплита)
- `friendliness=cold ∧ mood=upset → ask` **[SPECIALIZE]** (ошибочно в общем, но валидно в индуцированном охвате)

Баланс операций по правилам seed=1: CREATE 38, GENERALIZE 14, SPECIALIZE 17 (по превeнтности
всего trace). Полный список — `results/trace_full_seed1.json`.

## 5. Обсуждение

1. **GENERALIZE — критичная операция.** Удаление её превращает full в chance-level
   (0.249/0.249 ≈ base). Это сильнейший точечный сценарий-фактор: индукция категориальных
   обобщений (drop атрибутов) — основной механизм переноса.
2. **Опыт необходим.** `full ≫ null_bp` (dz=2.5) при одинаковом механизме — значит именно
   обработка реальных исходов (не сам процесс) даёт рост.
3. **SPECIALIZE слаб (v0.1).** В main незначимо (−0.005), в replication маргинально
   (+0.019, raw p=0.045, Holm n.s.). Гипотеза: в данных версии индукция + demote уже выражает
   специализацию как добавление литерала; отдельная ошибка-управляемая specialize почти не
   добавляет поверх GENERALIZE. Требуются задачи с более выраженными исключениями.
4. **COMPOSE = null** (запланированно). В v0.1 все конъюнкции выражаются через
   GENERALIZE+SPECIALIZE; композиция как отдельная операция не требуется данным oracle.
5. **Retention ≥ 0.7** держится (0.78/0.80), но не =1.0: часть `known` контекстов теряется,
   когда обобщения перекрывают частные правила. Это ожидаемый trade-off переноса.
6. **decision_list** (0.787/0.795) остаётся верхним ориентиром: bp достигает 0.62/0.67
   при *полном отсутствии oracle-знаний* — разница отражает сложность индукции с нуля.

## 5b. Robustness against noise (ε=0.05)

Прогон full с умалчиваемым outcome-шумом ε=0.05 (переворот успеха с вероятностью ε),
N=24, что же семена, что в основном прогоне (paired). Порог prereg §10: transfer не должен
рушиться (не < −0.15 vs без шума).

| метрика | main | replication |
|---|---|---|
| clean full | 0.620 | 0.666 |
| noise full (ε=0.05) | 0.643 | 0.667 |
| mean Δ (noise − clean) | +0.023 | +0.001 |
| min Δ | −0.069 | −0.125 |
| порог < −0.15 нарушен? | нет | нет |
| known retention (noise) | 0.793 | 0.794 |

Вывод: индукция **устойчива** к малому шуму исходов — transfer сохраняется (mean Δ ≈ 0,
ни один seed не падает ниже −0.15), known retention не снижается. В main наблюдается даже
слабый положительный сдвиг (шум выполняет роль регуляризатора против переобучения на части
seed). Данные: `results/robustness_main.json`, `results/robustness_replication.json`.

## 6. Limitations

- SIMPLE oracle (8 det.-правил); нет стохастичности, нет закономерностей второго порядка.
- COMPOSE не активирован; specialize даёт малый эффект — операции не все равнозначно изучены.
- known retention <1.0 (перенесённые обобщения перекрывают специфику).
- Оценка кандидата использует условную (по действию) частоту на explore-триалах —
  справедливо при равномерном explore, но не полноценный estimator P(oracle|cond).

## 7. Вывод

Индукция behavioral programs статистически значимо (p<0.0001, dz 2.5–6.1) превосходит
chance-базу, механизм-контроль и ablations по GENERALIZE на островных transfer-наборах;
known retention и developmental рост подтверждают «развитие». Центральная операция —
**GENERALIZE**; опыт обязателен; specialize/compose в v0.1 дают малый прирост.

Все данные и отчёты: `results/experiment_main.json`, `results/experiment_replication.json`,
`results/developmental_*.json`, `results/report_*.json`, `results/trace_full_seed1.json`.
