# Технический план для Codex: модель 5-летнего риска сердечно-сосудистой смертности

Дата фиксации и последнего содержательного обновления плана: 14 июля 2026 года.

Версия протокола: **2.4 — locked R backend, source-specific survey design and route-specific missingness**.

Обязательные решения amendment 2.4:

- secondary `recent_censored` охватывает только NHIS/NHANES 2015–2018; полный survival-анализ 1999–2018 не входит в v1;
- для pooled NHANES используются специальные четырехлетние веса 1999–2002 и корректные коэффициенты для каждого estimand;
- основная NHANES-переменная общего здоровья — `HUQ010`; `HSD010` используется только в sensitivity analysis;
- время NHIS сохраняется с квартальной точностью без искусственного преобразования в месяцы;
- NHANES сначала используется для одноразовой внешней проверки замороженной NHIS-модели и только после этого — для разработки клинических моделей;
- sample-size gate получает точные parameter budgets, формулы R² и numerical precision targets;
- после принятия amendment 2.4 labels held-out когорт до model lock доступны только через sealed aggregate audit; pre-amendment exposure документируется отдельно;
- CIF estimand, временные точки, competing-risk реализация, DCA thresholds, bootstrap и numerical tolerances фиксируются заранее;
- формулы единиц, `sbp_mean`, ожидаемое отсутствие `HSD010` в 1999–2000 и ограничения public-use LMF становятся частью data contract.
- pre-lock EDA ограничивается NHIS development; outcome plots, held-out outcome drift и subgroup/cycle cross-tabs выполняются только после lock и unseal;
- обычный Wilson gate заменяется design-based variance, design effect и effective sample size с учетом weights, strata и PSU;
- external-validation planning использует `pmvalsampsize 0.1.0`, явно заданные LP distribution, `simobs`, seed, CI scales и stability tolerance;
- ранее созданный `data/processed/nhanes_cohort.parquet` признается pre-amendment labelled artifact, помещается в quarantine по manifest и запрещается для v2.4.
- R является эталонным statistical backend для `survey`, `pmsampsize` и `pmvalsampsize`; Python остается backend для data engineering, ML, API и сайта;
- IPUMS NHIS использует уже интегрированные `STRATA`/`PSU` без добавления `YEAR`; NHANES использует опубликованные `SDMVSTRA`/`SDMVPSU` с отдельным collision audit;
- lonely PSU, design degrees of freedom, boundary prevalence и invalid variance получают точные blocker/status rules;
- event floors фиксируются по route: `500/500/200/200`, temporal `200`, external `100`;
- complete-case и imputation policy разделяются для `minimal`, `clinical_core` и paired `clinical_hba1c` analysis.

## 1. Как использовать этот файл

Этот документ является главным техническим заданием для разработки модели в рамках темы диссертации:

> **Development of Predictive Models for Cardiovascular Disease Risk Assessment Using Big Data and Machine Learning Techniques**

Codex должен прочитать файл полностью до начала реализации. Если фактические данные противоречат этому плану, нельзя незаметно менять когорту, целевую переменную или горизонт прогнозирования. Сначала нужно сформировать отчет о расхождении и согласовать изменение.

На текущем этапе необходимо разработать воспроизводимый исследовательский ML-пайплайн. Веб-сайт будет отдельным следующим этапом, но интерфейс входных признаков модели нужно зафиксировать уже сейчас.

Жесткое требование проекта: выбирать не самый простой, а наиболее сильный методологически защищаемый вариант в рамках нулевого бюджета. «Самый новый год» нельзя автоматически считать лучшим: для пятилетнего прогноза требуется достаточно длительное последующее наблюдение. Все решения о данных должны оцениваться по качеству outcome, полноте follow-up, воспроизводимости, внешней валидации и пригодности для веб-приложения.

## 2. Итоговая постановка исследования

### Рабочее название

**Development and External Validation of Interpretable Machine Learning Models for Five-Year Cardiovascular Mortality Risk Using Harmonized NHIS and NHANES Data**

### Основная задача

Разработать, откалибровать и проверить интерпретируемую модель, оценивающую вероятность смерти от сердечно-сосудистых причин в течение 5 лет у взрослых 40–79 лет без диагностированного сердечно-сосудистого заболевания на момент исходного обследования.

### Дизайн

- NHIS 1997–2009: разработка и подбор гиперпараметров быстрой модели.
- NHIS 2010–2014: временная валидация.
- NHANES 1999–2014: внешняя валидация быстрой модели на независимом исследовании.
- NHANES 1999–2014: разработка и внутренняя проверка `clinical_core` и кандидата `clinical_hba1c` как отдельный анализ.
- NHIS 2015–2018 и NHANES 2015–2018: отдельный secondary censored/short-horizon analysis; эти наблюдения не входят в основную полную пятилетнюю test cohort.
- Горизонт: 5 лет.
- Исход: смерть от болезней сердца или цереброваскулярных заболеваний.
- Основной сценарий: первичная профилактика, поэтому люди с исходным установленным ССЗ исключаются.

NHIS и NHANES нельзя просто объединять строками в одну таблицу и случайно делить на train/test. У них различаются способы измерения признаков, дизайн выборки и веса. Они должны оставаться отдельными когортами.

Полный pooled survival-анализ NHANES 1999–2018 является возможным будущим estimand, но не частью модели v1. Его нельзя незаметно подменять анализом `recent_censored` 2015–2018.

## 3. Почему используются именно эти данные

### NHIS

NHIS дает большую национально репрезентативную выборку и содержит признаки, которые пользователь способен быстро указать самостоятельно. Это основной источник для быстрой веб-модели.

Рекомендуемый способ получения NHIS — гармонизированный extract из IPUMS NHIS. Это существенно надежнее, чем вручную согласовывать названия и кодировки десятков ежегодных файлов CDC.

### NHANES

NHANES является независимым национальным обследованием. Оно содержит те же основные анкетные признаки, а также измеренное давление, антропометрию и лабораторные показатели. Поэтому NHANES используется:

1. для внешней проверки быстрой модели;
2. для сравнения `minimal`, `clinical_core` и кандидата `clinical_hba1c`.

### Linked Mortality Files

Исход определяется через связь NHIS/NHANES с National Death Index.

Важное ограничение: по состоянию на 14 июля 2026 года свободные **public-use** файлы имеют наблюдение только до 31 декабря 2019 года. Обновленные файлы с наблюдением до 2022 года являются restricted-use и требуют одобренной заявки в NCHS Research Data Center.

Для части записей public-use LMF NCHS заменил follow-up time или underlying cause of death синтетическими значениями для защиты конфиденциальности; vital status при этом не менялся. Это создает дополнительный outcome/time noise, который невозможно восстановить из public-use данных. Ограничение обязательно отражается в статье, model card и интерпретации cause-specific calibration.

Поэтому:

- primary fixed-horizon analysis использует baseline не позднее 2014 года;
- extract дополнительно включает 2015–2018 годы, поскольку это последние baseline-годы в public-use linkage и в сопоставимом NHIS design period;
- записи 2015–2018 годов используются только с явным временем наблюдения и цензурированием или для заранее заданного короткого горизонта;
- отсутствие зарегистрированной пятилетней CVD-смерти у участника 2015–2018 годов нельзя кодировать как обычный отрицательный `cvd_death_5y = 0`.

NHIS + NHANES выбраны как лучший бесплатный баланс большой выборки, настоящего будущего исхода, независимой внешней проверки и короткой пользовательской формы. ARIC + MESA являются клинически более прямой бесплатной альтернативой для incident MACE, но требуют институциональной заявки/DUA и другого протокола. Эти исходы нельзя смешивать внутри текущего эксперимента.

## 4. Ссылки на данные и документацию

### Обязательные источники

1. **IPUMS NHIS — создание гармонизированного extract:**  
   https://nhis.ipums.org/nhis/

2. **IPUMS NHIS — переменные Linked Mortality File:**  
   https://nhis.ipums.org/nhis-action/variables/group/mortality_mortality

3. **IPUMS NHIS — описание 2019 public-use mortality linkage:**  
   https://nhis.ipums.org/nhis/userNotes_LMF.shtml

4. **Оригинальные NHIS 1997–2018, запасной источник:**  
   https://archive.cdc.gov/www_cdc_gov/nchs/nhis/1997-2018.htm

5. **NHANES — страницы циклов, данные и кодбуки:**  
   https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx

6. **NHANES 1999–2000, начальная страница выбранного диапазона:**  
   https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=1999

7. **CDC public-use Linked Mortality Files — каталог файлов `.dat`:**  
   https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/

8. **Текущая страница NCHS National Death Index / Linked Mortality Files:**  
   https://www.cdc.gov/nchs/linked-data/mortality-files/index.html

9. **Описание public-use Linked Mortality Files:**  
   https://www.cdc.gov/nchs/data/datalinkage/public-use-linked-mortality-file-description.pdf

10. **Словарь переменных public-use Linked Mortality Files:**  
    https://www.cdc.gov/nchs/data/datalinkage/public-use-linked-mortality-files-data-dictionary.pdf

11. **Официальная программа чтения mortality-файлов:**  
    https://ftp.cdc.gov/pub/Health_statistics/NCHS/datalinkage/linked_mortality/R_ReadInProgramAllSurveys.R

12. **IPUMS: веса, STRATA, PSU и объединение нескольких лет:**  
   https://nhis.ipums.org/nhis/userNotes_variance.shtml

13. **ARIC — бесплатная request-based альтернатива через BioLINCC:**  
   https://biolincc.nhlbi.nih.gov/studies/aric/

14. **MESA — бесплатная request-based альтернатива через BioLINCC:**  
   https://biolincc.nhlbi.nih.gov/studies/mesa/

15. **AHA PREVENT — современный клинический ориентир, не обучающий датасет текущего проекта:**  
   https://professional.heart.org/en/guidelines-and-statements/prevent-calculator

16. **AHA: определение и корректное измерение систолического давления:**  
   https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings

17. **MedlinePlus: общий холестерин и единицы липидограммы:**  
   https://medlineplus.gov/lab-tests/cholesterol-levels/

18. **MedlinePlus: HDL blood test:**  
   https://medlineplus.gov/ency/article/007810.htm

19. **CDC: определение HbA1c:**  
   https://www.cdc.gov/diabetes/diabetes-testing/prediabetes-a1c-test.html

20. **CDC NHANES: объединение циклов и расчет многолетних весов:**  
   https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx

21. **NHANES 1999–2000 HUQ: переменная `HUQ010`:**  
   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/HUQ.htm

22. **NHANES HSQ: различия между `HUQ010` и `HSD010`:**  
   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles/HSQ_C.htm

23. **Riley et al.: расчет размера выборки для разработки prediction model:**  
   https://www.bmj.com/content/368/bmj.m441.abstract

24. **Riley et al.: расчет размера выборки для external validation:**  
   https://www.bmj.com/content/384/bmj-2023-074821

25. **CRAN `pmsampsize 1.1.3` и точные входные параметры:**  
   https://cran.r-project.org/package=pmsampsize

26. **NHANES HSQ 1999–2000: `HSD010` еще отсутствует:**  
   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/HSQ.htm

27. **NHANES HSQ 2001–2002: появление `HSD010`:**  
   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles/HSQ_B.htm

28. **NHANES BPX: официальное правило среднего давления:**  
   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles/BPX.htm

29. **NGSP/IFCC: master equation для HbA1c:**  
   https://ngsp.org/ifccngsp.asp

30. **CRAN `pmvalsampsize 0.1.0`: external-validation sample size и LP inputs:**  
   https://cran.r-project.org/package=pmvalsampsize

31. **CRAN `survey 4.5`: design-based variance для complex survey samples:**  
   https://cran.r-project.org/package=survey

32. **IPUMS NHIS: integrated `STRATA`/`PSU` для одного или многих лет:**  
   https://nhis.ipums.org/nhis/userNotes_variance.shtml

### Нужные mortality-файлы NHANES

Скачать из каталога CDC следующие файлы:

- `NHANES_1999_2000_MORT_2019_PUBLIC.dat`
- `NHANES_2001_2002_MORT_2019_PUBLIC.dat`
- `NHANES_2003_2004_MORT_2019_PUBLIC.dat`
- `NHANES_2005_2006_MORT_2019_PUBLIC.dat`
- `NHANES_2007_2008_MORT_2019_PUBLIC.dat`
- `NHANES_2009_2010_MORT_2019_PUBLIC.dat`
- `NHANES_2011_2012_MORT_2019_PUBLIC.dat`
- `NHANES_2013_2014_MORT_2019_PUBLIC.dat`
- `NHANES_2015_2016_MORT_2019_PUBLIC.dat`
- `NHANES_2017_2018_MORT_2019_PUBLIC.dat`

Для NHIS отдельные `.dat`-файлы CDC не нужны, если mortality-переменные уже включены в IPUMS extract.

## 5. Ручной шаг пользователя: IPUMS NHIS extract

IPUMS требует бесплатную учетную запись и принятие условий использования. Codex не должен просить пароль или пытаться обходить авторизацию.

Пользователь должен создать extract для годов **1997–2018** и сохранить полученный файл в:

```text
data/raw/nhis/
```

Предпочтительный формат: `.csv.gz`, `.csv` или Stata `.dta`. Вместе с данными нужно сохранить codebook, полученный для extract.

Годы 1997–2014 образуют primary fixed-horizon cohort. Годы 2015–2018 должны сохраняться с теми же переменными, но маркироваться отдельной ролью `recent_censored`; они не добавляются к бинарному пятилетнему train/test только ради увеличения выборки.

### Переменные, которые нужно включить в IPUMS extract

#### Идентификаторы и дизайн исследования

- `YEAR`
- `NHISPID`
- `QUARTER`
- `STRATA`
- `PSU`
- `MORTWTSA`

#### Основные признаки

- `AGE`
- `SEX`
- `BMI`
- `SMOKESTATUS2`
- `HYPERTENEV`
- `DIABETICEV`
- `HEALTH`

#### Audit-only демографические поля

- `RACEA`
- `HISPETH`

Эти поля не входят в основную веб-модель. Они нужны для описания когорты и subgroup/calibration audit. `RACEA` сохраняется ради сопоставимого покрытия 1997–2018 годов; любые ограничения исторической классификации должны быть описаны.

#### Исключение исходного ССЗ

- `CHEARTDIEV`
- `ANGIPECEV`
- `HEARTATTEV`
- `STROKEV`
- `CONHFAILEV`

#### Исход

- `MORTELIG`
- `MORTSTAT`
- `MORTUCODLD`
- `MORTDODY`
- `MORTDODQ`

IPUMS может автоматически добавить `SERIAL`, `NHISHID`, `HHX`, `FMX`, `PX`, `PERNUM`, стандартные веса и флаги `[preselected]`, а также `MORTWT`. Их не нужно удалять из extract. Они не становятся predictors автоматически; основной вес анализа sample-adult mortality остается `MORTWTSA`.

После получения extract Codex обязан сверить фактические имена, метки, коды пропусков и доступность по годам с приложенным IPUMS codebook. Нельзя считать, что числовой код `1` всегда означает `Yes` без проверки labels.

## 6. Автоматически загружаемые компоненты NHANES

Нужны циклы:

| Цикл | Суффикс файла |
| --- | --- |
| 1999–2000 | без суффикса |
| 2001–2002 | `_B` |
| 2003–2004 | `_C` |
| 2005–2006 | `_D` |
| 2007–2008 | `_E` |
| 2009–2010 | `_F` |
| 2011–2012 | `_G` |
| 2013–2014 | `_H` |
| 2015–2016 | `_I` |
| 2017–2018 | `_J` |

### Обязательные компоненты быстрой модели

- `DEMO`: возраст, пол, веса, strata и PSU.
- `BMX`: индекс массы тела.
- `SMQ`: курение.
- `BPQ`: диагностированная гипертензия.
- `DIQ`: диагностированный диабет.
- `HUQ`: основная самооценка общего здоровья через `HUQ010`.
- `HSQ`: `HSD010` загружается только для sensitivity analysis и проверки расхождений с `HUQ010`.
- `MCQ`: исходные сердечно-сосудистые заболевания.
- соответствующий NHANES mortality `.dat`.

### Дополнительные компоненты клинических моделей

- `BPX`: измерения систолического давления.
- `TCHOL`: общий холестерин.
- `HDL`: HDL-холестерин.
- `GHB`: гликированный гемоглобин HbA1c.

Codex должен сформировать `configs/nhanes_files.yaml`, где для каждого цикла указаны точные URL и ожидаемые имена файлов. Перед массовой загрузкой нужно проверить один цикл. Если CDC изменил имя компонента, скрипт должен завершиться с понятной ошибкой, а не пропустить файл.

## 7. Определение когорты

### Критерии включения: primary fixed-horizon cohort

- возраст от 40 до 79 лет включительно;
- участие в выбранном году или цикле;
- достаточные данные для linkage: `MORTELIG`/`ELIGSTAT` означает eligible;
- возможность полного пятилетнего наблюдения до 31 декабря 2019 года;
- отсутствие исходного диагностированного ССЗ;
- известный исход после linkage.

### Критерии включения: recent censored cohort 2015–2018

- возраст от 40 до 79 лет включительно;
- linkage-eligible adult;
- отсутствие исходного диагностированного ССЗ;
- известная дата или длительность наблюдения в доступной точности;
- исход кодируется через `followup_time` и `event_type`, а не через искусственный бинарный пятилетний target.

Primary и recent cohorts должны формироваться разными функциями или явными режимами одной функции. Их строки нельзя незаметно объединять для расчета обычных пятилетних ROC-AUC, Brier score или calibration.

### Исходное ССЗ

Исключить участника, если до обследования он сообщал хотя бы об одном из состояний:

- coronary heart disease;
- angina pectoris;
- myocardial infarction / heart attack;
- stroke;
- congestive heart failure.

NHIS-переменные: `CHEARTDIEV`, `ANGIPECEV`, `HEARTATTEV`, `STROKEV`, `CONHFAILEV`.

NHANES-переменные: `MCQ160C`, `MCQ160D`, `MCQ160E`, `MCQ160F`, `MCQ160B`. Codex должен проверить одинаковость кодов и доступность во всех циклах.

### Не менять возрастной диапазон автоматически

Если событий мало, Codex должен сначала показать числа. Расширение диапазона до 84 лет или изменение горизонта на 10 лет допустимо только как заранее согласованное изменение протокола.

## 8. Определение целевой переменной

### Сердечно-сосудистая смерть

```text
UCOD_LEADING ∈ {001, 005}
```

Где:

- `001` — diseases of heart;
- `005` — cerebrovascular diseases.

В IPUMS коды могут быть представлены иначе. Нужно сопоставлять по codebook/labels и затем выполнить assert по допустимым категориям.

### NHANES: точный пятилетний горизонт

Использовать `PERMTH_INT`, то есть число месяцев от интервью.

```text
cvd_death_5y = 1,
если MORTSTAT = deceased,
UCOD_LEADING является сердечно-сосудистой причиной,
и PERMTH_INT <= 60.
```

Все остальные linkage-eligible участники с полным потенциальным наблюдением получают `0`, включая:

- живых через 5 лет;
- умерших от другой причины до 5 лет;
- умерших от ССЗ позже 5 лет.

### NHIS: квартальная точность

NHIS public-use LMF содержит год и квартал смерти, а IPUMS содержит год и квартал интервью.

```text
followup_quarters =
    4 * (MORTDODY - YEAR) + (MORTDODQ - QUARTER)
```

Основное правило:

- сердечно-сосудистая смерть при `0 <= followup_quarters <= 19` — событие в течение 5 лет;
- случаи сердечно-сосудистой смерти при `followup_quarters = 20` имеют пограничную неопределенность до трех месяцев и исключаются из основного анализа;
- sensitivity analysis повторяет расчет, включая границу `<= 20`.

Codex должен проверить, что нет отрицательного времени до события. Если такие записи обнаружены, их нельзя молча исправлять.

### Secondary time-to-event target для 2015–2018 годов

Scope этого анализа строго ограничен baseline-годами 2015–2018. Он не включает циклы 1999–2014 и не использует веса полного pooled 1999–2018 estimand.

Создать как минимум:

```text
followup_time
followup_time_unit
followup_time_source
event_type = 0  # административно цензурирован / жив на конце follow-up
event_type = 1  # сердечно-сосудистая смерть
event_type = 2  # смерть от другой причины, competing event
```

Для NHANES использовать `PERMTH_INT`, `followup_time_unit = "month"` и `followup_time_source = "LMF_PERMTH_INT"`.

Для NHIS не создавать ложную месячную или дневную точность. Определить:

```text
quarter_index(year, quarter) = 4 * year + (quarter - 1)

для умершего:
followup_time = quarter_index(MORTDODY, MORTDODQ)
                - quarter_index(YEAR, QUARTER) + 1

для живого на конце follow-up:
followup_time = quarter_index(2019, 4)
                - quarter_index(YEAR, QUARTER) + 1

followup_time_unit = "quarter"
followup_time_source = "NHIS_INTERVIEW_AND_DEATH_QUARTER"
```

`+1` означает номер наблюдаемого квартального интервала и обеспечивает положительное время для смерти в квартале интервью; это не заявление о точной продолжительности в месяцах. Хранить также исходные `YEAR`, `QUARTER`, `MORTDODY`, `MORTDODQ` и флаг `time_interval_precision = "quarter"`. Отрицательное время, квартал вне `1..4` или смерть после административного окончания являются blocker, а не поводом для скрытой коррекции.

Основной secondary-метод для NHIS — discrete-time competing-risk analysis в квартальных интервалах. Cause-specific Cox с квартальными ties допускается как sensitivity analysis. Для NHANES разрешен непрерывный time-to-event подход в месяцах. Источники нельзя объединять в одну survival-модель без отдельного заранее утвержденного estimand.

Нельзя рассматривать `event_type = 2` как обычное независимое цензурирование при оценке cumulative incidence CVD death.

Точный estimand:

```text
F_CVD(t | X) = P(T <= t, event_type = CVD death
                 | alive and event-free at baseline, X)
```

Зафиксированные time points:

- primary recent horizon: NHIS `t = 4 quarters`, NHANES `t = 12 months`;
- secondary: `8 quarters` / `24 months`;
- exploratory: `12 quarters` / `36 months`.

Непараметрические CIF обеих причин рассчитываются Aalen–Johansen estimator. Для model-based analysis обязательно оценить обе cause-specific hazards: CVD death и non-CVD death; CIF восстанавливается совместно из обеих hazards. Fine–Gray допускается как заранее названный sensitivity analysis, но Kaplan–Meier с competing death как обычным censoring запрещен.

Discrimination/calibration на 24 и 36 месяцах публикуются как полноценные model metrics только если к горизонту наблюдается не менее `100` CVD-событий и weighted at-risk proportion непосредственно перед `t` составляет не менее `0.20` в соответствующем source. Иначе показываются CIF, counts at risk и CI с явной пометкой insufficient precision без claims о model performance.

Основная публикационная задача остается фиксированным пятилетним риском на полной follow-up cohort. Secondary survival analysis должен быть явно обозначен как дополнительный и не заменяет temporal/external validation 1997–2014/1999–2014.

## 9. Признаки быстрой модели и будущей веб-формы

### Канонические признаки модели

| Каноническое поле | Тип | NHIS | NHANES | Поле сайта |
| --- | --- | --- | --- | --- |
| `age` | число | `AGE` | `RIDAGEYR` | возраст |
| `sex` | категория | `SEX` | `RIAGENDR` | пол при рождении |
| `bmi` | число | `BMI` | `BMXBMI` | рассчитывается из роста и веса |
| `smoking_status` | 3 категории | `SMOKESTATUS2` | из `SMQ020` и `SMQ040` | никогда / раньше / сейчас |
| `hypertension_dx` | да/нет | `HYPERTENEV` | `BPQ020` | диагноз гипертензии |
| `diabetes_dx` | да/нет | `DIABETICEV` | `DIQ010` | диагноз диабета |
| `general_health` | 5 категорий | `HEALTH` | `HUQ010` | excellent–poor |

### Поля будущего сайта

Пользователь вводит только:

1. возраст;
2. пол при рождении;
3. рост;
4. вес;
5. статус курения;
6. диагностирована ли гипертензия;
7. диагностирован ли диабет;
8. общая самооценка здоровья.

BMI рассчитывается приложением. В модель нельзя передавать одновременно введенный BMI и рост/вес без проверки согласованности.

Провести обязательный ablation analysis с `general_health` и без него. Это субъективный показатель, и его понимание может различаться между языками и культурами; его польза должна быть показана экспериментально.

В форме для каждого неизвестного значения должен существовать честный путь «не знаю», но это не означает, что приложение обязано выдавать риск при любом наборе пропусков. Для v1 действуют следующие правила:

- `age`, `sex`, рост и вес для вычисления `bmi`, `smoking_status`, `hypertension_dx` и `diabetes_dx` обязательны для персонального расчета;
- если неизвестен только `general_health`, запрос может перейти на отдельный маршрут `minimal_no_general_health`, но только если модель без этого признака заранее обучена, откалибрована и валидирована;
- если неизвестен любой другой обязательный minimal-признак, приложение не рассчитывает персональный риск, объясняет, какого значения не хватает, и не подставляет медиану, «норму» или значение `No`;
- не создавать автоматически все комбинации fallback-моделей; каждый опубликованный route должен иметь собственный pipeline, метрики и запись в `routing_manifest.json`.

### Что не входит в предикторы быстрой модели

- раса и этничность;
- доход и образование;
- география;
- признаки, получаемые только при стресс-тесте, ангиографии или флюороскопии;
- признаки, появившиеся после начала наблюдения;
- причина смерти, длительность наблюдения или mortality weights как предикторы.

Раса/этничность, доход и образование могут использоваться только для описания когорты и subgroup/fairness audit.

## 10. Клинические модели NHANES и routing веб-формы

С точки зрения посетителя сайта все клинические показатели являются необязательными: без них всегда доступен `minimal` route. Нельзя создавать одну форму, которая заставляет каждого пользователя сдавать анализы.

### Определения клинических полей

| Каноническое поле | Значение для пользователя | Источник | Каноническая единица |
|---|---|---|---|
| `sbp_mean` | Среднее систолическое давление — верхнее число в записи, например `125` в `125/80` | NHANES: правило ниже для `BPX`; сайт: среднее минимум двух измерений | `mmHg` |
| `total_cholesterol` | Общий холестерин; в бланке может называться `Total cholesterol`, `TC`, `CHOL` | `TCHOL` | `mg/dL` внутри pipeline |
| `hdl_cholesterol` | HDL/HDL-C/ЛПВП; нельзя заменять LDL/ЛПНП | `HDL` | `mg/dL` внутри pipeline |
| `hba1c` | Гликированный гемоглобин, отражающий среднюю гликемию примерно за 2–3 месяца; не равен разовому измерению глюкометром | `GHB` | `%` внутри pipeline |

Пользователь может вводить холестерин в `mmol/L` или `mg/dL`, а HbA1c — в `%` или `mmol/mol`. Использовать только следующие формулы:

```text
cholesterol_mg_dl = cholesterol_mmol_l * 38.67
cholesterol_mmol_l = cholesterol_mg_dl / 38.67

hba1c_percent = 0.09148 * hba1c_mmol_mol + 2.152
hba1c_mmol_mol = (hba1c_percent - 2.152) / 0.09148
```

API обязан получить явную единицу, детерминированно преобразовать значение в каноническую систему и записать примененное преобразование в технический лог без сохранения персональных медицинских данных. В model pipeline сохраняется неокругленный `float64`; округление выполняется только для отображения. Unit tests используют `rtol = 1e-10`, `atol = 1e-10` для round-trip conversion.

Правило `sbp_mean`:

- NHANES: если валидна одна систолическая запись, использовать ее; если валидны две и более, исключить первую хронологическую запись и усреднить остальные валидные readings;
- не заменять отсутствующую запись нулем и не включать невалидные/прерванные измерения;
- сайт: пользователь вводит минимум два корректных измерения, выполненных в одинаковых единицах; использовать обычное арифметическое среднее всех введенных значений;
- при одном пользовательском измерении `clinical_core` не активируется и приложение предлагает `minimal`;
- model input не округляется, UI показывает `sbp_mean` с точностью `0.1 mmHg`.

### Отдельно валидируемые наборы признаков

1. `minimal`: семь канонических признаков без клинических измерений;
2. `clinical_core`: `minimal` + `sbp_mean` + `total_cholesterol` + `hdl_cholesterol`;
3. `clinical_hba1c`: `clinical_core` + `hba1c`.

`clinical_hba1c` включается в сайт только если HbA1c проходит заранее зафиксированные acceptance criteria на той же complete-case выборке и по out-of-fold predictions:

1. относительное улучшение основного weighted Brier score составляет не менее 2%;
2. paired cluster bootstrap 95% CI для `Brier_hba1c - Brier_core` полностью ниже `0`;
3. абсолютное отклонение calibration slope от `1` и calibration intercept от `0` не увеличивается более чем на `0.05`, а ухудшение ICI не превышает `0.002`;
4. на DCA thresholds `[0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]` нижняя граница 95% CI для `DeltaNetBenefit` не ниже `-0.0005` на всех порогах и выше `0` минимум на двух соседних порогах;
5. comparison cohort сама проходит sample-size/event stop-gate.

Для paired HbA1c inference использовать stratified cluster bootstrap по PSU внутри strata: `2000` repetitions, seed `20260714`. Если survey design конкретного pooled файла требует модифицированного rescaled bootstrap, реализация и scaling factor фиксируются до запуска; participant-level bootstrap не заменяет primary CI.

ROC-AUC не является самостоятельным критерием допуска. Если хотя бы один обязательный критерий не выполнен, HbA1c остается исследовательским результатом и не показывается как часть финального калькулятора.

Для активации `clinical_core` пользователь должен знать полный core-набор: систолическое давление, общий холестерин и HDL. Если core-набор неполный, приложение объясняет переключение на `minimal`; запрещено подставлять «нормальное» или среднее значение. Если core-набор полный, но HbA1c неизвестен, используется `clinical_core`, а не отказ в расчете.

Не добавлять десятки лабораторных показателей только ради повышения AUC. Клинические модели должны оставаться объяснимыми и реализуемыми по результатам обычного обследования.

Primary incremental HbA1c analysis проводится на одной paired complete-case NHANES cohort, где одновременно доступны все minimal fields, SBP, total cholesterol, HDL и HbA1c. `clinical_core` и `clinical_hba1c` обучаются и оцениваются на одних и тех же folds этой когорты. Это primary acceptance analysis, а не sensitivity analysis. Дополнительно показать:

- прирост ROC-AUC/PR-AUC/Brier/calibration с 95% CI;
- число и долю потерянных участников из-за каждого клинического поля;
- performance на полной доступной когорте для каждого route;
- multiple-imputation sensitivity analysis внутри folds; он не может сам по себе разрешить экспорт route;
- пользовательскую стоимость признаков: количество обязательных полей и необходимость анализа крови.

## 11. Особенности гармонизации

Codex должен создать явный слой гармонизации, а не переименовывать столбцы вручную в ноутбуке.

Файл:

```text
configs/feature_mapping.yaml
```

Каждый канонический признак должен содержать:

- источник;
- исходное имя;
- исходные коды;
- итоговые категории;
- правила пропусков;
- годы доступности;
- комментарий о сопоставимости.

Обязательные оговорки:

- BMI в NHIS основан на сообщенных росте и весе, а в NHANES — на измерениях;
- диагнозы гипертензии и диабета являются самоотчетом;
- формулировки и universe вопросов нужно сверять по циклам;
- «refused», «don't know», «not ascertained» и «not in universe» нельзя кодировать как `No`;
- `general_health` должен иметь один и тот же упорядоченный порядок категорий; основной NHANES source — `HUQ010`, а `HSD010` не должен автоматически заполнять его пропуски;
- создать `reports/tables/huq_hsd_disagreement.csv` с доступностью, cross-tab и weighted disagreement `HUQ010` против `HSD010`; для 1999–2000 записать `status = EXPECTED_NOT_AVAILABLE`, не считать это ошибкой и не создавать фиктивную колонку; comparison начинается с 2001–2002;
- единицы лабораторных показателей должны быть приведены к одной системе и проверены диапазонами;
- `HDL` и `LDL` должны иметь разные schema identifiers и не могут взаимозаменяться;
- функции преобразования `mmol/L ↔ mg/dL` для холестерина и `% ↔ mmol/mol` для HbA1c должны иметь unit tests и round-trip tolerance;
- сырое значение и указанная пользователем единица сохраняются только в памяти запроса, если пользователь явно не согласился на исследовательское хранение;
- неизвестные значения в live API маршрутизируют запрос к совместимой модели, а не проходят через скрытую клиническую imputation.

## 12. Survey design и веса

### Эталонный statistical backend

Python используется для загрузки/гармонизации данных, ML pipelines, API и веб-приложения. R является обязательным reference backend для complex-survey inference и sample-size calculations.

Зафиксированные зависимости:

```text
survey = 4.5
pmsampsize = 1.1.3
pmvalsampsize = 0.1.0
```

Точную версию R записать в `.R-version`, а все R dependencies — в `renv.lock`. Stage 0 создает `scripts/r/` и reference fixtures JSON/CSV. Если `Rscript` не найден, инфраструктуру репозитория можно создать, но Stage 0 получает `BLOCKED_RSCRIPT_NOT_FOUND`, а survey, `pmsampsize` и `pmvalsampsize` gates не выполняются.

Python reimplementation разрешается только как secondary portability layer после сравнения с фиксированными R outputs: absolute error для proportions/SE/CI не больше `1e-10`, relative error для `DEFF`/ESS не больше `1e-8`, required sample size совпадает как integer. Любое расхождение блокирует Python backend; primary article outputs остаются R-derived.

### NHIS

- основной вес: `MORTWTSA`;
- дизайн: `STRATA`, `PSU`;
- IPUMS уже интегрировал `STRATA` и `PSU` для одного или многих лет и разных sample-design periods;
- создавать `interaction(YEAR, STRATA)`, `interaction(YEAR, PSU)`, `STRATA_POOL` или `PSU_POOL` запрещено: это ошибочно объявит ежегодные выборки независимыми;
- reference design создается точно так:

```r
nhis_design <- survey::svydesign(
  ids = ~PSU,
  strata = ~STRATA,
  weights = ~analysis_weight,
  nest = TRUE,
  data = nhis_cohort
)
```

- для pooled описательных оценок годовой вес делится на число объединенных лет соответствующего анализа;
- split на 1997–2009 и 2010–2014 должен иметь свои корректно масштабированные веса.

### NHANES

- использовать MEC examination weights, поскольку BMI, давление и лабораторные данные связаны с MEC examination;
- дизайн: `SDMVSTRA`, `SDMVPSU`;
- published identifiers используются без изменения по умолчанию:

```r
nhanes_design <- survey::svydesign(
  ids = ~SDMVPSU,
  strata = ~SDMVSTRA,
  weights = ~analysis_weight,
  nest = TRUE,
  data = nhanes_cohort
)
```

- до расчётов создать collision audit: проверить mapping `SDMVPSU -> SDMVSTRA`, повторения pair `(SDMVSTRA, SDMVPSU)` между циклами и документацию каждого обнаруженного случая;
- сам факт повторения не разрешает автоматически добавлять cycle; composite identifiers допускаются только после документированного подтверждения истинной коллизии и отдельного amendment;
- для primary pooled 1999–2014 использовать `WTMEC4YR / 4` для участников 1999–2002 и `WTMEC2YR / 8` для циклов 2003–2014;
- для `recent_censored` 2015–2018 использовать `WTMEC2YR / 2` в обоих циклах;
- полный pooled survival estimand 1999–2018 не входит в v1; если он будет одобрен отдельным amendment, его веса составят `WTMEC4YR / 5` для 1999–2002 и `WTMEC2YR / 10` для 2003–2018;
- в таблице cohort audit обязательно показать исходный вес, формулу, множитель, итоговый вес и `weight_estimand` для каждой строки/цикла;
- проверить специальные веса лабораторных subsample-компонентов. Если показатель измерялся только в subsample, нельзя автоматически использовать `WTMEC2YR`.

### Edge-case правила survey design

- primary variance estimator — Taylor linearization; official replicate weights или заранее описанный rescaled survey bootstrap используются только как sensitivity analysis;
- установить `options(survey.lonely.psu = "adjust")` и `options(survey.adjust.domain.lonely = TRUE)`; `average` допускается как sensitivity, а `remove/certainty` запрещены без подтверждения genuinely self-representing PSU;
- `design_df = survey::degf(design) = number_of_PSU - number_of_strata`; при `design_df < 8` precision gate получает `FAIL_INSUFFICIENT_DESIGN_DF`, а результаты могут быть только descriptive/exploratory;
- если `p_hat` равен `0` или `1`, cohort получает `FAIL_NO_OUTCOME_VARIATION`; `R2_CS_max`, DEFF и sample-size gate по основной формуле не вычисляются, а beta interval показывается только описательно;
- `variance_tolerance = 1e-12 * max(p_hat * (1 - p_hat), 1e-12)`;
- non-finite variance или variance ниже `-variance_tolerance` являются hard failure;
- variance в диапазоне `[-variance_tolerance, 0)` может быть заменена на `0` только с флагом `NUMERICAL_CLAMP`; нулевая variance при `0 < p_hat < 1` остается hard failure;
- запрещено автоматически подставлять SRS variance при любой ошибке complex design;
- для subpopulation/domain analysis сохранять полный design object и применять `subset(design, eligibility_flag)`, а не физически удалять строки до создания design.

### Моделирование и оценка

- основной training analysis использует survey weights, нормированные до среднего `1` внутри каждого training fold; preprocessing, weighted loss и calibration fitting не должны видеть test weights или test outcomes;
- основная оценка discrimination, calibration и decision curves должна учитывать соответствующие ненормированные survey weights целевой когорты;
- параллельно сохранить unweighted sensitivity analysis;
- если библиотека модели не поддерживает `sample_weight` для fitting или calibration, такая реализация не может быть primary model без отдельного amendment;
- не объединять survey weights с `class_weight`, SMOTE или случайным undersampling в primary analysis;
- основные доверительные интервалы получать cluster bootstrap с учетом PSU и strata; обычный participant-level bootstrap допускается только как явно названный sensitivity analysis.

## 13. Разделение данных

### NHIS

- `development`: 1997–2009;
- `temporal_test`: 2010–2014;
- `recent_censored`: 2015–2018, только secondary analysis;
- temporal test не используется для feature selection, imputation fitting, calibration или подбора гиперпараметров.

Внутри development использовать `StratifiedGroupKFold`, где `groups=YEAR`, если это обеспечивает достаточное число событий в каждом fold. Если нет, предложить и документировать более устойчивую схему grouped cross-validation.

### NHANES

Гармонизированная NHANES cohort 1999–2014 является внешним test set для NHIS-модели. Она не используется для выбора NHIS-модели или ее порогов.

### Pre-amendment labelled artifact и честный статус blind validation

В текущем рабочем проекте ранее был создан `data/processed/nhanes_cohort.parquet`, содержащий `20 598` individual records, поля `MORTSTAT`, `UCOD_LEADING`, `PERMTH_INT`, `cvd_death_5y` и `225` CVD-событий. Поэтому запрещено утверждать, что NHANES labels никогда технически не присутствовали в окружении или что external validation была sealed from project inception.

Этот файл получает статус `PRE_AMENDMENT_LABELLED_ARTIFACT — PROHIBITED_FOR_V2_3`. До любых новых cohort builds необходимо:

1. вычислить SHA-256 и записать размер, schema, `N = 20 598`, `events = 225`, дату обнаружения и известную историю доступа в `artifacts/audit/access_provenance_manifest.json`;
2. зафиксировать, вычислялись ли до amendment NHANES performance metrics, predictor–outcome associations, subgroup/cycle results или только aggregate `N/events`; неизвестное состояние записывать как `UNKNOWN`, а не как `NO`;
3. логически поместить файл в quarantine: не удалять и не перезаписывать его, но исключить путь и SHA из всех v2.4 configs, loaders и reports;
4. добавить automated blocker, завершающий v2.4 run при попытке прочитать этот path/SHA;
5. заново построить все v2.4 NHANES cohorts из immutable raw source files в новом versioned output path; не переиспользовать старые processed features, imputations, selections, predictions или reports.

Допустимая формулировка: NHANES остается независимой **по источнику данных** external-validation cohort, однако validation не является полностью label-blinded с начала проекта. Можно утверждать только, что NHANES performance, associations и subgroup results не использовались для model selection, если это подтверждено access manifest.

До model lock labels NHIS temporal test и NHANES external test доступны только через `sealed_audit` command. Разрешенный aggregate output:

- общий `N`, aggregate cohort flow и общее число пятилетних CVD-событий;
- для recent cohorts — общие CVD/competing/censored counts и распределение follow-up;
- unstratified predictor missingness и availability без разделения по outcome;
- число linkage-ineligible и квартально-пограничных случаев.

До lock запрещены individual labels, outcome cross-tabs по predictors/subgroups/cycles, associations, outcome-stratified missingness, AUC/Brier/calibration/DCA, соединение predictions с labels и любые решения по модели на основании NHANES outcome. `sealed_audit` сохраняет SHA-256 raw LMF, audit code, config и aggregate output. Разблокировка выполняется только при наличии валидного `external_validation_lock.json`.

До lock также запрещены `outcome_by_year`, `outcome_by_cycle` и held-out outcome drift. Pre-lock plots строятся только для NHIS development 1997–2009. После model lock, но до unseal, разрешены held-out predictor distributions, label-free predictions и covariate-shift diagnostics при условии, что модель больше не изменяется. Outcomes, outcome drift, subgroup/cycle outcome tables и полные temporal plots разрешены только после unseal в Stage 6.

Порядок использования NHANES обязателен:

1. полностью завершить и заморозить NHIS preprocessing, imputation, модель, calibration и threshold policy;
2. сохранить SHA-256 pipeline, конфигурации и кода в `artifacts/minimal_v1/external_validation_lock.json`;
3. один раз получить external predictions и метрики NHANES 1999–2014 без дообучения;
4. записать дату, SHA и результаты в неизменяемый external-validation report;
5. только после этого разрешить доступ к NHANES labels при разработке `clinical_core` и `clinical_hba1c`.

После открытия NHANES labels для clinical analysis последующие изменения NHIS minimal-модели нельзя снова называть независимой внешней валидацией на том же NHANES. Для новой версии потребуется новая внешняя когорта или честное обозначение повторной проверки.

NHANES 2015–2018 получает отдельную роль `external_recent_censored` и не входит в обычную пятилетнюю external test table.

Для клинических NHANES-моделей создать отдельный внутренний grouped split по survey cycle. Результаты этого анализа нельзя называть внешней валидацией.

## 14. Модели

### Зафиксированный predictor-parameter budget

| Модель | Predictor parameters без intercept |
|---|---:|
| `AgeSexLogistic` | 4 |
| `minimal_no_general_health` | 11 |
| `minimal` | 15 |
| `clinical_core` | 20 |
| `clinical_hba1c` | 21 |

Кодирование budget:

- `age`: restricted cubic spline с четырьмя knots — 3 parameters;
- `bmi`: restricted cubic spline с четырьмя knots — 3;
- `sex`: 1;
- `smoking_status`: 2 dummy parameters;
- `hypertension_dx`: 1;
- `diabetes_dx`: 1;
- `general_health`: 4 dummy parameters, а не один произвольный linear score;
- `sbp_mean`: restricted cubic spline с четырьмя knots — 3;
- `total_cholesterol`, `hdl_cholesterol`, `hba1c`: по 1 linear parameter;
- interactions в v1 отсутствуют.

Knots для `age`, `bmi` и `sbp_mean`: weighted percentiles `[0.05, 0.35, 0.65, 0.95]`, вычисленные только внутри соответствующего training fold. External pipeline использует замороженные development knots. Любое добавление nonlinear term, interaction или новой категории увеличивает budget и требует повторного sample-size gate до fitting.

Обязательные модели:

1. `AgeSexLogistic` — контрольный baseline только по возрасту и полу.
2. `LogisticRegression` — основной интерпретируемый baseline по всем быстрым признакам.
3. `RandomForestClassifier` — нелинейное сравнение.
4. `XGBoost` или `LightGBM` — одна boosting-модель, не обе без необходимости.

Для secondary time-to-event analysis дополнительно реализовать:

5. discrete-time competing-risk model по кварталам как основной NHIS recent baseline;
6. cause-specific Cox или другую заранее зафиксированную competing-risk модель с `PERMTH_INT` как основной NHANES recent baseline;
7. Random Survival Forest только как дополнительный нелинейный benchmark, если число событий и используемая библиотека позволяют корректно учитывать competing events.

Нельзя считать несердечно-сосудистую смерть обычным независимым censoring при оценке cumulative incidence. Если выбранный Python-инструмент не поддерживает нужную competing-risk оценку, Codex должен остановиться и предложить проверенную реализацию/ограничение, а не молча заменить задачу обычным Kaplan–Meier.

Для logistic regression возраст и BMI нужно проверить на нелинейность. Предпочтительный вариант — restricted cubic splines или spline transformer внутри pipeline, а не произвольная ручная категоризация.

Deep learning не использовать: объем и структура данных не дают убедительного преимущества, а интерпретируемость и калибровка важнее.

### Дисбаланс классов

- SMOTE не использовать по умолчанию;
- не выполнять случайный undersampling для основной модели;
- class weights допустимы как sensitivity analysis;
- после любого reweighting проверять и при необходимости корректировать calibration;
- основной результат — вероятность риска, а не только бинарный класс.

### Калибровка

- калибровка выполняется только на development data посредством корректной cross-validation или выделенного calibration fold;
- temporal и external test не участвуют в обучении калибратора;
- сравнить sigmoid/Platt calibration с некалиброванной моделью;
- isotonic использовать только при достаточном числе событий и с защитой от переобучения.

## 15. Метрики

### Основные

- ROC-AUC;
- PR-AUC / average precision;
- Brier score;
- log loss;
- calibration intercept;
- calibration slope;
- calibration curve;
- observed/expected event ratio.

### Пороговые

- sensitivity;
- specificity;
- positive predictive value;
- negative predictive value;
- F1 — дополнительная, не главная метрика;
- confusion matrix.

Порог нельзя выбирать на test set. Сначала определить клинически понятную стратегию на development set, например достижение заданной sensitivity, затем зафиксировать порог.

### Не делать accuracy главной метрикой

Сердечно-сосудистая смерть за 5 лет является редким исходом. Высокая accuracy может быть получена моделью, которая почти всегда предсказывает отсутствие события.

### Доверительные интервалы

Для ключевых метрик сформировать 95% CI. Сохранить число событий и не-событий рядом с каждой оценкой.

### Метрики secondary survival analysis

- Harrell C-index или Uno C-index согласно зафиксированному протоколу;
- time-dependent AUC;
- Brier score на заранее выбранных временных точках;
- calibration cumulative incidence;
- число at risk, число CVD events, число competing deaths и число censored;
- 95% CI с учетом survey design настолько, насколько это поддерживается выбранным методом.

Secondary survival metrics должны находиться в отдельных таблицах и не использоваться для улучшения результатов основной пятилетней модели post hoc.

## 16. Интерпретация и fairness audit

### Интерпретация

- для logistic regression: коэффициенты/odds ratios с CI;
- для boosting: SHAP только на финальной зафиксированной модели;
- permutation importance как model-agnostic проверка;
- partial dependence или ALE для возраста и BMI;
- проверка того, что объяснения не используют post-outcome variables.

### Подгруппы

Оценить discrimination и calibration минимум по:

- полу;
- возрасту 40–54, 55–64, 65–79;
- race/ethnicity как audit-only атрибуту после unseal, если категории и число событий позволяют надежную оценку.

`RACEA`, `HISPETH` и harmonized race/ethnicity категорически запрещены как predictors, feature-selection inputs, imputation auxiliaries для model outcome или routing fields. До unseal разрешены только schema/availability и unstratified missingness; outcome-stratified fairness audit выполняется после lock/unseal.

Не делать вывод о «справедливости» только по одинаковой ROC-AUC. Сравнить также calibration, sensitivity и PPV, обязательно с CI.

## 17. Структура репозитория

```text
cvd-risk-model/
├── README.md
├── pyproject.toml
├── uv.lock                         # или другой lock-файл
├── .python-version
├── .R-version
├── renv.lock
├── .gitignore
├── Makefile
├── provenance/
│   └── protocol.sha256
├── configs/
│   ├── feature_mapping.yaml
│   ├── nhanes_files.yaml
│   └── experiment.yaml
├── data/
│   ├── raw/
│   │   ├── nhis/
│   │   └── nhanes/
│   ├── interim/
│   ├── processed/
│   └── quarantine/
│       └── pre_amendment/          # v2.4 loader запрещён
├── src/cvd_risk/
│   ├── __init__.py
│   ├── data/
│   │   ├── download_nhanes.py
│   │   ├── read_lmf.py
│   │   ├── load_nhis.py
│   │   ├── load_nhanes.py
│   │   ├── harmonize.py
│   │   ├── cohort_primary.py
│   │   └── cohort_recent.py
│   ├── features/
│   │   ├── minimal.py
│   │   ├── clinical.py
│   │   └── units.py
│   ├── modeling/
│   │   ├── pipelines.py
│   │   ├── train.py
│   │   ├── calibrate.py
│   │   ├── evaluate.py
│   │   ├── survival.py
│   │   ├── competing_risks.py
│   │   ├── sample_size.py
│   │   ├── survey_precision.py
│   │   ├── routing.py
│   │   └── bootstrap.py
│   ├── reporting/
│   │   ├── tables.py
│   │   ├── figures.py
│   │   └── model_card.py
│   └── schemas.py
├── scripts/
│   ├── 01_download_nhanes.py
│   ├── 02_build_nhis_cohort.py
│   ├── 03_build_nhanes_cohort.py
│   ├── 04_audit_cohorts.py
│   ├── 04a_sealed_audit.py
│   ├── 05_train_minimal.py
│   ├── 06_external_validate.py
│   ├── 07_build_recent_cohorts.py
│   ├── 08_train_extended.py
│   └── r/
│       ├── survey_audit.R
│       ├── development_sample_size.R
│       └── validation_sample_size.R
├── notebooks/
│   └── 01_eda.ipynb
├── tests/
│   ├── test_lmf_parser.py
│   ├── test_harmonization.py
│   ├── test_outcome.py
│   ├── test_cohort.py
│   ├── test_units.py
│   ├── test_routing.py
│   ├── test_sealed_audit.py
│   ├── test_pre_amendment_blocker.py
│   ├── test_survey_precision.py
│   ├── reference/
│   │   └── r_reference_outputs.json
│   └── test_no_leakage.py
├── reports/
│   ├── tables/
│   ├── figures/
│   └── model_card.md
└── artifacts/
    ├── audit/
    │   ├── sample_size_gate.json
    │   ├── sealed_audit_manifest.json
    │   ├── access_provenance_manifest.json
    │   └── quarantine_manifest.json
    ├── minimal_v1/
    │   ├── minimal_model.joblib
    │   └── external_validation_lock.json
    ├── clinical_core_model.joblib
    ├── clinical_hba1c_model.joblib    # только если прошел acceptance criteria
    ├── routing_manifest.json
    ├── input_schema.json
    └── metadata.json
```

Исходные данные не коммитить в Git. Добавить `data/raw`, `data/interim`, `data/processed` и обученные бинарные артефакты в `.gitignore`, если их лицензия или размер не позволяют хранение.

## 18. Этапы реализации и критерии приемки

### Этап 0. Инициализация

Задачи:

- создать структуру репозитория;
- зафиксировать Python и зависимости;
- настроить logging, random seeds и конфиг эксперимента;
- записать обязательные решения amendment 2.4 в `configs/experiment.yaml`;
- до нового build создать access/quarantine manifests для pre-amendment `nhanes_cohort.parquet`, его path и SHA blocker;
- проверить `Rscript --version`; зафиксировать R в `.R-version`, создать `renv.lock` с `survey 4.5`, `pmsampsize 1.1.3`, `pmvalsampsize 0.1.0`;
- создать R smoke tests и frozen reference outputs для survey proportion/CI/DEFF/ESS и sample-size calls;
- создать runtime lock-файлы и сохранить SHA-256 этого MD-протокола в provenance каждой когорты и каждого model artifact;
- создать `data_manifest.yaml` с URL, датой загрузки, размером и SHA-256;
- добавить команды `make test`, `make data`, `make audit`, `make train`.

Критерий приемки: чистая установка Python/R окружений и запуск unit/reference tests одной документированной командой. Если `Rscript` отсутствует, repository scaffold может быть создан, но Stage 0 получает `BLOCKED_RSCRIPT_NOT_FOUND` и не может получить PASS. IPUMS extract не нужен для инфраструктуры, но необходим для фактического выполнения NHIS Stages 1–3.

### Этап 1. Загрузка и парсинг данных

Задачи:

- проверить наличие ручного IPUMS extract;
- проверить contract NHIS 1997–2018: `RACEA` и `HISPETH` обязательны как audit-only поля, но запрещены как predictors;
- автоматически скачать нужные NHANES XPT и mortality `.dat`;
- проверить наличие всех циклов 1999–2018, включая XPT и LMF для `_I` и `_J`, и явно присвоить циклам 2015–2018 secondary role;
- реализовать retry, timeout, checksum и idempotent download;
- читать XPT через `pandas.read_sas` или эквивалент;
- читать mortality `.dat` через `pandas.read_fwf`.

Для NHANES public LMF использовать фиксированные позиции официальной программы CDC:

```text
SEQN          1–6
ELIGSTAT     15–15
MORTSTAT     16–16
UCOD_LEADING 17–19
DIABETES     20–20
HYPERTEN     21–21
PERMTH_INT   43–45
PERMTH_EXM   46–48
```

Критерии приемки:

- `SEQN` уникален в каждом cycle-level mortality-файле;
- merge с `DEMO` имеет ожидаемую cardinality `1:1`;
- нет дублирования участников после merge;
- parser проходит тест на нескольких вручную проверенных строках.

### Этап 2. Гармонизация

Задачи:

- привести признаки к канонической схеме;
- использовать `HUQ010` как primary NHANES `general_health`, а `HSD010` сохранить отдельным sensitivity-полем;
- для 1999–2000 подтвердить `HSD010 = EXPECTED_NOT_AVAILABLE`; отсутствие не является download/schema error;
- отдельно сохранить raw value, canonical value и missing reason;
- создать таблицу доступности признаков по годам/циклам;
- проверить диапазоны и единицы измерения;
- реализовать и протестировать преобразования единиц для общего холестерина, HDL и HbA1c;
- реализовать точные formulas и tolerances из раздела 10 и отдельный тест `sbp_mean` для 1, 2, 3 и 4 доступных readings;
- реализовать отдельные builders `nhis_recent_censored` и `nhanes_recent_censored`, не вызываемые primary binary builder;
- реализовать `followup_time`, `followup_time_unit`, `followup_time_source`, `event_type` и квартальные boundary tests;
- создать отчет о различиях NHIS и NHANES.

Критерий приемки: все семь быстрых признаков имеют одинаковые имена, типы и допустимые категории в двух когортах.

### Этап 3. Построение когорты и обязательный stop-gate

До обучения создать:

- `reports/tables/cohort_flow_nhis.csv`;
- `reports/tables/cohort_flow_nhanes.csv`;
- `reports/tables/event_counts.csv`;
- `reports/tables/missingness_by_source.csv`;
- `reports/tables/feature_availability.csv`;
- `reports/tables/recent_censored_followup.csv`;
- `reports/tables/event_type_counts.csv`;
- `reports/tables/nhanes_weight_audit.csv`;
- `reports/tables/huq_hsd_disagreement.csv`;
- `reports/tables/clinical_availability_by_cycle.csv`;
- `reports/tables/sample_size_gate.csv`;
- `reports/tables/survey_design_precision.csv`;
- `reports/tables/effective_sample_size.csv`;
- `artifacts/audit/sample_size_gate.json`;
- `artifacts/audit/access_provenance_manifest.json`;
- `reports/figures/outcome_by_year_nhis_development.png`;
- `reports/figures/age_distribution_nhis_development.png`.

Stop-gate является жестким `PASS/FAIL`, а не рекомендацией:

1. Для каждой survey cohort создать design object с ненормированными weights, strata и PSU. Для NHIS development рассчитать weighted пятилетнюю CVD event proportion и design-based variance:

```text
p_hat = sum(w_i * y_i) / sum(w_i)
var_design = Taylor-linearized variance либо variance по официальным replicate weights
DEFF_y = var_design / (p_hat * (1 - p_hat) / n_nominal)
DEFF_y_gate = max(1, DEFF_y)
n_eff_design = n_nominal / DEFF_y_gate
n_eff_Kish = (sum(w_i)^2) / sum(w_i^2)
```

`n_eff_design` является primary effective sample size. Kish ESS учитывает неравномерность weights, но не заменяет design-based ESS с clustering/stratification и публикуется только как diagnostic. Сообщать nominal `N`, raw event count, `p_hat`, design-based SE/95% CI, `DEFF_y`, `n_eff_design` и `n_eff_Kish`.

2. Primary 95% CI для overall risk рассчитывать design-based методом, например logit interval через `survey::svyciprop`; обычный Wilson interval по nominal N не является primary. Требование precision: максимальная из двух half-width относительно `p_hat` не больше `0.0025` (эквивалентная максимальная полная целевая ширина `0.005`).

3. При отсутствии сопоставимого externally validated R² вычислить консервативный input:

```text
R2_CS_max = 1 - exp(2 * (p_hat * ln(p_hat) + (1 - p_hat) * ln(1 - p_hat)))
expected_R2_CS = 0.15 * R2_CS_max
shrinkage = 0.90
parameters = 15  # primary minimal
```

4. Запустить `pmsampsize(type="b", csrsquared=expected_R2_CS, parameters=15, shrinkage=0.90, prevalence=p_hat)`. Результат интерпретируется как требуемый IID/SRS-equivalent N. Design-corrected nominal requirement:

```text
required_N_nominal = ceil(pmsampsize_required_N * DEFF_y_gate)
PASS_information = n_eff_design >= pmsampsize_required_N
PASS_risk_precision = max(p_hat - CI_low, CI_high - p_hat) <= 0.0025
PASS_events = raw_CVD_events >= route_event_floor
PASS_design_df = design_df >= 8
```

Route-specific floors:

| Route/роль | Raw CVD event floor |
|---|---:|
| NHIS `minimal` development | 500 |
| NHIS `minimal_no_general_health` development | 500 |
| NHANES `clinical_core` development | 200 |
| NHANES paired `clinical_hba1c` development | 200 |
| NHIS temporal validation | 200 |
| NHANES external validation | 100 |

Для budgets `11`, `20` и `21` расчёт выполняется на фактической analytic cohort соответствующего route. Export разрешён только если одновременно выполнены `PASS_events`, `PASS_information`, `PASS_risk_precision` и `PASS_design_df`. Поэтому известные `225` событий не гарантируют экспорт `clinical_core`, а HbA1c route после complete-case exclusions может остаться exploratory.

5. Stage 3 выполняет для NHIS temporal и NHANES external только preliminary feasibility: nominal N, raw events, design-based prevalence/CI, `DEFF_y` и ESS. Полный calibration-slope validation gate здесь запрещён, потому что distribution linear predictor ещё не существует.
6. Абсолютные floors сохраняются: не менее `200` raw CVD-событий в NHIS temporal test и `100` в NHANES external validation. Они не заменяют Stage 5a/6 precision gate, design ESS или minimum design df.
- для recent cohorts отдельно показать CVD deaths, competing deaths, censored и распределение follow-up; к ним не применяются primary fixed-horizon thresholds;
- для logistic candidate дополнительно должно быть не менее 20 событий на один non-redundant design parameter, считая все spline basis coefficients и dummy variables, но исключая intercept; penalization нельзя использовать для искусственного обхода этого gate;
- для penalized logistic отдельно рассчитать и сообщить `effective_df = trace(H_lambda) - 1`, где вычитается intercept; это report metric, а не замена консервативного design-parameter gate;
- до model lock test labels используются только sealed-audit процессом из раздела 13; никакой individual-level performance или association analysis не разрешен.

Если критерий не выполнен, Codex останавливается после feasibility report. Нельзя молча увеличить горизонт, включить людей с исходным ССЗ или объединить test с train.

### Этап 4. EDA

До model lock EDA выполняется только на NHIS development 1997–2009. Задачи:

- описать NHIS development weighted и unweighted;
- показать prevalence и `outcome_by_year_nhis_development.png` только для NHIS development;
- оценить пропуски;
- выявить implausible values;
- проверить внутри-development drift без доступа к held-out outcomes.

До lock для held-out cohorts разрешены только schema/availability checks, unstratified missingness и outputs sealed audit из раздела 13. Сравнение NHIS 2010–2014 с 2015–2018, NHIS/NHANES predictor distributions, полный temporal drift и любые held-out outcome plots переносятся в Stage 6: predictor-only analyses выполняются после lock, outcome analyses — только после unseal.

EDA-ноутбук не должен содержать единственную рабочую реализацию preprocessing. Вся воспроизводимая логика находится в `src/`.

### Этап 5. Обучение быстрой модели

Задачи:

- единый sklearn-compatible pipeline;
- imputation и encoding обучаются только внутри folds;
- подбор гиперпараметров только на NHIS development;
- сохранить out-of-fold predictions;
- откалибровать модели;
- выбрать финальную модель по совокупности discrimination, calibration, простоты и устойчивости.

Критерий приемки: запуск из пустого `artifacts/` создает один и тот же набор результатов при фиксированном seed в пределах ожидаемой численной погрешности.

### Этап 5a. Provisional external-validation precision planning

После получения NHIS development out-of-fold predictions, но до final model lock, выполнить provisional planning без NHANES labels:

```text
package = pmvalsampsize 0.1.0
type = "b"
cstatistic = 0.75
cstatciwidth = 0.10
cslope = 1
csciwidth = 0.40
oe = 1
oeciwidth = 0.20       # полная ширина CI на шкале O/E ratio, не log(O/E)
simobs = 1_000_000
seed = 20260714
```

Вероятности ограничить только для численной устойчивости: `p_clip = clip(prediction, 1e-6, 1 - 1e-6)`, затем `LP = log(p_clip / (1 - p_clip))`. Рассчитать weighted mean/SD NHIS out-of-fold LP и передать их через `lpnormal = c(weighted_mean_LP, weighted_sd_LP)`. Не использовать temporal/NHANES outcomes для выбора LP distribution.

Повторить расчёт с `simobs = 2_000_000`. Требуемые N должны различаться не более чем на `1%`; иначе использовать большее N и удваивать `simobs` до достижения этого tolerance. Сохранить package lock, все arguments, LP summaries, seed, результаты повторов и итоговый provisional N.

`pmvalsampsize` даёт IID/SRS-equivalent N. Design correction применяется в Stage 6 после появления фактических label-free predictions замороженной модели.

### Этап 6. Временная и внешняя валидация

Задачи:

- создать `external_validation_lock.json` с SHA pipeline, preprocessing, config, code commit и временем фиксации;
- создать label-free predictions замороженной модели для NHIS temporal и NHANES external; после этого любые изменения модели запрещены;
- повторить `pmvalsampsize 0.1.0` с фактическими weighted mean/SD held-out LP и теми же arguments, seed и stability tolerance;
- рассчитать label-free `DEFF_LP` и `DEFF_prediction`, взять `DEFF_gate = max(1, DEFF_y, DEFF_LP, DEFF_prediction)` и проверить `n_nominal / DEFF_gate >= N_pmvalsampsize`; эквивалентный nominal requirement равен `ceil(N_pmvalsampsize * DEFF_gate)`;
- проверить full CI widths: C-statistic `0.10`, calibration slope `0.40`, O/E ratio `0.20` и design-based overall risk `0.005`; `log(O/E)` сообщается дополнительно, но не является sample-size scale;
- при FAIL не изменять модель: validation можно выполнить только как underpowered/exploratory с полными CI и без заявлений о достаточной точности;
- только после валидного lock один раз unseal и оценить final pipeline на NHIS 2010–2014;
- без дообучения применить его к NHANES 1999–2014;
- сохранить external predictions и report до открытия NHANES labels для clinical analysis;
- отдельно показать исходную calibration и recalibration-in-the-large как sensitivity analysis;
- построить subgroup tables;
- построить полный `outcome_by_year.png`, outcome-by-cycle tables и post-unseal temporal drift; отдельно сравнить 2010–2014 и 2015–2018 без присвоения recent cohort полного пятилетнего outcome;
- вычислить CI.

Критерий приемки: внешняя проверка выполняется тем же сериализованным pipeline, который был зафиксирован до external performance analysis; SHA, quarantine старого artifact, access provenance и одноразовый порядок подтверждены артефактами. Нельзя утверждать, что labels отсутствовали в окружении с начала проекта. Только после этого этап 8 получает право использовать rebuilt v2.4 NHANES labels.

### Этап 7. Secondary censored/competing-risk analysis

Задачи:

- сформировать `followup_time` и `event_type` для NHIS/NHANES 2015–2018;
- сохранить единицу и источник времени; для NHIS использовать квартальные интервалы и административный конец `2019-Q4`, для NHANES — `PERMTH_INT` в месяцах;
- применить `WTMEC2YR / 2` к NHANES 2015–2018 и запретить веса полного 1999–2018 estimand;
- оценить обе cause-specific hazards и Aalen–Johansen CIF на `4/8/12 quarters` и `12/24/36 months`;
- показать time-dependent discrimination, Brier/calibration cumulative incidence и числа at risk;
- применить support rule `>=100` CVD events и weighted at-risk proportion `>=0.20` для claims на 24/36 месяцах;
- не использовать secondary results для post-hoc выбора основной пятилетней модели;
- явно описать ограничения короткого follow-up.

Критерий приемки: ни одна запись без полного пятилетнего наблюдения не появляется в primary binary tables как обычный negative case.

### Этап 8. Клинические модели

Задачи:

- построить `clinical_core` с SBP, total cholesterol и HDL;
- построить `clinical_hba1c` как отдельный кандидат, добавляющий HbA1c;
- использовать отдельный NHANES grouped validation protocol;
- сравнить каждую пару моделей на одной и той же аналитической подвыборке;
- показать прирост и его CI, а не только отдельные AUC;
- показать потерю участников и событий из-за каждого дополнительного поля;
- применить все пять acceptance criteria HbA1c из раздела 10;
- не экспортировать `clinical_hba1c`, если хотя бы один обязательный acceptance criterion не выполнен.

### Этап 9. Экспорт модели

Создать:

- сериализованный preprocessing + model pipeline;
- `input_schema.json`;
- `routing_manifest.json` с правилами выбора `minimal`, `clinical_core` и допустимого `clinical_hba1c`;
- диапазоны валидации входов;
- явные единицы и версии функций преобразования;
- версию модели;
- список обучающих данных;
- дату обучения;
- метрики temporal/external validation;
- model card с ограничениями.

## 19. Проверки против утечки данных

Автоматические тесты должны запрещать:

- `MORTSTAT`, `UCOD_LEADING`, время наблюдения и год/квартал смерти в признаках;
- imputation до split;
- encoding до split;
- feature selection на test set;
- calibration на test set;
- подбор порога на test set;
- использование NHANES labels при выборе NHIS-модели;
- чтение pre-amendment `data/processed/nhanes_cohort.parquet`, совпадающего SHA или любых производных v2.2 artifacts в v2.4 pipeline;
- создание held-out `outcome_by_year`, outcome-by-cycle, association или drift outputs до unseal;
- использование nominal Wilson CI как primary survey precision gate;
- добавление `YEAR`/cycle к IPUMS NHIS `STRATA` или `PSU`;
- автоматическое добавление cycle к NHANES `SDMVSTRA`/`SDMVPSU` без collision report и amendment;
- получение PASS survey/sample-size gates без доступного locked R reference backend;
- дубли участников между splits;
- случайное разбиение, игнорирующее временной протокол.
- включение 2015–2018 годов в primary binary cohort как обычных отрицательных случаев;
- использование imputed/guessed clinical values в live routing без отдельно валидированного маршрута.

## 20. Обработка пропусков

- во всех routes сначала различать true missing, refused, don't know, not ascertained и not in universe; missingness report создается до любых exclusions/imputation;
- `minimal` primary analysis: numeric median imputation и заранее утверждённые missing indicators только внутри training fold; categorical `Unknown`/most-frequent policy фиксируется до held-out evaluation; complete-case является sensitivity analysis;
- `clinical_core` exportable primary analysis: complete-case по обязательным `sbp_mean`, total cholesterol и HDL; missing required clinical field исключает record из route, а clinical values не импутируются в primary export model;
- `clinical_hba1c` acceptance primary analysis: paired complete-case cohort с доступными core fields и HbA1c; `clinical_core` comparator и `clinical_hba1c` используют идентичные participants/folds;
- multiple imputation clinical fields выполняется только внутри training folds как sensitivity analysis; route, прошедший acceptance criteria только после imputation, не экспортируется;
- live API: отсутствующий core-набор маршрутизирует запрос к `minimal`; отсутствие HbA1c маршрутизирует к отдельно валидированному `clinical_core`; вымышленные, средние или скрыто imputed пользовательские значения запрещены.

Итог: complete-case является sensitivity для `minimal`, primary для exportable clinical routes и primary paired analysis для проверки инкрементальной пользы HbA1c.

## 21. Что должно попасть в отчет/статью

- диаграмма формирования обеих когорт;
- таблица характеристик NHIS development, NHIS temporal test и NHANES external test;
- точные определения outcome и predictors;
- nominal N, raw events, weighted prevalence, design-based CI, `DEFF`, design-based и Kish effective sample size;
- missingness;
- полное описание split и cross-validation;
- ROC и precision-recall curves;
- calibration plots;
- Brier score и calibration slope/intercept;
- weighted и unweighted sensitivity analyses;
- subgroup performance;
- сравнение `minimal` и клинических моделей;
- отдельное сравнение `clinical_core` и `clinical_hba1c` с анализом инкрементальной пользы HbA1c;
- отдельные результаты recent censored cohort 2015–2018;
- таблица доступности пользовательских полей, единиц и причин переключения между routes;
- model card и ограничения применимости;
- явное limitation: в public-use LMF для отдельных records follow-up time или underlying cause заменены synthetic values, хотя vital status не изменен.
- явное limitation: individual NHANES mortality labels были материализованы до amendment 2.4; сообщить ранее известные агрегаты, quarantine/rebuild и отсутствие использования performance/associations для model selection только в той мере, в какой это подтверждает access manifest;
- не использовать формулировку `sealed from project inception`; корректная формулировка — external by data source и prospectively locked before v2.4 external performance analysis.

Нельзя утверждать, что модель клинически валидирована для Казахстана. Корректная формулировка: исследовательская модель, разработанная и проверенная на репрезентативных данных населения США.

## 22. Что Codex не должен делать

- не использовать старый Statlog Heart dataset как основной;
- не предсказывать наличие болезни вместо будущего риска;
- не объединять NHIS и NHANES без явной роли каждой когорты;
- не делать random 80/20 split основной схемой;
- не использовать accuracy как единственное доказательство качества;
- не применять SMOTE автоматически;
- не подбирать признаки по всему датасету;
- не добавлять признаки, которые пользователь не сможет знать до обследования, в быструю модель;
- не выдавать результат сайта как медицинский диагноз;
- не строить сайт до завершения cohort audit и фиксации модели;
- не скрывать падение размера выборки из-за лабораторных пропусков;
- не называть данные 2015–2018 полноценной пятилетней test cohort;
- не называть прогноз общей вероятностью ССЗ: endpoint — пятилетняя CVD mortality;
- не требовать от посетителя сайта лабораторные показатели для доступа к базовой оценке;
- не просить пользователя угадывать SBP, общий холестерин, HDL или HbA1c;
- не путать HDL/ЛПВП с LDL/ЛПНП и не принимать значение без единицы;
- не включать HbA1c в финальный сайт без отдельно подтвержденной пользы;
- не делать выводы о населении Казахстана по данным США.

## 23. Definition of Done для модели v1

Версия v1 считается готовой, если:

- данные загружаются или обнаруживаются воспроизводимо;
- R reference backend установлен и locked (`survey 4.5`, `pmsampsize 1.1.3`, `pmvalsampsize 0.1.0`), либо Stage 0 честно имеет статус BLOCKED;
- IPUMS NHIS design использует исходные integrated `STRATA`/`PSU` без YEAR, а NHANES — published `SDMVSTRA`/`SDMVPSU` с collision audit;
- lonely PSU, design df, boundary prevalence и invalid variance edge cases покрыты tests/status codes;
- обе когорты строятся из кода;
- primary и recent censored cohorts имеют разные явные роли;
- NHANES primary и recent weights реализованы по estimand, включая `WTMEC4YR` для 1999–2002;
- `HUQ010` является primary general-health source, а `HSD010` проверен отдельно;
- `HSD010` 1999–2000 корректно маркирован `EXPECTED_NOT_AVAILABLE`;
- target и exclusions покрыты unit tests;
- hard sample-size stop-gate пройден или выпущен честный feasibility report без обучения;
- pre-amendment NHANES labelled artifact зарегистрирован, quarantined и заблокирован по path/SHA; v2.4 cohorts построены заново из raw sources;
- access manifest честно фиксирует прежнюю техническую доступность labels и известные aggregate `N/events`;
- после принятия amendment 2.4 held-out outcome access ограничен sealed-audit процессом до model lock;
- development/validation gates используют design-based variance, `DEFF`, ESS, route-specific event floors и locked R sample-size workflow;
- pre-lock EDA содержит outcomes только NHIS development; полный outcome/temporal drift появляется после unseal;
- baseline и три ML-модели обучены без leakage;
- есть calibration и CI;
- проведены temporal и external validation;
- подготовлена сравнительная таблица моделей;
- выбран один финальный быстрый pipeline;
- `clinical_core` экспортирован только после отдельной валидации, а `clinical_hba1c` — только после доказанного прироста;
- missing-data policy route-specific: minimal imputation primary/complete-case sensitivity, clinical routes complete-case primary, MI sensitivity only;
- routing и преобразования лабораторных единиц покрыты unit tests;
- неизвестный `general_health` переводит запрос на `minimal_no_general_health` только при наличии валидированного route; отсутствие любого другого обязательного minimal-признака блокирует персональный расчет;
- неизвестные клинические значения корректно переводят запрос на совместимую модель без вымышленных подстановок;
- создан model card;
- входная схема совместима с будущей формой из восьми полей;
- README позволяет повторить результаты на чистом окружении.

## 24. Первый запрос, который можно передать Codex

```text
Прочитай файл CODEX_CVD_MODEL_IMPLEMENTATION_PLAN.md полностью и используй amendment 2.4 как обязательный протокол проекта. Сначала не обучай модели и не создавай сайт. Выполни этапы 0–3. Проверь `Rscript`; если его нет, создай инфраструктуру, но останови Stage 0 со статусом `BLOCKED_RSCRIPT_NOT_FOUND`. Зафиксируй `.R-version` и `renv.lock` с `survey 4.5`, `pmsampsize 1.1.3`, `pmvalsampsize 0.1.0`, создай R smoke/reference tests. Зарегистрируй и заблокируй pre-amendment `nhanes_cohort.parquet`, затем заново построй v2.4 cohorts только из raw sources. Для IPUMS NHIS используй integrated `PSU`/`STRATA` без YEAR; для NHANES — published `SDMVPSU`/`SDMVSTRA` и collision audit без автоматического добавления cycle. Реализуй lonely-PSU, design-df, boundary-prevalence и invalid-variance rules, route event floors `500/500/200/200`, temporal `200`, external `100`, а также route-specific missing-data policy. Race/ethnicity сохраняй только audit-only и никогда не используй как predictors. Pre-lock EDA выполняй только на NHIS development. Перед продолжением покажи flow, aggregate counts, survey-design audit, nominal/raw/effective N, CI/DEFF/ESS, package locks, reference-test status и hard PASS/FAIL. Не вычисляй held-out performance/associations до model lock и не меняй protocol decisions без подтверждения.
```

После прохождения stop-gate можно дать второй запрос на обучение, калибровку и валидацию моделей.
