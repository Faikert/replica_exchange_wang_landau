# План будущей оптимизации Wang–Landau

## Цель

Уменьшить время и пиковую память финальной relaxed-union агрегации, stitching и MPI-сбора,
не меняя физику, RNG-последовательность, научные форматы и результаты sampling.

Relaxed-union выполняется после основного Monte Carlo sampling и почти не влияет на flips/s.
Исключения — расчёт WL coverage по check interval и биннинг предложенного состояния.

## Текущая стоимость

Плотный `DosFragment` хранит на одну ячейку:

- `log_g`: 8 байт;
- histogram: 8 байт;
- standard error: 8 байт;
- valid: 1 байт;
- contributors: 4 байта;
- support component: 4 байта.

Итого приблизительно 33 байта на ячейку без учёта allocator overhead.

Внутриоконное выравнивание `M` walkers по `C` ячейкам сейчас имеет сложность
`O(M^2*C)`. Плотная MPI-передача walker estimators требует примерно 17 байт на глобальную
ячейку каждого walker, а передача готовых fragments лидерами — примерно 33 байта на
глобальную ячейку каждого окна.

## Этап 1. Малорисковые копирования и WL checks

### 1.1. Убрать копирование fragments перед stitching

Вместо `std::vector<DosFragment>` использовать отсортированный массив
`const DosFragment*` или индексов. Сортировать ссылки, не плотные массивы.

Ожидаемый эффект:

- примерно вдвое меньшая временная память stitching;
- устранение полного memory-bandwidth прохода по всем fragments.

### 1.2. Перемещать временные результаты

- В 1D wrapper переносить массивы из временного `JointDensityOfStates` через
  `std::move`.
- Не создавать полную копию joint DOS при marginalization.
- Ввести read-only view на общие массивы для `marginalize_fragment` и
  `marginalize`.

### 1.3. Исключить двойной расчёт histogram statistics

Основной цикл уже вычисляет `HistogramStatistics`, после чего
`ready_for_iteration()` повторно сканирует active cells.

Изменение:

- передавать готовую статистику в readiness-проверку;
- при `nalivaiko_mod=false` проверять inverse-time coverage за `O(1)` через размеры
  cumulative active-list и iteration-list;
- при `nalivaiko_mod=true` сохранять current-iteration coverage, которое после первого
  посещения имеет значение один;
- полный проход сохранять только для traditional flatness и диагностического вывода.

### 1.4. Упростить биннинг proposal

После проверки конечности и границ нормированная координата неотрицательна, поэтому
преобразование в `std::size_t` эквивалентно `floor`. Убрать `std::floor` из hot path,
сохранив специальную обработку точной верхней границы.

До принятия изменения сравнить fixed-seed последовательность побитово.

## Этап 2. Active-cell aggregation

### 2.1. Открыть read-only active-cell list

Добавить внутренний accessor к накопленному `active_cells_`. Формат checkpoint менять
не требуется: список продолжает восстанавливаться из active mask.

### 2.2. Заменить плотный парный overlap

Рассмотреть два варианта:

1. Для каждой пары walkers обходить меньший active-list и проверять active mask второго.
2. Одним проходом по union support строить список contributors ячейки и накапливать все
   парные weighted-LS коэффициенты.

Предпочтителен второй вариант. Его ожидаемая сложность:

`O(sum(A_m) + sum(k_cell^2))`,

где `A_m` — число active cells walker, а `k_cell` — число contributors ячейки.

### 2.3. Однопроходные mean и SEM

После определения shifts использовать Welford update для одновременного вычисления:

- aligned mean;
- `M2`;
- histogram sum;
- contributors;
- support component.

Это удалит второй проход по walkers для каждой valid-ячейки и сохранит численную
устойчивость.

### 2.4. Valid-cell lists

Сохранять список valid indices в оконном fragment и итоговой DOS. Использовать его для:

- sparse CSV writers;
- metadata statistics;
- marginalization;
- построения межоконных overlap.

Dense scientific arrays пока оставить для совместимости.

## Этап 3. Window-local fragments

Перевести `DosFragment` с глобального layout на локальный диапазон:

`(window.end-window.begin)*Q_bins`.

Добавить явные преобразования:

- global `(e,q)` → local fragment index;
- local index → global flat index.

Ожидаемый эффект особенно велик при большом числе окон: память fragments перестанет
масштабироваться как `windows*global_cells`.

Необходимо обновить:

- local и MPI summary;
- stitching;
- leader gather offsets;
- adaptive statistics interfaces;
- writers и analyzer;
- synthetic MPI tests с `cells != energy_bins`.

Форматы CSV можно сохранить без изменения, поскольку они уже содержат глобальные
`e_bin` и `q_bin`.

## Этап 4. Sparse MPI payload

### 4.1. Walker → window leader

Передавать только:

- active cell indices;
- соответствующие `log_g`;
- histogram.

Использовать chunked variable gather с отдельным обменом размеров и проверками
переполнения.

### 4.2. Window leader → global root

Передавать только valid cells готового fragment:

- global или window-local index;
- `log_g`;
- histogram;
- SEM;
- contributors;
- support component.

Root заполняет итоговые dense arrays один раз перед научным анализом.

### 4.3. Убрать промежуточные root buffers

Текущий root сначала получает плоские `all_*` массивы, затем копирует их в отдельные
`DosFragment`. После sparse gather данные должны сразу устанавливаться в целевые fragments
или в общий fragment storage.

## Этап 5. Stitching solver и lookup

### 5.1. Быстрый component lookup

Заменить линейные `find_if` при построении и использовании node lookup на:

- плотный vector, если component IDs компактны;
- `unordered_map<int32_t, size_t>` для внешних/legacy fragments.

`support_component_count` также перевести с линейного поиска уникальных значений на set
или bitmap.

### 5.2. Graph Laplacian solver

Weighted-LS система является закреплённым графовым лапласианом. После профилирования
заменить Gauss–Jordan на один из вариантов:

- Cholesky/LDLT для небольших плотных компонент;
- conjugate gradient для больших разреженных графов.

Для типичных 2–16 walkers это имеет низкий приоритет; основная экономия должна сначала
прийти от устранения плотных cell scans.

### 5.3. Унифицировать два LS solver

Внутриоконный и межоконный solver содержат близкую реализацию. Вынести общий внутренний
graph-alignment helper, чтобы оптимизации и численные проверки применялись одинаково.

## Этап 6. I/O и analyzer

### 6.1. Sparse writers по valid-cell list

Не сканировать полный dense layout при записи sparse CSV. Итерировать непосредственно по
valid indices.

### 6.2. Буферизованный CSV output

Для больших joint DOS рассмотреть:

- увеличенный stream buffer;
- накопление блоков строк;
- `std::to_chars` вместо множества formatted stream operations.

Формат и точность чисел должны остаться прежними либо изменение формата должно быть
отдельно задокументировано и протестировано.

### 6.3. Быстрый `wl_analyze` parser

Заменить `stringstream + vector<string>` на ручное разделение фиксированных колонок и
`from_chars`. Сохранить чтение старых fragments без contributors/components.

## Benchmark и критерии приёмки

Текущий `wl_bench` измеряет физический incremental flip, но не покрывает relaxed union,
WL checks, stitching или MPI serialization. Добавить режимы:

- `walker-1d` и `walker-joint`;
- `coverage` для разных active support и check interval;
- `aggregate-dense` и `aggregate-sparse`;
- `stitch` с прямым overlap, цепочкой и несколькими компонентами;
- `mpi-summary` и `mpi-leader-gather`.

Для каждого режима сообщать:

- wall time;
- processed/valid cells per second;
- переданные MPI bytes;
- peak resident memory;
- число временных allocations;
- параметры `E_bins`, `Q_bins`, walkers, windows и support density.

Обязательные проверки после каждого этапа:

- fixed-seed sampling sequence не изменилась;
- log_g, histogram, spins, energy, Q, counters и RNG state совпадают;
- weighted shifts, contributors, SEM и component IDs совпадают с эталоном;
- disconnected-support policy и exit codes не изменились;
- serial, warning build, MPI tests и ASan/UBSan проходят;
- старые CSV по-прежнему читаются `wl_analyze`.

## Рекомендуемый порядок

1. Удалить копии fragments и временных DOS.
2. Устранить двойной histogram scan и сделать inverse-time coverage `O(1)`.
3. Реализовать active-cell relaxed-union и Welford SEM.
4. Добавить valid-cell lists.
5. Перейти на window-local fragments.
6. Реализовать sparse MPI payload.
7. Оптимизировать component lookup и LS solver.
8. Оптимизировать CSV writer и analyzer.
