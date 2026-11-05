# NIB7143CEM: Sri Lanka men's international cricket, ball by ball

This repository is the coursework for NIB7143CEM Programming for Data Science. It contains `cricket_project`, a Python package that loads, validates, cleans, stores and analyses 904 Sri Lanka men's international matches (659,890 deliveries) from Cricsheet, plus the commands that reproduce every table and figure in the report.

**Question:** for Sri Lanka, is playing at home, winning the toss or batting first associated with winning? Does that differ by format? How does scoring tempo differ by phase of the innings?

## 1. Environment used for the reported numbers

| | |
|---|---|
| OS | Windows 11 (10.0.26200) |
| CPU | AMD Ryzen 7 PRO 4750U, 8 cores / 16 threads |
| RAM | 29.7 GB |
| Disk | NVMe SSD |
| Python | 3.14.7 |
| Java (PySpark only) | 25.0.4 LTS |
| Libraries | exact versions in [requirements.txt](requirements.txt) |

`python main.py benchmark` also writes the machine and library details to `outputs/tables/environment.json`. Timings depend on the machine, so they are not comparable across computers.

## 2. Reproduce from a clean checkout

```bash
git clone https://github.com/DininduSahan/NIB7143CEM-Programming-for-Data-Science.git
cd NIB7143CEM-Programming-for-Data-Science

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
python -m pip install -r requirements.txt

python -m pytest                     # unit tests (about 10 s)
python main.py checksums --verify    # Obtain: raw extract unchanged
python main.py pipeline              # Task 3: Obtain, Scrub, Explore, Model (about 30 s)
python main.py benchmark             # Task 2: storage formats (about 10-15 min)
python main.py spark                 # Task 4: pandas vs PySpark (needs Java 17+, about 10 min)
```

Run `pipeline` before `benchmark` and `spark`, because both read the cleaned Parquet file that `pipeline` writes. To get quicker, rougher numbers, use `python main.py benchmark --repeats 3` or `python main.py spark --scales 1 4 --repeats 3`. The reported numbers use the defaults: 6 runs, the first discarded as a warm-up, and the median reported with the min to max range.

If PySpark on Windows prints `Did not find winutils.exe`, you can ignore it. The Spark code only reads files.

## 3. What each command produces

| Report section | Command | Output |
|---|---|---|
| 3. Storage formats (Task 2) | `benchmark` | `outputs/tables/storage_benchmark.csv`, `environment.json`, `outputs/figures/fig4_storage_formats.png` |
| 4. Obtain | `checksums --verify` | verifies `Data/raw_checksums.sha256` (SHA-256 of all 906 raw files) |
| 4. Scrub | `pipeline` | `scrub_report.csv` (problem, rows and matches affected, action, reason per rule), `scrub_summary.csv`, `missing_values_before_after.csv`, `load_failures.csv` |
| 4. Explore | `pipeline` | `first_innings_totals.csv`, `phase_run_rates.csv`, `outcomes.csv`, `fig1_first_innings_totals.png`, `fig2_phase_run_rates.png` |
| 4. Model | `pipeline` | `win_rates_home_away.csv`, `win_rates_home_batting_first.csv`, `independence_tests.csv`, `fig3_factor_win_rates.png` |
| 5. Scale (Task 4, route A) | `spark` | `spark_vs_pandas.csv`, `spark_details.json`, `spark_physical_plan.txt`, `fig5_spark_vs_pandas.png` |

Tables are written to `outputs/tables/` and figures to `outputs/figures/`. The cleaned dataset goes to `outputs/data/deliveries_clean.parquet`. That file is git-ignored, so rebuild it with `pipeline`.

## 4. Data

* **Source:** Cricsheet, "Sri Lanka men" JSON archive, <https://cricsheet.org/downloads/sri_lanka_male_json.zip>. The format is described at <https://cricsheet.org/format/json/>.
* **Licence:** Open Data Commons Attribution License (ODC-By 1.0). See `Data/Raw/LICENSE.txt`. The data is published by Cricsheet and is not scraped.
* **Extract:** 904 matches (Tests, ODIs, T20Is) from 2002 to 22 Oct 2026, JSON format versions 1.2.0 and 1.3.0. It was committed to this repository on 2026-11-03.
* **Immutability:** nothing under `Data/Raw/` is ever modified. The pipeline checks every file against `Data/raw_checksums.sha256` before loading and stops with `RawDataIntegrityError` if any file has changed. You can run the same check with `sha256sum -c` on Linux and macOS.

## 5. Package design (Task 1)

```
cricket_project/
  exceptions.py     CricketDataError and its subclasses (domain failures, distinct from ValueError)
  data_loader.py    MatchFile (one validated match, read-only) and MatchLoader (directory -> DataFrame)
  cleaning.py       CleaningRule ABC + 10 rules (template method), Scrubber (all-or-nothing runner)
  storage.py        StorageFormat ABC + CsvFormat / JsonFormat / PickleFormat / ParquetFormat
  benchmark.py      time_callable, FormatBenchmark
  checksums.py      SHA-256 manifest write and verify
  privacy.py        pseudonymise(): drop player-name columns, keep registry ids
  analysis.py       match summary, win rates, chi-square tests, phase run rates
  figures.py        report figures
  spark_compare.py  Task 4: the same stage in PySpark, timed against pandas
  pipeline.py       command-line entry points used by main.py
tests/              pytest suite (no real data needed)
```

How the package is used from Python:

```python
from cricket_project import MatchLoader, Scrubber, default_rules, get_format

raw = MatchLoader("Data/Raw").load_frame()
clean, reports = Scrubber(default_rules()).run(raw)
get_format("parquet", "zstd").write(clean, "deliveries.parquet")
```

## 6. Personal data

Cricsheet records the names of players and match officials, and names are personal data under UK GDPR and EU GDPR. The analysis here works at team, format and venue level, so `pipeline` drops every name column (`batter`, `bowler`, `non_striker`, `player_out`) before writing any derived file. It keeps only Cricsheet's registry identifiers (`batter_id`, `bowler_id`). Every committed table and figure is an aggregate. The raw files are the unmodified, openly licensed public extract.

## 7. Running the tests

```bash
python -m pytest -v
```

The tests build small synthetic matches in temporary folders, so they do not depend on the real dataset.
