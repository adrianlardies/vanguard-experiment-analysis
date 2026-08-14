# Vanguard Digital Experiment — A/B Test Analysis

**Collaborative data analytics project by Adrián Lardiés and Irene Sifre.**

This repository is a historical portfolio project analyzing a dataset related to a digital-interface A/B experiment. It does not represent employment by, consulting for, or a client relationship with Vanguard.

## Overview

The project compares behavior recorded for a Test experience (the newer interface) and a Control experience (the existing interface). It follows a five-step digital journey and explores sequence behavior, elapsed timing, event-level errors, client characteristics, and statistical differences between the experiment groups.

The analysis is preserved in its historical form. This maintenance pass improves repository hygiene, execution paths, reproducibility documentation, and the wording of public claims; it does not redesign the study or replace its methodology.

## Data

The raw data consists of three logical sources:

- `df_final_demo.txt`: demographic and account attributes, with one row per client.
- `df_final_experiment_clients.txt`: experiment assignment (`Test`, `Control`, or an unassigned `NA` value) by client.
- `df_pt_1.txt` and `df_pt_2.txt`: digital-process events identified by client, visitor, visit, process step, and timestamp.

The event logs span **15 March 2017 through 20 June 2017**. The repository does not establish that these records describe current clients or current production behavior.

`data/cleaned/vanguard.csv` is the historical analysis dataset consumed directly by Streamlit and referenced by the Tableau artifacts. It contains one row per recorded process event, so client- and visit-level attributes repeat across rows.

## Data Preparation

`src/main.py` loads the four raw files, concatenates both event partitions, removes incomplete demographic rows and duplicate events, and retains the intersection of clients present in demographics, experiment assignment, and process events. It converts selected demographic fields and timestamps, renames columns, merges client attributes with experiment assignment, creates descriptive client segments, and derives journey fields.

The experiment-assignment file contains 20,109 `NA` assignments. Because pandas reads that label as missing, those clients are removed by the historical `dropna()` step before the common-client intersection. This is an exclusion of assignment records/clients—not evidence that exactly 20,000 event observations were “rejected.”

## Journey Definition

The expected sequence is:

```text
start → step_1 → step_2 → step_3 → confirm
```

`lineal=True` is assigned at the **visit** level when all five steps are present in that order, each step occurs once or twice, and the resulting event list contains no other ordering pattern. It is therefore stricter than merely reaching `confirm`, while still allowing one repetition of each step. `lineal=False` includes incomplete visits as well as visits that are out of order or exceed the repetition rule.

Historical “completion rates” are then calculated at the **client** level: a client is counted as lineal if they have at least one lineal visit. Consequently, this metric should be read as a client-level *linear-sequence rate*, not as a conventional visit completion or conversion rate. Abandonment is not modeled as a separate outcome.

## Metrics

- Client-level rate of having at least one visit that meets the linear-sequence rule.
- Event-level error indicator: a one-step backward transition or the third-and-later occurrence of a step within a visit.
- Mean, skewness, and kurtosis of the historical timing field by group, step, and lineality.
- Pearson and Spearman correlations among balance, activity, age, and number of accounts.

Despite its name, `total_time_in_step` is accumulated elapsed time within the visit/repetition grouping, rather than isolated dwell time in each named step. Timing results should be interpreted accordingly.

## Statistical Analysis

The notebook and Streamlit app implement:

- a two-sample proportion z-test for the client-level linear-sequence rates;
- Welch independent-samples t-tests (`equal_var=False`) on the `confirm` timing values, split by lineality;
- two-sided Mann–Whitney U tests on event-level timing values for Test versus Control and lineal versus non-lineal rows;
- a chi-square test and Cramér’s V for association between variation and lineality;
- Pearson and Spearman correlations as descriptive analyses.

Cramér’s V is the only explicit effect-size measure in the historical analysis. The timing tests and chi-square calculation use repeated event rows, so their independence assumptions are limited; statistical significance must not be treated as practical importance or causation.

## Results

The persisted notebook reports a client-level linear-sequence rate of **47.46% for Test** and **46.06% for Control**. Its proportion z-test reports `p = 0.0015` under the historical calculation. The chi-square analysis also reports an association between variation and lineality, but Cramér’s V is **0.0205**, indicating a very weak association.

Descriptive timing and error patterns vary by step and lineality: Test is not uniformly lower across all comparisons. Given the cumulative timing definition, repeated event rows, and rule-based outcome, these findings support a cautious statement that the recorded groups differ on some historical metrics—not a claim that the Test interface caused a general improvement in efficiency or user experience.

## Streamlit

The Streamlit application is an interactive explorer for the prepared dataset and historical analysis:

```bash
streamlit run src/app.py
```

[Historical Streamlit demo](https://vanguard-ab-testing.streamlit.app/) — availability is not guaranteed.

## Additional Artifacts

- `notebooks/notebook.ipynb`: executed historical analysis with retained outputs.
- `slides/Presentation.pdf` and `slides/Presentation.pptx`: presentation artifacts.
- `slides/vanguard.twb` and `slides/Vanguard.twbx`: Tableau workbook artifacts.

## Project Structure

```text
.
├── .streamlit/config.toml
├── data/
│   ├── raw/
│   └── cleaned/vanguard.csv
├── notebooks/notebook.ipynb
├── slides/
├── src/
│   ├── app.py
│   └── main.py
├── README.md
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python src/main.py
```

This rebuilds the 27-column core analysis dataset in memory from the raw files and prints its dimensions; it does not overwrite historical data. The committed 31-column CSV additionally contains `error` and three transformed timing columns used by historical artifacts, so it is retained as a required input for the app.

```bash
streamlit run src/app.py
```

This starts the interactive explorer using `data/cleaned/vanguard.csv`.

## Limitations

- This is a historical portfolio analysis, not evidence of production deployment or work performed for Vanguard.
- The linear journey is a project-specific, rule-based definition and is not equivalent to simply reaching `confirm`.
- Clients without a Test/Control assignment are excluded during preparation.
- Event rows are repeated within clients and visits; several historical tests do not model that dependence.
- Timing fields represent accumulated elapsed time rather than isolated step dwell time.
- Error logic captures only a one-step regression (`step_diff == -1`) or more than two occurrences of a step.
- Persisted notebook outputs document the historical environment but were not regenerated during this maintenance pass.

## Authors

- Adrián Lardiés
- Irene Sifre
