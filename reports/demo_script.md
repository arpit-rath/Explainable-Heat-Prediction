# Demonstration script (about 6–8 minutes)

*Open the Streamlit app (local: `streamlit run app.py`, or the public URL once deployed) and the
notebook in Colab in another tab.*

---

**1. The problem (30 s)**
"Chennai's hottest weeks are April to June, when daily maxima average about 36 °C, and the city is also
humid. Knowing a day ahead that tomorrow will be unusually hot helps people plan. I built a model that
predicts tomorrow's maximum temperature, gives the chance that it will be unusually hot *for Chennai at
that time of year*, and explains why. It is a student project, not an IMD forecast."

**2. Data (45 s), on the *Data, method & limitations* tab**
"The data come from NASA's POWER service: 45 years of daily weather for the Chennai grid cell, from the
MERRA-2 reanalysis. This is the exact request URL, so anyone can download the same data. I checked it:
16,436 days, no gaps, no duplicates, no missing-value codes, nothing physically impossible. It is a
50-kilometre grid average, so it is smoother than a thermometer at Nungambakkam. And it arrives two to
three days late, which the app always shows."

**3. Next-day prediction (60 s), on the *Next-day prediction* tab**
- Point to the info box: "The latest complete day is [date], so this predicts [date + 1]. Because of the
  delay, that day may already have passed."
- Point to the predicted value, the threshold for that date, the chance and the flag: "The flag goes on
  at a 35 % chance or more. That cut-off was chosen on 2013–2018 data, before testing."
- Point to the chart: "Grey is what the model saw. The blue diamond is its prediction."

**4. Explanation (60 s), on *Why this prediction?***
"The starting point is what the model predicts on a normal day for this date. Each bar shows how today's
departure from normal moved the prediction. For a linear model this split is exact: the bars add up to
the prediction, and the details table proves it. These are patterns learned from data, not causes."

**5. Historical backtest (90 s), on the *Historical backtest* tab**
- Choose *Hottest test-period day* (3 May 2019): "The actual was 41.4 °C. The model said 39.9 °C, an
  under-prediction of 1.5 °C. But it gave a 47 % chance of an unusually hot day, so the flag was on."
- Choose *Largest test-period miss*: "This is the model's biggest error on the test years: [read the
  date and values]. Sudden changes such as rain arriving or a wind shift are where a one-day statistical
  model fails."
- Pick a 1990s date: "Note the warning. The deployed model was trained on 1981–2018, so this date is
  in-sample. A fair check uses 2019 or later."
- Say: "The prediction for any date uses only data up to the day before. The data are cut off before
  any calculation, and an automated test confirms it."

**6. Performance (90 s), on the *Model performance* tab**
"On seven held-out years, 2019–2025, the model's average error is 0.77 °C. 'Tomorrow equals today' gives
0.85 and the seasonal average 1.20, so it is about 10 % better than the best simple rule. SVR was
slightly more accurate, but my rule, set in advance, picks the simpler model when the gap is under
0.05 °C, because then every prediction can be explained exactly."

"Here is the honest part. On days that turned out unusually hot, all the models under-predict by about
a degree, and my heat flag catches 44 % of them. That is better than my first version, which caught
17 %, but it still does not beat 'tomorrow equals today' on the F1 score. The chances themselves are
reliable: days given 30–50 % (38 % on average) were hot 35 % of the time."

**7. Climate context and limitations (60 s), back on the *Data* tab**
"Over 45 years, daytime maximums in this grid cell show almost no trend, while night-time minimums and
humidity have risen. Warmer, more humid nights still matter for heat stress. This is one reanalysis
cell, not a formal trend study, and the model cannot say climate change caused any particular hot day."

**8. Close (20 s)**
"So the contribution is a Chennai-specific seasonal heat threshold, a calibrated chance of unusual heat,
exact plain-language explanations and an honest interactive backtest. The notebook reproduces
everything from the raw download."

---

### Likely questions and short answers

* **Why not a random train/test split?** Neighbouring days are nearly identical, and a random split
  mixes future years into training, so it overstates accuracy. The notebook shows the random-split
  number for contrast only.
* **Why Linear Regression if SVR was best?** The difference was 0.02–0.03 °C, under the pre-set
  tolerance, and the linear model's explanations are exact rather than approximate.
* **Did you tune on the test set?** No hyperparameter or cut-off was chosen on test data. The test
  period was scored twice (design v1 and v2), the v2 changes were justified on validation data, and both
  results are reported.
* **Why does humidity sometimes lower the prediction?** Inputs such as temperature and humidity are
  correlated, and each effect is conditional on the others. The model describes associations, not
  physics.
* **Is this an official heat-wave warning?** No. IMD uses station data and its own criteria.
