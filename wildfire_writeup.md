# Predicting Wildfire Occurrence in Los Angeles County from Weather, Vegetation, and Terrain

## Research Question

Can weather conditions, satellite-derived vegetation dryness, and terrain characteristics predict wildfire occurrence in Southern California over a given time horizon?

Rather than explicitly defining wildfire risk using self-selected thresholds, this project uses actual recorded wildfire occurrences as the prediction target. To accomplish this, the models were trained on data describing vegetation moisture, weather, and topography at specific locations and dates where fires have occurred. By using real fire occurrences as a ground truth, rather than a self-chosen threshold, the model has to learn the relationships between physical conditions and what actually happened.

## Data Sources

Four independent, primary data sources were used to create our dataset. Each source was pulled and validated directly rather than relying on pre-cleaned, aggregated data.

**Weather and fire-danger indices — GridMET.** GridMET is a ~4km resolution gridded climate dataset that provides daily maximum temperature, minimum relative humidity, wind speed and direction, vapor pressure deficit, precipitation, potential evapotranspiration, and three operational fire-danger indices (Energy Release Component, and 100-hr/1000-hr dead fuel moisture). A single coordinate call will return the desired values of the grid that this coordinate falls within. Since GridMET represents an interpolated dataset derived from weather station observations, any coordinate call provides the value associated with that specific grid cell rather than a point-specific measurement.

**Vegetation — MODIS NDVI via Google Earth Engine.** MODIS NDVI is raw satellite imagery with a 250m native resolution and a 16-day composite window. As this is not a gridded dataset, we explicitly reduced many individual pixels into a single value per location by taking a spatial mean of a ~2km radius. This buffered spatial mean allowed for a rough match of GridMET's spatial resolution and kept the feature set consistent in what spatial scale each variable represents.

**Fire occurrences — CAL FIRE / FRAP.** CAL FIRE is a real, recorded fire perimeter dataset for LA County (`UNIT_ID = 'LAC'`) from 2000 onward. The data was pulled directly from CAL FIRE's ArcGIS REST API.

**Topography — Open-Elevation API.** Calculated how steep and which direction the land tilts by sampling the location's elevation and four additional 1km offset elevations from that point in the cardinal directions. By first hand-calculating the expected slope and aspect for a test coordinate, there was a baseline to compare the computational calculations against. Slope and aspect were then computed as a discrete approximation of the elevation gradient, in the form of partial derivatives approximated through finite differences. This calculation accounted for the fact that a degree of longitude covers less real distance than a degree of latitude at this latitude, through a cosine correction. The calculations were then combined into a gradient magnitude and a compass-bearing angle.

## Feature Engineering

When it came to choosing the features to build the model around, there was a distinct effort to ground them to the fire behavior triangle. The three parts of this triangle are fuel, weather, and topography. In order to pull features suited to this triangle, we either pulled from the data sources above or calculated them using those same features.

- **Temperature, humidity, wind speed, and vapor pressure deficit (VPD)** are core atmospheric drivers of fire behavior, so it seemed appropriate to include them as features. Higher ambient temperature raises fuel temperature, increasing the likelihood of ignition. Humidity directly affects fuel moisture; drier fuel, resulting from lower humidity, ignites more readily. As for wind speed, it matters in two distinct ways: it supplies additional oxygen that sustains and intensifies combustion, and it can carry embers across distances, spreading fire beyond its original perimeter. VPD reflects how much moisture the atmosphere is pulling from the environment; a high VPD signals dry, drought-prone conditions, one of the more favorable settings for fire to start and spread.
- As said before, drought-prone conditions are a favorable setting for ignition and spread. To additionally account for this, especially in drought-prone Southern California, the inclusion of a **drought term** seemed appropriate. This was calculated using GridMET's precipitation amount variable (`pr`) and the daily mean reference evapotranspiration variable (`pet`). By subtracting `pet` from `pr`, a drought variable was established. Furthermore, droughts are sustained dry periods, so a rolling period of 30 and 90 days was used to represent this properly. This metric allows for an attempt to measure the severity of drought at a given date and location.
- To complement the drought features, a **days since rain** feature was calculated through a cumulative-sum, group-by, and cumulative-count pattern, capturing a running total that only increases on rain days. This provides a visual complement to help see the severity of drought conditions.
- Since the model is capturing ignition within Southern California, the inclusion of a **Santa Ana wind flag**, one of the driving wind forces in the area from September through May, seemed appropriate. This was accomplished by determining whether wind direction and humidity satisfied the conditions of typical Santa Ana behavior (45–90°, northeast-to-east direction, and <20% humidity). These thresholds were grounded in NWS advisories and published Santa Ana climatology rather than arbitrary cutoffs. We also validated the occurrence of these conditions against known seasonality using two independent checks: event rate and raw count by month.
- The inclusion of **month** assisted with seasonality signals. Within Southern California, winds pick up from September–May, but heat picks up from June–August, two conditions that satisfy the fire behavior triangle, so it seemed appropriate to incorporate seasonality to see which condition influences the model more.
- **NDVI** captured live vegetation greenness/fuel conditions through satellite measurement of red and near-infrared light. Healthy, moist vegetation absorbs red light and reflects near-infrared light, producing a high NDVI value. The opposite is true for dying vegetation, which reflects less near-infrared light, resulting in a lower NDVI. This value helps represent fuel/vegetation moisture, which influences rates of ignition.
- **Slope and aspect** were included as features because they represent topography. The ignition location's steepness and orientation, relative to the surrounding terrain, often dictates the spread speed of a fire. Because heat, smoke, and embers rise, a fire spreading uphill will spread more quickly than one spreading downhill. Capturing the steepness and orientation of the terrain can therefore help indicate whether an ignition grew violently enough to be considered a wildfire.

## Methodology

With the above feature engineering, a dataset was designed using `pandas` and `geopandas`. One row represents a single (location, date) pair, with weather and drought features computed using only data available up to ~120 days before that date. This lookback window gives a sufficient amount of time for the 90-day rolling drought feature. Additionally, NDVI is drawn from the most recent available satellite composite before that date. By only allowing for dates before a certain day, this prevents leakage of information the model should not have access to during evaluation, which matters so the model's results reflect genuine learning rather than an inflated result from having seen future information.

For labels and sampling, 373 real, recorded fire events in LA County (2000–2025) formed the positive examples, labeled 1. Each of these examples had a centroid location and a date denoting the fire, called the alarm date. These parameters allowed us to match a location and date to the designed feature dataframe. Additionally, each location from the positive examples was matched to a negative example with a different, non-fire date, labeled 0. This allowed for consistency in location with only variation in time. This positive and negative split produced a balanced dataset of 746 rows.

When it came to training the models, a random 80/20 split was used over a chronological split, since each row is already an independent instance. The two models found to be appropriate for the situation were a logistic regression model and a random forest model. This was due to the nature of the research question: the model seeks to determine whether a fire took place at a location or not. This is a binary, not a continuous, outcome, so these models suit the situation.

## Exploratory Data Analysis

The EDA was approached in two waves: data quality, and feature vs. label.

**Data quality.** The final joined dataset showed zero missing values across all columns, and a clean 373/373 label balance. Additionally, each feature had a physically plausible range, leaving the feature set in a healthy state to examine the relationships within and between features.

**Feature analysis.** Bivariate analysis was approached first. The dataset was split by the 0 and 1 labels, and histogram visualizations were produced for both labels in relation to their features. These histograms provided visuals for the distributions of the features, and within these distributions, clear, coherent trends in relation to fire occurrence appeared. Comparing fire vs. non-fire revealed:

| Feature | Non-Fire (Label = 0) | Fire (Label = 1) |
|---|---|---|
| Drought_90days (mm) | -300.6 | -478.2 |
| days_since_rain | 31.5 | 55.3 |
| tmmx (K) | 296.5 | 303.1 |
| rmin (%) | 28.2 | 19.5 |
| vpd (kPa) | 1.36 | 2.15 |
| ndvi | 0.339 | 0.339 |

There are coherent differences in means between fire and non-fire days. Fire days show substantially drier conditions across 90-day drought measures, nearly double the number of days since rain, and notably higher temperatures. Both relative humidity and vapor pressure deficit also point toward dry, low-moisture environments on fire days. These physical conditions are all consistent with what the fire behavior triangle would predict as ideal conditions for ignition. However, NDVI's raw mean of 0.339 shows virtually no difference between non-fire and fire days. This is likely due to a subtle, non-linear relationship rather than no relationship at all, explored further in the Modeling section.

For seasonality and wind conditions, fire days typically clustered from June to September, accounting for roughly two-thirds of all fires recorded. Non-fire days, by contrast, showed no visible clustering by season. Given typically higher temperatures and drier conditions in Southern California during this window, this clustering is physically coherent. However, it runs against the initial assumption that the Santa Ana wind regime would produce more fire days in this sample. The Santa Ana flag itself shows only a small, directionally correct difference (3.2% of fire days flagged vs. 2.1% of non-fire days), limited by a small number of flagged days overall (20 of 746 rows).

A multivariate feature analysis was also conducted to see how features relate to each other, beyond just their relationship to the label. A single, overall correlation matrix/heatmap (rather than one split by label) was used for this, since redundancy between features is a property of the features themselves, not something that should change depending on the outcome being predicted. Splitting by label would answer a different question than the pruning question this step was meant to address.

The outcome helped inform potential feature pruning due to redundancy between features, but this did not come at the risk of ignoring components of the fire behavior triangle. `tmmx`, `rmin`, and `vpd` showed strong pairwise correlation (up to r = 0.90 between `tmmx` and `vpd`). This is inherently expected, since VPD is mathematically derived from temperature and humidity, and warmer air is able to hold more moisture before becoming saturated, so relative humidity naturally drops as temperature rises. These three were considered for pruning, but they are important representations of the fire behavior triangle this project is based on; dropping any of them would remove a real physical signal, not just redundant noise.

Beyond this weather cluster, `Drought_90days` showed a moderate negative correlation with `tmmx`, `vpd`, and `days_since_rain` (r ≈ -0.60 to -0.67). This direction makes sense: since `Drought_90days` is calculated as accumulated precipitation minus evapotranspiration, more negative values mean more severe drought. Higher temperature and VPD both increase moisture loss, pushing the drought term further negative. This correlation is real but moderate, not extreme, consistent with drought representing accumulated conditions over time rather than the same instantaneous signal as the raw weather variables.

`NDVI` stood apart from the rest of the features, showing consistently low correlation with everything else (its strongest relationship was only r = -0.29 with VPD). This shows NDVI is not redundant with the weather or drought features; it is capturing a genuinely different signal, the vegetation's own physiological response to conditions, rather than the atmospheric conditions themselves. This is part of why NDVI was kept as a feature, even though it showed weak separation between fire and non-fire days in the bivariate section above.

Overall, no features were dropped purely based on correlation. Every strong relationship found in the matrix had a clear physical explanation behind it, rather than pointing to a data-quality issue, and features were kept based on their distinct role in the fire behavior triangle rather than a blanket statistical rule.

## Modeling

Before comparing model results, it is worth defining the metrics used to evaluate them. **Accuracy** is the simplest measure: the percentage of all predictions, across both fire and non-fire cases, that the model got right. Because this dataset is a balanced 50/50 split between fire and non-fire examples, accuracy is a meaningful metric here, rather than a misleading one, as it can be on a naturally imbalanced dataset. **Precision** looks only at the cases where the model predicted "fire," and asks what fraction of those predictions were actually correct; a lower precision means more false alarms. **Recall** looks only at the cases that were actually fires, and asks what fraction the model successfully identified; a lower recall means more real fires were missed. In a wildfire context, recall carries particular practical weight, since a missed fire (a false negative) is generally a more costly error than a false alarm. **F1** combines precision and recall into a single balanced score, useful as a quick summary when both false alarms and missed fires matter.

**Baseline: Logistic Regression.** Logistic regression is well suited to the classification nature of the research question. By utilizing a sigmoid function, this model determines the probability of whether a fire has occurred or not. The model was trained on standardized data; due to the varying units and scales across features, achieving convergence during training was difficult without it, so standardization was necessary for gradient descent to converge properly. After training on an 80/20 split of the feature dataset, the model achieved:

- Accuracy: 0.713
- Precision: 0.716
- Recall: 0.744
- F1: 0.730

**Comparison: Random Forest.** This model weighs feature importance through decision trees. A single tree, using the known labels during training, asks a yes/no threshold question about each feature, one at a time. These questions eventually lead to a conclusion about the label. By combining many trees, each built with randomized subsets of data and features, the model can average out potential fitting mistakes any single tree might make, allowing for sound predictions on new data. This randomized, threshold-based structure is exactly why Random Forest can capture non-linear/interaction effects that logistic regression can't; a tree can naturally learn something like "high drought only matters a lot when wind is also high" by splitting first on drought, then on wind within that branch, whereas logistic regression's single weighted sum has no natural way to represent that kind of conditional relationship. This model achieved:

- Accuracy: 0.747
- Precision: 0.738
- Recall: 0.795
- F1: 0.765

Every metric improved with the Random Forest model. Though modest, this suggests the presence of non-linear or interaction effects that logistic regression cannot capture. The improvement in recall is practically the most important, given the desire for fewer false negatives: Random Forest missed 16 of 78 actual fires, versus 20 of 78 for logistic regression.

**Feature importance.** The most heavily weighted features were `pet`, `vpd`, `tmmx`, and both drought windows, directly consistent with the bivariate findings above. `ndvi` ranked 6th of 18, however, despite showing no separation in its raw mean between labels, suggesting NDVI's relationship to fire occurrence is non-linear or conditional rather than genuinely absent. Contrary to the earlier assumptions about Santa Ana winds, the Santa Ana flag ranked lowest of all features, consistent with the seasonality findings. This is likely a consequence of this sample being dominated by summer fires, leaving the model too little data to learn a strong relationship with Santa Ana wind conditions specifically.

## Limitations

CAL FIRE's labels come with real, stated constraints. The dataset applies minimum size thresholds (timber fires ≥10 acres, brush fires ≥30-50 acres, grass fires ≥300 acres), so small, quickly-contained fires are systematically missing from the label set. Because of this, "fire occurrence" in this project really means occurrence large enough to meet CAL FIRE's reporting criteria, not every ignition that actually happened. CAL FIRE's own documentation also notes known gaps and duplicate or imprecise historical records, so this isn't a perfectly clean ground truth to begin with.

Human ignition is invisible to this feature set. CAL FIRE's `CAUSE` field shows that a substantial share of ignitions are human-caused, things like equipment use or arson, rather than purely weather or vegetation-driven. This field was available but intentionally not used as a feature, since including it would essentially leak the outcome. Because of this, the model can only ever estimate conditions that are favorable for fire spread and occurrence; it has no way of knowing whether a human ignition source was actually present. This puts a real, honest ceiling on how accurate the model could ever be.

The sample also skews toward the summer fire regime. As found in the EDA, the 373 fires in this dataset are disproportionately drawn from the hot, dry summer months rather than the fall/winter Santa Ana-driven regime. This limits how much the model can really say about Santa Ana-driven fires specifically, since there just weren't many of them in this particular sample.

There's also some centroid imprecision worth noting. Fire perimeter centroids were computed without first reprojecting to a projected, meters-based coordinate reference system. For a single fire's spatial extent, this only introduces minor imprecision, but it's a known simplification, not something fully solved.

Lastly, the negative examples were built using a fairly simple randomization approach, sampling the same location on a random date from a different year, rather than being restricted to a specific "calm conditions" window. This is a standard, defensible way to build negative examples, but a more targeted control-selection strategy, like excluding dates within some window of any fire, not just the matched one, could be a real refinement in future work.

## Conclusion

Using physically motivated features built from four independent primary sources, a Random Forest model was able to achieve 74.7% accuracy and 79.5% recall in distinguishing real wildfire-occurrence dates from matched non-fire dates in Los Angeles County (2000–2025). The model leaning most heavily on drought, temperature, and humidity features lines up well with the physical mechanisms these features were originally chosen to represent. That said, the model's performance ceiling is honestly bounded by real, known limitations: a labeling process that excludes small fires, no way to see human-caused ignition, and a sample that currently leans toward the summer fire regime rather than the Santa Ana-driven fall regime.

There are a few directions this project could be extended in: incorporating slope and aspect interactions with wind direction more directly, expanding the fire sample so it better represents the Santa Ana regime, adding a human-ignition proxy feature like distance to nearest road or population density, and eventually moving toward a more physics-informed approach to modeling fire spread itself, rather than just occurrence.
