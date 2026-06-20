# 5-Day Cadence Plan

If the available dataset is 10 years with one state roughly every 5 days, the model must be reframed.

## What changes?

The old 3-hourly assumption:

```text
past 5 days at 3-hour resolution -> next 24h response
```

does not apply.

The correct 5-day setup is:

```text
past 8 snapshots, about 40 days -> next 5-day radiation/temp response
```

or:

```text
past 12 snapshots, about 60 days -> next 5/10-day response
```

## What the model can learn

At 5-day cadence, it can learn:
- cloud regime persistence
- broad cloud/radiation association
- seasonal/radiation anomaly response
- rain-cloud vs dry-cloud regimes
- slow temperature anomaly response
- multi-week trend patterns

It cannot reliably learn:
- hourly cloud drift
- same-day cloud shading
- cloud formation/dissipation cycles
- immediate afternoon cooling
- exact short-lived precipitation events

## Recommended windows

### Small/default
```text
input_len = 8     # 40 days
horizon = 1       # next 5-day response
```

### More context
```text
input_len = 12    # 60 days
horizon = 1 or 2  # next 5-10 days
```

### Dataset sample count

One location, 10 years, 5-day cadence:

```text
~3650 days / 5 = ~730 snapshots
```

With input_len=8 and horizon=1:

```text
~721 windowed samples per location
```

Ten locations:

```text
~7,200 samples
```

So 30,000 truly separate time-window samples would require roughly:
- more locations,
- more years,
- smaller spatial patches per scene,
- or additional data sources with denser cadence.

## Best target

Use next-5-day averaged targets:

```text
shortwave_anomaly_next_5day
longwave_anomaly_next_5day
net_radiation_anomaly_next_5day
temperature_anomaly_next_5day
```

At 5-day cadence, avoid pretending the model predicts hourly temperature response.

## Strong recommendation

Use trend channels explicitly:

For each core variable, add or precompute:
- current value
- 5-day delta
- 15-day delta
- 30-day rolling mean or slope
- anomaly from same-location seasonal normal

If channel count must stay near 20, prioritize:
- raw cloud/radiation/atmosphere fields first
- add trends by replacing weaker channels or training a v2 model with 24-28 channels
