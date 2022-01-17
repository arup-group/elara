# Benchmarks

Following is guidance for using benchmarks in the elara benhmarking module, including how to format the required benchmark data.

We refer to this module as the `elara.benchmarking` module, however as ***benchmarking*** is quite a loaded term, it might be sensible for readers to consider this the ***comparison*** module. Smells like a refactor.

## Overview

The `elara.benchmarking` module is intended to allow automated assurance and validation of MATSim outputs. In practice this involves comparing the outputs of previous elara modules, such as volume counts, to known or 'observed' volume counts (such as might be expected from traffic counter sites). A key requirement is therefore to provide correctly formatted 'observed' data for the comparison to be made. This documentation seeks to assist this process.

A key advantage of using elara for benchmarking is that comparison can be orchestrated by a single configured operation and that we can make use of elara's `factory` module to ensure all required pre-processing is efficiently undertaken as required. For example, if we want to use the `link_speed_comparison` for cars, elara will know to also output simulated link speeds for cars.

The module is intended to provide a summary of individual and combined benchmarks quality. For example, providing Mean Squared Errors. These are commonly referred to as simulation scores, to be used to quickly compare  simulations during calibration, and provide assurance of individual simulations. Ultimately this module is being built to provide a 'loss' or 'value' function to facilitate automated calibration/training.

## Types of Benchmarks

There are several types of comparisons within the module, some traditional "benchmarks" such as flow counters and some less traditional:

- [Trip Based Mode Shares and Counts](#trip-based-mode-shares-and-counts)
- [Plan Based Mode Shares and Counts](#plan-based-mode-shares-and-counts)
- [Trip Distance and Duration Distributions](#trip-distance-and-duration-distributions)
- [Cordons, ScreenLines and Counters](#cordons-screenlines-and-counters)
- [Link Speeds Comparisons](#link-speed-comparisons)
- [Agent Plan Comparisons](#agent-plan-comparisons)
- [Input Plan Comparisons](#input-plan-comparisons)

## Available Benchmarks

The following may not be a complete documentation of all available benchmarks in elara. All available benchmarks can be viewed in `elara.benchmarking.BenchmarkingWorkStation.tools`.

## Configuration

Benchmarks are configured as per other elara modules, except that they typically require or support the passing of additional arguments and options:

```{toml}
[benchmarks]
trip_mode_shares_comparison = {benchmark_data_path = "./example_benchmark_data/test_fixtures/mode_shares.csv"}
link_counter_comparison = {modes = ["car"], benchmark_data_path = "./example_benchmark_data/test_town/test_town_cordon/test_link_counter.json"}
```

Most benchmarks require passing of the observed data as an additional option - `benchmark_data_path`. This data must be formatted as the benchmark handler expects. Examples can be found in `./example_benchmark_data/`.

Benchmark results are output into a special `benchmarks` output sub directory. Many benchmarks additionally create plots as default.

### Naming Benchmarks

The toml config format prevents duplicating keys, such that it is not possible to use the same benchmark handler twice, this is a problem as it is common to want to use the same type of benchmark more than one. For example, for two different traffic counter groups, one for a "central" area and another for an "outer" area:

```{.toml}
[benchmarks]
link_counter_comparison = {modes = ['car'], benchmark_data_path = "./path/to/CENTRAL_cordon.csv"}
link_counter_comparison = {modes = ['car'], benchmark_data_path = "./path/to/OUTER_cordon.csv"}
```

This will throw an error due to the duplicated toml key `duration_comparison`.

Therefore elara allows the use of an additional syntax to name the handlers: `{HANDLER_KEY}--{UNIQUE_IDENTIFIER}`, eg:

```{.toml}
[benchmarks]
link_counter_comparison--central = {modes = ['car'], benchmark_data_path = "./path/to/CENTRAL_cordon.csv"}
link_counter_comparison--outer = {modes = ['car'], benchmark_data_path = "./path/to/OUTER_cordon.csv"}
```

Outputs from the named handlers will be similarly named. An example config using naming is included: `./example_configs/using_benchmarks.toml`

----------------------
# Benchmark Documentation:

----------------------

# Trip Based Mode Shares and Counts

Mode share and count benchmarks.

Can be optionally grouped by agent attribute or filtered by destination activity if such data is available.

## trip_mode_shares_comparison

Compare trip based mode shares.

### Example Configuration

```{.toml}
[benchmarks]
trip_mode_shares_comparison = {benchmark_data_path = PATH_A}
trip_mode_shares_comparison--subpopulations = {benchmark_data_path = PATH_B, groupby_person_attributes = ["subpopulation"]}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `modes` must be set to `modes = ["all"]`, (this is the default)
- `groupby_person_attributes` may be optionally set as a list with a **single** agent attribute key, eg `groupby_person_attributes = ["subpopulation"]`, default is `None`

### Benchmark (Observed) Data Format

csv format as follows:

```{csv}
mode,share
walk,0
bike,0.2
bus,0.4
rail,0
car,0.4
```

Omitted or unrecognised modes will be ignored. If `groupby_person_attributes` is being used, the attribute values should be provided as an additional column `class`:

```{csv}
mode,class,share
bike,None,0.0
bike,poor,0.2
bike,rich,0.0
bus,None,0.0
bus,poor,0.4
bus,rich,0.0
car,None,0.0
car,poor,0.2
car,rich,0.2
pt,None,0.0
pt,poor,0.0
pt,rich,0.0
transit_walk,None,0.0
transit_walk,poor,0.0
transit_walk,rich,0.0
walk,None,0.0
walk,poor,0.0
walk,rich,0.0
```

## trip_mode_counts_comparison

Compare trip based mode counts. As per **trip_mode_shares_comparison** but with trip counts rather than shares.

### Example Configuration

```{.toml}
[benchmarks]
trip_mode_counts_comparison = {benchmark_data_path = PATH}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `modes` must be set to `modes = ["all"]`, (this is the default)
- `groupby_person_attributes` may be optionally set as a list with a **single** agent attribute key, eg `groupby_person_attributes = ["subpopulation"]`, default is `None`

### Benchmark (Observed) Data Format

csv format as follows:

```{csv}
mode,count
bike,20000.0
bus,40000.0
car,40000.0
walk,0.0
```

Omitted or unrecognised modes will be ignored. If `groupby_person_attributes` is being used, the attribute values should be provided as an additional column `class`:

```{csv}
mode,class,count
...
```

## trip_activity_mode_shares_comparison

Compare trip based mode shares. As per **trip_mode_shares_comparison** but additionally filter for only trips ending in certain activities, such as `work` or `education`.

### Example Configuration

```{.toml}
[benchmarks]
trip_activity_mode_shares_comparison = {benchmark_data_path = PATH, destination_activity_filters = ["work", "education"]}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `destination_activity_filters` (required) list of destination activity types to filter
- `modes` must be set to `modes = ["all"]`, (this is the default)
- `groupby_person_attributes` may be optionally set as a list with a **single** agent attribute key, eg `groupby_person_attributes = ["subpopulation"]`, default is `None` 

### Benchmark (Observed) Data Format

csv format as follows:

```{csv}
mode,share
bike,0.01
bus,0.1
car,0.8
walk,0.09
```

Omitted or unrecognised modes will be ignored. If `groupby_person_attributes` is being used, the attribute values should be provided as an additional column `class`:

```{csv}
mode,class,share
...
```

## trip_activity_mode_counts_comparison

Compare trip based mode counts filtered on destination activity. As per **trip_activity_mode_shares_comparison** but for trip counts.

### Example Configuration

```{.toml}
[benchmarks]
trip_activity_mode_counts_comparison = {benchmark_data_path = PATH, destination_activity_filters = ["work", "education"]}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `destination_activity_filters` (required) list of destination activity types to filter
- `modes` must be set to `modes = ["all"]`, (this is the default)
- `groupby_person_attributes` may be optionally set as a list with a **single** agent attribute key, eg `groupby_person_attributes = ["subpopulation"]`, default is `None`

### Benchmark (Observed) Data Format

csv format as follows:

```{csv}
mode,count
bike,20000.0
bus,40000.0
car,40000.0
walk,0.0
```

Omitted or unrecognised modes will be ignored. If `groupby_person_attributes` is being used, the attribute values should be provided as an additional column `class`:

```{csv}
mode,class,count
...
```

-----------------------

# Plan Based Mode Shares and Counts

Where plan based is the dominant mode for the day (based on max distance).

Configuration and data as per the trip based equivalents.

## plan_mode_shares_comparison

## plan_mode_counts_comparison

## plan_activity_mode_shares_comparison

## plan_activity_mode_counts_comparison

-----------------------

# Trip Distance and Duration Distributions

Compare trip distributions (binned)s.

## duration_breakdown_comparison

Compare histogram of trip durations to observed. Observed data may be from a validating data source, such as a survey, or more simply extracted from the input synthetic population using pam.

### Example Configuration

```{.toml}
[benchmarks]
duration_breakdown_comparison = {benchmark_data_path = PATH}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `modes` must be set to `modes = ["all"]`, (this is the default)

### Benchmark (Observed) Data Format

csv formatted as follows:

```{csv}
duration,trips
0 to 5 min,1
5 to 10 min,3
10 to 15 min,0
15 to 30 min,1
30 to 45 min,0
45 to 60 min,4
60 to 90 min,1
90 to 120 min,0
120+ min,0
```

## duration_mode_breakdown_comparison

Compare histogram of trip durations by mode to observed. Observed data may be from a validating data source, such as a survey, or more simply extracted from the input synthetic population using pam.

### Example Configuration

```{.toml}
[benchmarks]
duration_mode_breakdown_comparison = {benchmark_data_path = PATH}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `modes` must be set to `modes = ["all"]`, (this is the default)

### Benchmark (Observed) Data Format

csv formatted as follows:

```{csv}
,mode,duration,trips
0,bike,0 to 5 min,0.0
1,bike,5 to 10 min,0.0
2,bike,10 to 15 min,0.0
3,bike,15 to 30 min,0.0
4,bike,30 to 45 min,0.0
5,bike,45 to 60 min,20000.0
6,bike,60 to 90 min,0.0
7,bike,90 to 120 min,0.0
8,bike,120+ min,0.0
9,bus,0 to 5 min,0.0
10,bus,5 to 10 min,0.0
11,bus,10 to 15 min,0.0
12,bus,15 to 30 min,10000.0
13,bus,30 to 45 min,0.0
14,bus,45 to 60 min,20000.0
15,bus,60 to 90 min,10000.0
16,bus,90 to 120 min,0.0
17,bus,120+ min,0.0
18,car,0 to 5 min,10000.0
19,car,5 to 10 min,30000.0
20,car,10 to 15 min,0.0
21,car,15 to 30 min,0.0
22,car,30 to 45 min,0.0
23,car,45 to 60 min,0.0
24,car,60 to 90 min,0.0
25,car,90 to 120 min,0.0
26,car,120+ min,0.0

```

## duration_d_act_breakdown_comparison

Compare histogram of trip durations by destination activity to observed. Observed data may be from a validating data source, such as a survey, or more simply extracted from the input synthetic population using pam.

### Example Configuration

```{.toml}
[benchmarks]
duration_d_act_breakdown_comparison = {benchmark_data_path = PATH}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `modes` must be set to `modes = ["all"]`, (this is the default)

### Benchmark (Observed) Data Format

csv formatted as follows:

```{csv}
,d_act,duration,trips
0,home,0 to 5 min,0.0
1,home,5 to 10 min,20000.0
2,home,10 to 15 min,0.0
3,home,15 to 30 min,10000.0
4,home,30 to 45 min,0.0
5,home,45 to 60 min,10000.0
6,home,60 to 90 min,10000.0
7,home,90 to 120 min,0.0
8,home,120+ min,0.0
9,work,0 to 5 min,10000.0
10,work,5 to 10 min,10000.0
11,work,10 to 15 min,0.0
12,work,15 to 30 min,0.0
13,work,30 to 45 min,0.0
14,work,45 to 60 min,30000.0
15,work,60 to 90 min,0.0
16,work,90 to 120 min,0.0
17,work,120+ min,0.0
```

## euclidean_distance_breakdown_comparison

Compare histogram of trip distances to observed. Observed data may be from a validating data source, such as a survey, or more simply extracted from the input synthetic population using pam. Trip distances are euclidean rather than routed. Therefore this is not generally intended as a benchmark as destination choice and therefore euclidean distance are not expected to change.

### Example Configuration

```{.toml}
[benchmarks]
euclidean_distance_breakdown_comparison = {benchmark_data_path = PATH}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `modes` must be set to `modes = ["all"]`, (this is the default)

### Benchmark (Observed) Data Format

csv formatted as follows:

```{csv}
euclidean_distance,trips
0 to 1 km,1
1 to 5 km,0
5 to 10 km,0
10 to 25 km,0
25 to 50 km,2
50 to 100 km,1
100 to 200 km,0
200+ km,0

```

## euclidean_distance_mode_breakdown_comparison

Compare histogram of trip distances by mode to observed. Observed data may be from a validating data source, such as a survey, or more simply extracted from the input synthetic population using pam. Trip distances are euclidean rather than routed.

### Example Configuration

```{.toml}
[benchmarks]
euclidean_distance_mode_breakdown_comparison = {benchmark_data_path = PATH}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `modes` must be set to `modes = ["all"]`, (this is the default)

### Benchmark (Observed) Data Format

csv formatted as follows:

```{csv}
,mode,euclidean_distance,trips
0,bike,0 to 1 km,0.0
1,bike,1 to 5 km,0.0
2,bike,5 to 10 km,0.0
3,bike,10 to 25 km,20000.0
4,bike,25 to 50 km,0.0
5,bike,50 to 100 km,0.0
6,bike,100 to 200 km,0.0
7,bike,200+ km,0.0
8,bus,0 to 1 km,0.0
9,bus,1 to 5 km,0.0
10,bus,5 to 10 km,0.0
11,bus,10 to 25 km,40000.0
12,bus,25 to 50 km,0.0
13,bus,50 to 100 km,0.0
14,bus,100 to 200 km,0.0
15,bus,200+ km,0.0
16,car,0 to 1 km,0.0
17,car,1 to 5 km,0.0
18,car,5 to 10 km,20000.0
19,car,10 to 25 km,20000.0
20,car,25 to 50 km,0.0
21,car,50 to 100 km,0.0
22,car,100 to 200 km,0.0
23,car,200+ km,0.0
```

-------------------

# Cordons, ScreenLines and Counters

Traditional counter comparisons for either vehicles or person boardings and alightings.

## link_counter_comparison

Compare hourly observed vehicle volume flows at links.

### Example Configuration

```{.toml}
[benchmarks]
link_counter_comparison--inner_cordon = {modes = ["car"], benchmark_data_path = "PATH_A"}
link_counter_comparison--outer_screen = {modes = ["car"], benchmark_data_path = "PATH_B"}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `modes` must be set to `modes = ["all"]`, (this is the default)

### Benchmark (Observed) Data Format

json formatted as follows:

```{json}
{
  MODE: { . # eg "car"
    NAME: { . # eg "M5_junctionA"
      DIRECTION: {  # eg "inbound"
        "links": [
          "2-3"  # these must match the correct matsim link ids for the counter location*
        ],
        "counts": {
          "0": 0.0,
          "1": 0.0,
          "2": 0.0,
          "3": 0.0,
          "4": 0.0,
          "5": 0.0,
          "6": 0.0,
          "7": 10000.0,
          "8": 0.0,
          "9": 0.0,
          "10": 0.0,
          "11": 0.0,
          "12": 0.0,
          "13": 0.0,
          "14": 0.0,
          "15": 0.0,
          "16": 0.0,
          "17": 0.0,
          "18": 0.0,
          "19": 0.0,
          "20": 0.0,
          "21": 0.0,
          "22": 0.0,
          "23": 0.0
        }
      },
      ...
    }}
}
```

Where `MODE` is typically "car" and `NAME` and `DIRECTION` can be used to label links with real world names, such as "M5_junctionA", "NorthBound".

Link ids must match those in the simulation.

## transit_interaction_comparison

Compare hourly observed person flows onto transit vehicles (boardings and alightings).

### Example Configuration

```{.toml}
[benchmarks]
transit_interaction_comparison = {modes = ["bus", "rail"], benchmark_data_path = "PATH"}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `modes` must be set to `modes = ["all"]`, (this is the default)

### Benchmark (Observed) Data Format

json formatted as follows:

```{json}
{
  MODE: {  # eg "bus"
    NAME: {  # eg "Central stops" or "Bus stop A"
      "boardings": {
        "stop_ids": [  # these must match the relevant matsim schedule stop ids
          "home_stop_out"
        ],
        "counts": {
          "0": 0.0,
          "1": 0.0,
          "2": 0.0,
          "3": 0.0,
          "4": 0.0,
          "5": 0.0,
          "6": 0.0,
          "7": 10000.0,
          "8": 10000.0,
          "9": 0.0,
          "10": 0.0,
          "11": 0.0,
          "12": 0.0,
          "13": 0.0,
          "14": 0.0,
          "15": 0.0,
          "16": 0.0,
          "17": 0.0,
          "18": 0.0,
          "19": 0.0,
          "20": 0.0,
          "21": 0.0,
          "22": 0.0,
          "23": 0.0
        }
      },
      "alightings": {
        "stop_ids": [
          "home_stop_out"
        ],
        "counts": {
          "0": 0.0,
          "1": 0.0,
          "2": 0.0,
          "3": 0.0,
          "4": 0.0,
          "5": 0.0,
          "6": 0.0,
          "7": 10000.0,
          "8": 10000.0,
          "9": 0.0,
          "10": 0.0,
          "11": 0.0,
          "12": 0.0,
          "13": 0.0,
          "14": 0.0,
          "15": 0.0,
          "16": 0.0,
          "17": 0.0,
          "18": 0.0,
          "19": 0.0,
          "20": 0.0,
          "21": 0.0,
          "22": 0.0,
          "23": 0.0
        }
      },
      ...
    }}
}
```

Where `MODE` is a public transit mode eg "bus", and `NAME` can be used to label comparisons with real world names, such as "Bus_stop_A".

Stop ids must match those in the simulation.

-------------------

# Link Speed Comparisons

Compare average hourly link speeds. Link speeds are in meters per second. A common source of this data is google api queries.

## link_vehicle_speeds_comparison

### Example Configuration

```{toml}
link_vehicle_speeds_comparison = {modes = ["car"], benchmark_data_path = "PATH", time_slice=8}
link_vehicle_speeds_comparison--am_freight = {modes = ["car"], benchmark_data_path = "PATH", time_slice=8, groupby_person_attributes=["subpopulation"]}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `modes` (required) list of single mode for which speed data applies, typically `modes = ["car"]`
- `time_slice` (required) hour of day for which average speed data applies. Note that "8" is the 8 till 9 period.
- `groupby_person_attributes` may be optionally set as a list with a **single** agent attribute key, eg `groupby_person_attributes = ["subpopulation"]`, default is `None`

### Benchmark (Observed) Data Format

csv formatted as follows:

```
id,8
1-5,9.5
5-1,9.8
```

or if `groupby_person_attributes` is being used:

```
id,class,8
1-5,freight,9.5
5-1,freight,9.8
```

Note that unrecognised or missing keys will be quietly ignored.

Also note that the csv can contain multiple time slices but that each benchmark will only use one as per the `time_slice` option:

```
id,class,8,17
1-5,freight,9.5,9.2
5-1,freight,9.8,9.3
```

# Agent Plan Comparisons

Compare agent trip durations. Durations are in seconds. Trip durations must belong to specific agent trips (this is ensured using agent unique ids and trip enumeration). This data typically comes from travel diary data. Because this diary data is typically used in the input synthetic population, ie it has not been partitioned - this comparison is not a proper validation.

## trip_durations_comparison

### Example Configuration

```{toml}
[benchmarks]
trip_durations_comparison = {benchmark_data_path = "./PATH.csv"}
trip_durations_comparison = {benchmark_data_path = "./PATH.csv", mode_consistent = true}
```

- `benchmark_data_path` (required) path to observed data for comparison
- `mode_consistent` (optional, {`true`,`false`}, default is `false`) optionally ensure mode consistency such that only trips with the same mode in the simulation and observed data are compared, if set to `true`, the observed data format should additionally include a `mode` column

### Benchmark (Observed) Data Format

csv formatted as follows:

```{csv}
agent,seq,duration_s
chris,1,454.0
chris,2,463.0
nick,1,4.0
nick,2,454.0
other,1,1000
```

Where `agent` is the agent unique identifier (sometimes refered to as `pid`) and `seq` is the trip enumeration (starting at 1). Duration is in seconds.

If `mode_consistency` is set to `true` then a mode column should additionally be included:

```{csv}
agent,seq,mode,duration_s
chris,1,car,454.0
chris,2,car,463.0
nick,1,car,4.0
nick,2,car,454.0
other,1,car,1000
```

Note that unrecognised or missing keys will be quietly ignored. This includes lines for which the mode does not match if `mode_consistency` is set to true.

# Input Plan Comparisons

This is a special class of pseudo-benchmarks which compares simulation input plans to simulation output plans. 

The purpose of these benchmarks is to allow modellers to understand where and how agents have altered their plans as a result of a MATSim simulation.

The currenly supported set of handlers includes:
- Trip start times (seconds since start of simulation)
- Trip durations (seconds)
- Activity start times (seconds since start of simulation)
- Activity durations (seconds)

These handlers have two two requirements: TripLogs and InputTripLogs -- which mirror the trip logs, but are built from the input data rather than output data. To use these handlers you must specify an additional `input_plans` path in your `config.toml` file, eg.

```
[inputs]
plans = "/path/to/plans.xml"
input_plans = "path/to/input_plans.xml"
```

This group of handlers do not support `group_by_person_attributes` or `destination_activity_filters` options, and will ignore these quietly.

## input_plan_comparison_trip_start

## input_plan_comparison_trip_duration

## input_plan_comparison_activity_start

## input_plan_comparison_activity_duration

### Example Configuration

```
[benchmarks]
input_plan_comparison_trip_duration = ["all"]
input_plan_comparison_trip_start = ["all"]
input_plan_comparison_activity_start = ["all"]
input_plan_comparison_activity_duration = ["all"]
```

- `benchmark_data_path` is not required and will be overwritten.
- `modes` should be `["all"]`

### Benchmark Data (Observed) Data Format
Data is generated by Elara -- there is no need for users to generate it.
