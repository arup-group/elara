# Elara

A command line utility for processing (in big batches or bit by bit) MATSim outputs.

MATSim model runs output a variety of `xml` and `xml.gz` formatted data. Some of this data is unchanged from what was input to MATSim (for example in the case of `output_network.xml.gz`). But two outputs are new or changed: `output_plans.xml.gz` and `output_events.xml.gz`. Together these outputs contain the information available about what happened within model simulation iterations.

Elara allows processing of these outputs into more easilly useable forms and formats. For example extracting hourly flows of vehicles for all links in a network (`volume_counts`). Elara outputs are typically some form of aggregation of simulation outputs. Elara outputs are typically made available in tabular (`csv`) and spatial (`geojson`) formats. Spatial representations are converted to EPSG:4326, which works in kepler.

## Inputs

Inputs to Elara are MATSim format output files, eg:

* **events** = "./tests/test_fixtures/output_events.xml"
* **network** = "./tests/test_fixtures/output_network.xml"
* **transit_schedule** = "./tests/test_fixtures/output_transitSchedule.xml"
* **transit_vehicles** = "./tests/test_fixtures/output_transitVehicles.xml"
* **attributes** = "./tests/test_fixtures/output_personAttributes.xml"
* **plans** = "./tests/test_fixtures/output_plans.xml"
* **output_config_path** = "./tests/test_fixtures/output_config.xml"
* **road pricing config** = "./tests/test_fixtures/roadpricing.xml"

Depending on what outputs you require from elara, some of these inputs may not be required, but it is often conveneient to have them all available as a default. In most cases these Elara inputs may be zipped (`xml.gz`).

If you are running a scenario with road pricing, it will be needed to calculate toll log outputs.

### Outputs

Elara supports and can be selectively configured to output a growing number of outputs. The units responsible
for each output are referred to as 'handlers'. There are four main types of output/handler:

* **Event Based Handlers/Outputs**:
These are processed by streaming (in order) through all output events from simulation.
  * ``link_vehicle_counts``: Produce link volume counts and volume capacity ratios by time slice. Counts **vehicles** entering link (PT vehicles counted once).
  * ``link_passenger_counts``: Produce link passenger counts by time slice. Counts **agents** entering link.
  * ``link_vehicle_speeds``: Produce average vehicle speeds across link.
  * ``route_passenger_counts``: (WIP) Produce vehicle occupancies by transit routes.
  * ``stop_passenger_interactions``: Boardings and Alightings by time slice.
  * ``stop_to_stop_passenger_counts``: Passenger counts between directly connected stops/stations.
  * ``stop_passenger_waiting``: Agent waiting times for unique pt interaction events.
  * ``vehicle_departure_log``: Vehicle departures and delays from facilities (stops in the case of PT).
  * ``vehicle_passenger_graph``: Experimental support for building interaction graph objects (networkx).

* **Plan Based Handlers/Outputs**:
These are processed by streaming through all output plans from simulation. Compared to the event based outputs
these are typically more aggregate but can be computationally faster and can be used to expose agent plan
'memories' and plan scoring.
  * ``mode_shares``: Produce global modeshare of final plans using a mode hierarchy.
  * ``trip_destination_mode_share``: Produce global modeshare to specified destination activities from final plans using a mode hierarchy.
  * ``trip_logs``: Produce agent activity logs and trip logs for all selected plans. Ommitts pt interactions and individual trip legs. Trip mode is based on maximum leg distance.
  * ``leg_logs``: Produce agent activity logs and leg logs for all selected plans.
  * ``plan_logs``: Produce agent plans including unselected plans and scores.
  * ``agent_highway_distance_logs``: Produce agent distances by car on different road 
  types. Requires network to have `osm:way:highways` attribute.
  * ``trip_highway_distance_logs``: Produce flat output of agent trip distances by car on different road types. Requires network to have `osm:way:highways` attribute.
  * ``toll_logs``: Produces summary of amounts agents paid at tolls, depending on the route they drove. Requires road pricing input file. Only works for option ``["car"]``.

* **Post Processing Handlers**:
These are outputs produced through additional post-processing of the above outputs.
  * ``vkt``: Produce link volume vehicle kms by time slice.
  * ``plan_summary``: Produce leg and activity time and duration summaries (png).
  * ``trip_duration_breakdown``: Produce binned trip durations.
  * ``trip_euclid_distance_breakdown``: Produce binned trip distances.

* **Benchmarking Handlers**:
Where correctly formatted project specific observed data has been made available, Elara can also assist with validation or 'benchmarking'.
 Elara will compare and present simulation results from the above outputs to the available benchmarks, it will aditionally output a
 distance based score for the model. Where distance is some measure of how different the simulation is from the observed data.

## Contents

* [Introduction](https://github.com/arup-group/elara#introduction)
* [Installation](https://github.com/arup-group/elara#installation)
* [Configuration](https://github.com/arup-group/elara#configuration)
* [Advanced Configuration](https://github.com/arup-group/elara#advanced-configuration)
* [Command Line Reference](https://github.com/arup-group/elara#command-line-reference)
* [Example CLI Commands](https://github.com/arup-group/elara#example-cli-commands)
* [Tests](https://github.com/arup-group/elara#tests)
* [Debug](https://github.com/arup-group/elara#debug)
* [About](https://github.com/arup-group/elara#about)
* [Adding Features](https://github.com/arup-group/elara#adding-features)
* [Todo](https://github.com/arup-group/elara#todo)
* [What does the name mean?](https://github.com/arup-group/elara#what-does-the-name-mean)

## Introduction

Elara uses a Command Line Interface (CLI), within this interface you can choose to run elara 
via a [config](https://github.com/arup-group/elara#configuration) or purely through the 
[CLI](https://github.com/arup-group/elara#command-line-reference). The CLI is preferred for 
producing single outputs, for example to extract vehicle kms, whereas using
a config can handle big batches of outputs in one go. So if you are after a quick output - best 
to use the CLI, whereas if you need a collection of outputs and/or reproducibility, a config is preferred.

The CLI and configs share naming conventions and structure so we recommend reading about both below.

Also note that elara is often used within simulation pipelines 
[ie BitSim](https://github.com/arup-group/bitsim). In which case it is typically run using a config. 

## Installation

Clone or download this repository. Once available locally, navigate to the folder and create a virtual environment to
install the libraries in. Assuming Python >=3.7 and using git:
```
git clone git@github.com:arup-group/elara.git
cd elara
python3 -m venv venv (or equivalent in conda for example) #intall into virtual environment
source venv/bin/activate (or equivalent in conda for example)
pip3 install -e .
elara --help
```

We strongly recommend using a virtual environment.

Note that we require pyproj=='2.4.0' because older version have proven to be very slow/hang for converting 
between coordinate reference systems.

**Note for Windows Users:**
If installation fails with a traceback indicating `geos_c.dll` cannot be found, this is usually a problem with installing `shapely` and `fiona` requirements.

One workaround is is the following (assumes you are using an Anaconda or Miniconda distribution):
- Activate a conda environment
- Install `geopandas` first using `conda install geopandas -c conda-forge`. (Geopandas requires both `shapely` and `fiona`, and conda may be able to install them without issue.)
- Remove `geopandas`, `shapely`, and `fiona` from the `requirements.txt` file in a text editor (nb: please don't commit these changes when pushing new code).
- Install the rest of the dependencies as normal: `pip install -e .`

## Configuration

For reproducibility or to process a range of outputs, configuration is the most sensible and 
processing efficient way to use Elara.

Configuration is accessed via the CLI:

`elara run <CONFIG PATH>`

Config files must be `.toml` format and be roughly formatted as follows. The various fields are 
detailed further below and an [example config](https://github.com/arup-group/elara/blob/master/example_config.toml) is also included in the repo:

```{.toml}
[scenario]
name = "test_town"
time_periods = 24
scale_factor = 0.01
version = 11
crs = "EPSG:27700"
verbose = true

[inputs]
events = "./tests/test_fixtures/output_events.xml"
network = "./tests/test_fixtures/output_network.xml"
transit_schedule = "./tests/test_fixtures/output_transitSchedule.xml"
transit_vehicles = "./tests/test_fixtures/output_transitVehicles.xml"
attributes = "./tests/test_fixtures/output_personAttributes.xml"
plans= "./tests/test_fixtures/output_plans.xml"
output_config_path = "./tests/test_fixtures/output_config.xml"
road_pricing = "./tests/test_fixtures/roadpricing.xml"

[outputs]
path = "./tests/test_outputs"

[event_handlers]
link_vehicle_counts = ["car", "bus"]
link_passenger_counts = ["bus"]
stop_passenger_counts = ["bus"]
stop_passenger_waiting = ["all"]

[plan_handlers]
mode_shares = ["all"]
trip_destination_mode_share = {destination_activity_filters = ["work"]}
trip_logs = ["all"]
agent_highway_distance_logs = ["car"]
trip_highway_distance_logs = ["car"]
toll_logs = ["car']

[post_processors]
vkt = ["car"]
plan_summary = ["all"]
trip_duration_breakdown = ["all"]
trip_euclid_distance_breakdown = ["all"]

[benchmarks]
test_pt_interaction_counter = ["bus"]
test_link_cordon = ["car"]
test_duration_comparison = ["all"]
test_euclidean_distance_comparison = ["all"]

```

You can run this config on some toy data: `elara run example_config.toml` (from the project root).

**#** scenario.**name** *string* *(required)*

The name of the scenario being processed.

**#** scenario.**time_periods** *integer* *(required)*

The number of time slices used to split a 24-hour period for the purposes of reporting. A value 
of ``24`` will produce summary metrics for each our of the day. Similarly, a value of ``96`` will
 produce 15-minute summaries.

**#** scenario.**scale_factor** *float* *(required)*

The sample size used in the originating MATSim scenario run. This is used to scale metrics such 
as volume counts. For example, if the underlying scenario was run with a 25% sample size, a value
 of ``0.25`` in this field will ensure that all calculated volume counts are scaled by 4 times.

 **#** scenario.**version** *int {11,12}* *(default 11)*

Set `version = 12` if using MATSim version 12 outputs (in which case there will be no output personAttributes file).

**#** scenario.**crs** *string* *(required)*

The EPSG code specifying which coordinate projection system the MATSim scenario inputs used. This
 is used to convert the results to WGS 84 as required.

**#** scenario.**verbose** *string* *(required)*

Logging module [level](https://docs.python.org/3/library/logging.html#levels), for example, either ERROR, WARNING, INFO or DEBUG.

**#** inputs.**events** *file*

Path to the MATSim events XML file. Can be absolute or relative to the invocation location.

**#** inputs.**network** *file*

Path to the MATSim network XML file. Can be absolute or relative to the invocation location.

**#** inputs.**etc** *file*

Path to additional MATSim resources.

**#** outputs.**path** *directory* *(required)*

Desired output directory. Can be absolute or relative to the invocation location. If the 
directory does not exist it will be created.

**#** outputs.**contract** *boolean*

If set to *true*, removes rows containing only zero values from the generated output files. 

**#** event_handlers.**[handler name]** *list of strings* *(optional)*

Specification of the event handlers to be run during processing. Currently available handlers 
include:

  * ``link_vehicle_counts``: Produce link volume counts and volume capacity ratios by time slice.
  * ``link_passenger_counts``: Produce vehicle occupancy by time slice.
  * ``link_vehicle_speeds``: Produce average vehicle speeds across link.
  * ``route_passenger_counts``: (WIP) Produce vehicle occupancies by transit routes.
  * ``stop_passenger_interactions``: Boardings and Alightings by time slice.
  * ``stop_to_stop_passenger_counts``: Passenger counts between directly connected stops/stations.
  * ``stop_passenger_waiting``: Agent waiting times for unique pt interaction events.
  * ``vehicle_departure_log``: Vehicle departure times and delays from facilities (stops in the case of PT).
  * ``vehicle_passenger_graph``: Experimental support for building interaction graph objects (networkx).

The associated list attached to each handler allows specification of which network modes should be processed using that handler. This allows certain handlers to be activated 
for public transport modes but not private vehicles for example. Possible modes currently include:

* eg ``car, bus, train, tram, ferry, ...``.
* note that ``waiting_times`` only supports the option of ``["all"]``.
* note that ``vehicle_departure_log`` supports any valid mode options, but will produce empty csv outputs for modes without events of type ``VehicleDepartsAtFacility``.

The above format of `HANDLER_NAME = ["car", "bus"]` is a shorthand way of passing options. These pptions can also be passed in a dictionary format, ie `HANDLER_NAMES = {modes=["car","bus"]}`.

**#** plan_handlers.**[handler name]** *list of strings* *(optional)*

Specification of the plan handlers to be run during processing. Currently available handlers 
include:

  * ``mode_shares``: Produce global modeshare of final plans using a mode hierarchy.
  * ``trip_destination_mode_share``: Produce global modeshare to specified destination activities from final plans using a mode hierarchy. Requires ``destination_activity_filters`` - a list of destination activities as a parameter, eg: ``trip_destination_mode_share = {destination_activity_filters = ["work"]``. 
  * ``trip_logs``: Produce agent activity logs and trip logs for all selected plans. Ommitts pt interactions and individual trip legs. Trip mode is based on maximum leg distance.
  * ``leg_logs``: Produce agent activity logs and leg logs for all selected plans.
  * ``plan_logs``: Produce agent plans including unselected plans and scores.
  * ``agent_highway_distance_logs``: Produce agent distances by car on different road 
  types. Requires network to have `osm:way:highways` attribute.
  * ``trip_highway_distance_logs``: Produce flat output of agent trip distances by car on different road types. Requires network to have `osm:way:highways` attribute.
  * ``toll_logs`` : Produces summaries of tolls paid by agents. Requires ``roadpricing.xml`` as input parameter. 

The associated list attached to each handler allows specification of additional options:

* in most cases ``all``
* agent_plans support subpopulation selection, eg ``rich, poor``
* highway_distances only supports ``car``

The mode share handler as well as the trip destination mode share handler additionally support a `subpopulation` option, eg: `mode_shares = {attribute="subpopulation"}`. This adds an additional output breaking down mode counts and shares by the chosen person attribute.

**#** post_processors.**[post-processor name]** *list of strings* *(optional)*

Specification of the event handlers to be run post processing. Currently available handlers include:

  * ``vkt``: Produce link volume vehicle kms by time slice.
  * ``plan_summary``: Produce leg and activity time and duration summaries (png).
  * ``trip_duration_breakdown``: Produce binned trip durations.
  * ``trip_euclid_distance_breakdown``: Produce binned trip distances.

The associated list attached to each handler allows specification of which modes of transport 
should be processed using that handler. This allows certain handlers to be activated for public 
transport modes but not private vehicles for example. Possible modes currently include:

* eg ``car, bus, train, ...``
* note that ``plan_summary`` only support the option of ``["all"]``.

## Advanced Configuration

### Additional Options

The above configurations typically pass a modes option, eg:

```{.toml}
[event_handlers]
link_vehicle_counts = ["car", "bus"]
```

This is equivalent to passing a dictionary with the `modes` option as a key, eg:

```{.toml}
[event_handlers]
link_vehicle_counts = {modes=["car", "bus"]}
```

This more verbose method allows more options to be passed to handlers. Allowing them to have more complex functionality.

Most handlers support a groupby operation that groups outputs such as counts based on person attribute. This can be enabled by passing the `groupby_person_attributes` option as follows:

```{.toml}
[event_handlers]
link_vehicle_counts = {modes = ["car","bus"], groupby_person_attributes = ["income", "age"]}
```

The above config will produce a number of link counts for car and bus with breakdowns by the person attributes named 'income' and 'age'. Note that cross tabulations are not supported and that specifying an unavailable attribute key will result in a warning and outputs being assigned to the 'None' group.

An example config using the `groupby_person_attributes` option is included: `./example_config_complex_options.toml`.

### Letting Elara Deal With Dependancies

The `modes` and `groupby_person_attributes` options are understood by elara as dependancies. Therefore if you ask for a postprocessor such as VKT (or a benchmark - described later below), elara will make sure that any handlers required to produce these outputs will also be created and given the correct options.

This means that a config can be very minimal. As an example, the following configurations are equivalent:

```{.toml}
...
[event_handlers]
link_vehicle_counts = ["bus"]

[post_processors]
vkt = {modes=["car"], groupby_person_attributes=["age"]}
...
```

```{.toml}
...
[event_handlers]
link_vehicle_counts = {modes=["car","bus"], groupby_person_attributes=["age"]}

[post_processors]
vkt = {modes=["car"], groupby_person_attributes=["age"]}
...
```

Care should be taken not to specify unnecesary options as they can add significant compute time. An example configuration that passes complex dependancies is included: `./example_configcomplex_dependancies.toml`

### Benchmarks

Benchmarks provide functionality to compare standard elara outputs to observed data. Benchmarks require passing of the observed data as an additional option. This data must be formatted as the benchmark handler expects. Examples can be found in `elara.benchmark_data`.

Benchmarks are added to the config as follows:

**#** benchmarks.**[benchmarks name]** *list of strings* *(optional)*

Specification of the benchmarks to be run. These include a variety of highway counters, 
cordons and mode share benchmarks for specific projects. **Benchmarks calculated using preprocessed
 data unique to the given scenario network. This means that a given benchmark will not work for a
  given scenario (say 'London') unless the same network and/or schedule are in use.** Where a 
  network or schedule has been changed, the project [bench](https://github.com/arup-group/bench) 
  has been created to pre-process this data. 
  'Normalised' volume plots (e.g those produced by ``ireland_highways``) refer to total volumes normalised by number of counters, so figure shows profile for the 'average' counter.

Currently available benchmarks include:

_newer formats (produced using `bench`):_

* ``ireland_highways``
* ``ireland_highways_NI``
* ``ireland_DCC``
* ``ROI_modeshares``
* ``london_board_alight_subway``
* ``london_central_cordon_car``
* ``london_central_cordon_bus``
* ``london_inner_cordon_car``
* ``london_inner_cordon_bus``
* ``london_boundary_cordon_car``
* ``london_boundary_cordon_bus``
* ``london_thames_screen_car``
* ``london_thames_screen_bus``
* ``test_pt_interaction_counter``
* ``test_link_cordon``

^ note that ``ROI_modeshares`` and other mode share benchmarks only support the option of ``["all"]``.

_older formats (no longer maintained):_

* ``london_inner_cordon_car``
* ``dublin_canal_cordon_car``
* ``ireland_commuter_modeshare``
* ``test_town_highways``
* ``squeeze_town_highways``
* ``test_town_cordon``
* ``test_town_peak_cordon``
* ``test_town_modeshare``

The associated list attached to each handler allows specification of which modes of transport 
should be processed using that handler. This allows certain handlers to be activated for public 
transport modes but not private vehicles for example. Possible modes currently include:

* eg ``car, bus, train, subway...``

However please note that benchmarks are often mode specific and should be configured as such, eg:

* ``london_central_cordon_car = ["car"]``

### Naming Handlers

The toml config format prevents duplicating keys, such that it is not possible to use the same handler twice, eg:

```{.toml}
[benchmarks]
duration_comparison = {benchmark_data_path = "./benchmark_data/test_fixtures/trip_duration_breakdown_all.csv"}
duration_comparison = {benchmark_data_path = "./benchmark_data/test_fixtures/trip_duration_breakdown_all_ALTERNATE.csv"}
```

Will throw an error due to the duplicated key `duration_comparison`.

Instead Elara allows us to use an additional syntax to name the handlers: `{HANDLER_KEY}--{UNIQUE_IDENTIFIER}`, eg:

```{.toml}
[benchmarks]
duration_comparison = {benchmark_data_path = "./benchmark_data/test_fixtures/trip_duration_breakdown_all.csv"}
duration_comparison--ALTERNATE = {benchmark_data_path = "./benchmark_data/test_fixtures/trip_duration_breakdown_all_ALTERNATE.csv"}
```

Outputs from the named handlers will be similalry named. An example config using naming is included: `./example_config_using_named_keys.toml`.

## Command Line Reference

Elara can also be more generally used via the CLI to process individual outputs. USed in this manner the CLI should be pretty discoverable, once installed, try the command `elara` in your terminal to find out about the available options:

```
$ elara
Usage: elara [OPTIONS] COMMAND [ARGS]...

  Command line tool for processing a MATSim scenario events output.

Options:
  --help  Show this message and exit.

Commands:
  benchmarks       Access benchmarks output group.
  event-handlers   Access event handler output group.
  plan-handlers    Access plan handler output group.
  post-processors  Access post processing output group.
  run              Run Elara using a config.toml, examples are included in...

```
Further commands can then be explored. In general the commands and options available follow the 
same structure as configuration, described in the next section:

```
$ elara event-handlers volume-counts --help

Usage: elara event-handlers volume-counts [OPTIONS] MODES...

  Create a volume counts output for a given mode or modes. Example
  invocation for "car" and "bus" modes with name "test" and scale factor at
  20% is:

  $ elara event-handlers volume-counts car bus -n test -s .2

Options:
  -f, --full                  Option to disable output contracting.
  -e, --epsg TEXT             EPSG string, defaults to 'EPSG:27700' (UK).
  -s, --scale_factor FLOAT    Scale factor, defaults to 0.1 (10%).
  -v, --version INT           MATSim version {11,12}, defaults to 11.
  -p, --time_periods INTEGER  Time period breakdown, defaults to 24 (hourly.
  -o, --outputs_path PATH     Outputs path, defaults to './elara_out'.
  -i, --inputs_path PATH      Inputs path location, defaults to current root.
  -n, --name TEXT             Scenario name, defaults to root dir name.
  -d, --debug                 Switch on debug verbosity.
  --help                      Show this message and exit.
```

**Note that defaults will not always be suitable for your scenario. `-s` `--scale_factor` in 
particular, should be updated accordingly.**

Similarly, note that the CLI assumes inputs will have standard (at the time of writing) MATSim 
names, ie `output_plans.xml.gz`, `output_personAttributes.xml.gz`, `output_config.xml` and so on.

## Example CLI Commands

Produce **vehicle kilometres travelled by car (VKT)** for a London scenario, all the defaults are correct but you'd like to read the 
inputs from a different location to your current path:

`elara post-processors vkt car -inputs_path ~/DIFFERENT/DATA/LOCATION`

or, more succinctly:

`elara post-processors vkt car -i ~/DIFFERENT/DATA/LOCATION`

Produce **volume counts for cars and buses** using a New Zealand projection (2113). The scenario was 
a 1% sample. You'd like to prefix the outputs as 'nz_test' in a new directory '~/Data/nz_test':

`elara event-handlers volume-counts car bus -epsg EPSG:2113 -scale_factor .01 -name nz_test 
-outputs_path ~/Data/nz_test`

or, much more succinctly:

`elara event-handlers link-vehicle-counts car bus -e EPSG:2113 -s .01 -n nz_test -o ~/Data/nz_test`

Produce a **benchmark**, in this case we assume that a benchmark has already been created 
called `ireland-highways` and that it works for buses and cars.

`elara benchmarks ireland-highways car bus -e "ESPG:2157"`

Note that we are assuming that all other option defaults are correct. ie:
- --scale_factor = 0.1
- --time_periods = 24
- etc

## Tests

### Run the tests (from the elara root dir)

    python -m pytest -vv tests

### Generate a code coverage report

To generate XML & HTML coverage reports to `reports/coverage`:

    ./scripts/code-coverage.sh
    
## Debug

Logging level can be set in the config or via the cli, or otherwise defaults to False (INFO). We 
currently support the following levels: DEBUG, INFO, WARNING.

Note that the configured or default logging level can be overwritten to debug using an env variable:

    export ELARA_LOGGINGLEVEL='True'
    
## About

Elara is designed to undertake arbitrarily complex pipelines of output processing, postprocessing
 and benchmarking as defined by a configuration file or via the CLI.
 
Elara defines a graph of connected `WorkStations`, each responsible for building certain types of
 output or intermediate data requirements (`Tools`). These workstations are connected to their 
 respective dependees and dependencies (managers and suppliers) to form a **DAG**.
 
 ![dag](images/dag.png)
 
Elara uses this **DAG** to provide:

* **Minimal** dependency processing
* **Early validation** of all intermediate requirements 
* **Early Failure**/Ordering

Elara does this by traversing the DAG in three stages:

1) Longest path search to ensure correct order.

2) Breadth first initiation of `.tools` as `.resources` and supplier validation.

3) Reverse Breadth first build all tools in `.resources` and build workstation.

## Adding Features

Elara is designed to be extendable, primarily with new tools such as handlers or benchmarks. 

Where new tools are new classes that implement some process that fits within the implementation 
(both in terms of code and also abstractly) of a WorkStation. New tools must be added to the 
relevant workstations 'roster' of tools (ie `.tools`).

New tool classes must correctly inherit and implement a number of values and methods so that they 
play well with their respective workstation:
 
* ``.__init__(self, config, mode)``: Used for early validation of mode and 
subsequent requirements.
* ``.build(self, resource=None)``: Used to process required output (assumes required resources are 
available).

`tool_templates` provides some templates and notes for adding new tools.
 
 It may also be required to add or connect new workstations, where new workstations are new types
  of process that might contain a new or multiple new tools. New workstations must be defined and
   connected to their respective suppliers and managers in `main`.

### Conventions

Where possible please name new handlers based on the following:

`<Where><What>`

For example: `LinkPassengerCounts` or `StopPassengerCounts`.

## Todo

* More descriptive generated column headers in handler results. Right now column headers are 
simply numbers mapped to the particular time slice during the modelled day. 
* Try and move away from test_data towards unit tests.
* More outputs, ie mode distances/times/animations.
* S3 integration (read and write).
* automatic discovery of inputs in directory.

**Is it fast/sensible?**

 Sometimes - 
 
 ![dag](images/design.png)

## What does the name mean?
[Elara]("https://en.wikipedia.org/wiki/Elara_(moon)") is a moon of Jupiter. There's not much else
 interesting to say about it, other than that the wikipedia article states that it orbits 11-13 
 *gigametres* from the planet. Gigametres is a cool unit of measurement. 
