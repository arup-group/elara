# Elara

A command line utility for processing (in big batches or bit by bit) MATSim XML outputs (events or plans files) into useful tabular and geospatial outputs for analysis.

## Contents

* [Introduction](https://github.com/arup-group/elara#introduction)
* [Installation](https://github.com/arup-group/elara#installation)
* [Inputs](https://github.com/arup-group/elara#inputs)
* [Output Handlers](https://github.com/arup-group/elara#output-handlers)
* [Running from a Configuration File](https://github.com/arup-group/elara#running-from-a-configuration-file)
* [Command Line Reference](https://github.com/arup-group/elara#command-line-reference)
* [Configuration Options](https://github.com/arup-group/elara#configuration-options)
* [Tests](https://github.com/arup-group/elara#tests)
* [Debug](https://github.com/arup-group/elara#debug)
* [About](https://github.com/arup-group/elara#about)
* [Adding Features](https://github.com/arup-group/elara#adding-features)
* [Todo](https://github.com/arup-group/elara#todo)
* [What does the name mean?](https://github.com/arup-group/elara#what-does-the-name-mean)

## Introduction

Elara allows processing of complex MATSim outputs into more easilly useable forms and formats. For example extracting hourly flows of vehicles for all links in a network (`link_vehicle_counts`). Elara outputs are typically some form of aggregation of simulation outputs. Elara outputs are typically made available in tabular (`csv`) and spatial (`geojson`) formats. Spatial representations are converted to EPSG:4326, which works in kepler.

Once installed Elara is intended to be used as a Command Line Interface (CLI), within this interface you can choose to run elara via a [config](https://github.com/arup-group/elara#configuration) or purely through the [CLI](https://github.com/arup-group/elara#command-line-reference) using command line arguments. The CLI is preferred for producing single outputs, for example to extract vehicle kms, whereas using
a config can handle big batches of outputs in one go. So if you are after a quick output - best to use the CLI, whereas if you need a collection of outputs and/or reproducibility, a config is preferred.

The CLI and configs share naming conventions and structure. But we recommend learning about configs first as this provides an overview of most available tools.

## Installation

Assuming Python >=3.7, virtualenv and using git, clone or download this repository.

```{sh}
git clone git@github.com:arup-group/elara.git
```

Once available locally, navigate to the folder to install Elara and its dependencies. Using a virtual environment is highly recommended.

### OSX

```{sh}
cd elara
virtualenv -p python3.7 venv
source venv/bin/activate
pip3 install -e .
pytest
elara --help
```

### Windows

It is recommended to use an Anaconda environment for installation on Windows:

```{sh}
conda create -n elara python=3.7
conda activate elara
conda install geopandas
git clone git@github.com:arup-group/elara.git
cd elara
pip install -e .
pytest
elara --help
```

On Windows, installation may fail due to failure to locate underlying geospatial dependencies (with a traceback indicating `geos_c.dll` cannot be found, or similar). In this case, it is recommended to first install `geopandas` from conda-forge.

```{sh}
conda install geopandas -c conda-forge
```

After removing `geopandas`, `fiona`, and `shapely` from the `requirements.txt` file, install Elara as per above.

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
* **vehicles** = "./tests/test_fixtures/output_vehicles.xml"

Depending on what outputs you require from elara, some of these inputs may not be required, but it is often convenient to have them all available as a default. In most cases these Elara inputs may be compressed using gzip (`file.xml.gz`).

The road pricing config is required when using the (deprecated) ``toll_log`` plan handler, but *not* when calcuating toll logs using the `agent_tolls_log` event handler.

## Output Handlers

Elara supports and can be selectively configured to output a growing number of outputs. The units responsible
for each output are often referred to as 'handlers', although `elara.factory` describes them more generally as `Tools`.

There are four main types of handler/tools, arranged into python modules:

* elara.event_handlers
* elara.plan_handlers
* elara.postprocessing
* elara.benchmarking

Within each module a `WorkStation` is responsible for orchestrating all that modules tools. The `elara.factory` module is then responsible for orchestratinig all the `Workstations` together.

### **Event Handlers/WorkStation Tools**

*Users are encouraged to add new tools as they require. The below lists are updated periodically but may not be complete.*

These are processed by streaming (in order) through all output events from simulation.

Currently supported handlers include:

* ``link_vehicle_counts``: Produce link volume counts and volume capacity ratios by time slice. Counts **vehicles** entering link (PT vehicles counted once).
* ``link_vehicle_capacity``: Produce link capacity counts by time slice for understanding PT crowding. Sums the vehicle capacities of all **vehicles** entering link for the particular PT mode.
* ``link_passenger_counts``: Produce link passenger counts by time slice. Counts **agents** entering link.
* ``link_vehicle_speeds``: Produce average vehicle speeds across link (in kilometers per hour).
* ``route_passenger_counts``: (WIP) Produce vehicle occupancies by transit routes.
* ``stop_passenger_interactions``: Boardings and Alightings by time slice.
* ``stop_to_stop_passenger_counts``: Passenger counts between directly connected stops/stations.
* ``stop_passenger_waiting``: Agent waiting times for unique pt interaction events.
* ``vehicle_departure_log``: Vehicle departures and delays from facilities (stops in the case of PT).
* ``vehicle_passenger_log``: A log of every passenger boarding and alighting to/from a transit vehicle.
* ``vehicle_passenger_graph``: Experimental support for building interaction graph objects (networkx).
* ``agent_tolls_log``: Produce a log and 24-hour summary (total amount, number of tolls incurred) of agents' tolling events.

### **Plan Handlers/WorkStation Tools**

These are processed by streaming through all output plans from simulation.
Compared to the event based outputs these are typically more aggregate but can be computationally faster and can be used to expose agent plan
'memories' and plan scoring.

Currently supported handlers include:

* ``trip_modes``: Produce modeshares and counts of all trips.
* ``trip_activity_modes``: Produce modeshares and counts of all trips to specified destination activities.
* ``plan_modes``: Produce modeshares and counts of all plans (ie the dominant mode for each person).
* ``plan_activity_modes``: Produce modeshares and counts of all plans (ie the dominant mode for each person) to specified destination activities.
* ``trip_logs``: Produce agent activity logs and trip logs for all selected plans. Omits pt interactions and individual trip legs. Trip mode is based on maximum leg distance.
* ``leg_logs``: Produce agent activity logs and leg logs for all selected plans.
* ``plan_logs``: Produce agent plans including unselected plans and scores.
* ``agent_highway_distance_logs``: Produce agent distances by car on different road types. Requires network to have `osm:way:highways` attribute.
* ``trip_highway_distance_logs``: Produce flat output of agent trip distances by car on different road types. Requires network to have `osm:way:highways` attribute.
* **(DEPRECATED)** ``toll_logs``: Produces summary of amounts agents paid at tolls based on route information contained in agent plans. Requires road pricing input file. Only works for option ``["car"]``. *The AgentTollsPaidFromRPConfig handler is still supported, but has been superseded by an event handler. This handler calculates toll payments via a link-lookup with the road pricing configuration file, and will not account for any in-simulation adjustments for differential or capped road pricing.*

### **Post Processing Handlers/Workstation Tools**

These are outputs produced through additional post-processing of the above outputs. Currently supported postprocessors include:

* ``vkt``: Produce link volume vehicle kms by time slice.
* ``trip_duration_breakdown``: Produce binned trip durations.
* ``trip_euclid_distance_breakdown``: Produce binned trip distances.
* ``plan_summary``: Produce summary statistics for all agent plans

### **Benchmarking Handlers/WorkStation Tools**

Elara can also assist with validation or 'benchmarking' of simulations. Elara will compare and present simulation results from the above outputs to available benchmarks, it will aditionally output a distance based score for the model. Where distance is some measure of how different the simulation is from the observed data.

The handlers in this module require correctly formatted benchmark data. Examples can be found in the project `example_benchmark_data` folder. This documentation has been moved to [here](https://github.com/arup-group/elara/benchmarks.md).

## Running from a Configuration File

For reproducibility or to process a range of outputs, using a config file is the most sensible and processing efficient way to use Elara. From your terminal:

`elara run <CONFIG PATH>`

Config files must be `.toml` format and be roughly formatted as follows. The various fields are detailed further below and an [example config](https://github.com/arup-group/elara/blob/master/example_configs/config.toml) is also included in the repo:

```{.toml}
[scenario]
name = "test_town"
time_periods = 24
scale_factor = 0.01
crs = "EPSG:27700"

[inputs]
events = "./tests/test_fixtures/output_events.xml"
network = "./tests/test_fixtures/output_network.xml"
transit_schedule = "./tests/test_fixtures/output_transitSchedule.xml"
transit_vehicles = "./tests/test_fixtures/output_transitVehicles.xml"
attributes = "./tests/test_fixtures/output_personAttributes.xml"
plans= "./tests/test_fixtures/output_plans.xml"
output_config_path = "./tests/test_fixtures/output_config.xml"
road_pricing = "./tests/test_fixtures/roadpricing.xml"
vehicles = "./tests/test_fixtures/output_vehicles.xml"

[outputs]
path = "./tests/test_outputs"

[event_handlers]
link_vehicle_counts = ["car", "bus"]
link_passenger_counts = ["bus"]
stop_passenger_counts = ["bus"]
stop_passenger_waiting = ["all"]

[plan_handlers]
trip_modes = {modes=["all"]}
trip_activity_mode_shares = {destination_activity_filters = ["work"]}
trip_logs = ["all"]
agent_highway_distance_logs = ["car"]
trip_highway_distance_logs = ["car"]

[post_processors]
vkt = ["car"]
plan_summary = ["all"]
trip_duration_breakdown = ["all"]
trip_euclid_distance_breakdown = ["all"]
```

You can run this config on some toy data: `elara run example_configs/config.toml` (from the project root).

If your MATSim outputs use default names and are in the same directory, you may optionally pass the path of this directory as single argument to `[inputs]` using, e.g. `inputs_directory = "./tests/test_fixtures/"`. **NB:** If using the ``toll_log`` plan handler, you must still provide the `road_pricing` path separately.

### Configuration dry run

`elara run` supports some optional flags and arguments which can be discovered with `elara run --help`, most useful is `elara run -d` (or `elara run --dry`) which can be use to test a config without providing valid inputs or undertaking the processing.

## Command Line Reference

Elara can also be more generally used via the CLI to process individual outputs. Used in this manner the CLI should be pretty discoverable, once installed, try the command `elara` in your terminal to find out about the available options:

```{sh}
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

Further commands can then be explored. In general the commands and options available follow the same structure as configuration.

```{sh}
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
  -v, --version INT           MATSim version {11,12}, defaults to 12.
  -p, --time_periods INTEGER  Time period breakdown, defaults to 24 (hourly.
  -o, --outputs_path PATH     Outputs path, defaults to './elara_out'.
  -i, --inputs_path PATH      Inputs path location, defaults to current root.
  -x, --no_experienced_plans  Switch for turning off Experienced Plans. Set to
                              use output_plans instead.
  -n, --name TEXT             Scenario name, defaults to root dir name.
  -d, --debug                 Switch on debug verbosity.
  --help                      Show this message and exit.
```

*Note that defaults will not always be suitable for your scenario. `-s` `--scale_factor` in
particular, should be updated accordingly.*

*Similarly, note that the CLI assumes inputs will have standard (at the time of writing) MATSim
names, ie `output_plans.xml.gz`, `output_personAttributes.xml.gz`, `output_config.xml` and so on.*

### Example CLI Usage

To produce **vehicle kilometres travelled by car (VKT)** for a London scenario, all the defaults are correct but you'd like to read the inputs from a different location to your current path:

`elara post-processors vkt car -inputs_path ~/DIFFERENT/DATA/LOCATION`

or, more succinctly:

`elara post-processors vkt car -i ~/DIFFERENT/DATA/LOCATION`

To reduce **volume counts for cars and buses** using a New Zealand projection (2113). The scenario was
a 1% sample. You'd like to prefix the outputs as 'nz_test' in a new directory '~/Data/nz_test':

`elara event-handlers volume-counts car bus -epsg EPSG:2113 -scale_factor .01 -name nz_test
-outputs_path ~/Data/nz_test`

or, much more succinctly:

`elara event-handlers link-vehicle-counts car bus -e EPSG:2113 -s .01 -n nz_test -o ~/Data/nz_test`

To produce a **benchmark**, in this case we assume that a benchmark has already been created called and that it works for all modes.

`elara benchmarks mode_shares_comparison all`

Note that we are assuming that all other option defaults are correct. ie:

* --scale_factor = 0.1
* --time_periods = 24
* etc

## Configuration Options

### Configuration Reference

Configured fields are described below:

**[scenario]**

**name** *string* *(required)*

The name of the scenario being processed.

**time_periods** *integer* *(required)*

The number of time slices used to split a 24-hour period for the purposes of reporting. A value
of ``24`` will produce summary metrics for each our of the day. Similarly, a value of ``96`` will produce 15-minute summaries.

**scale_factor** *float* *(required)*

The sample size used in the originating MATSim scenario run. This is used to scale metrics such as volume counts. For example, if the underlying scenario was run with a 25% sample size, a value of ``0.25`` in this field will ensure that all calculated volume counts are scaled by 4 times.

 **version** *int {11,12}* *(default 12)*

Set `version = 11` if using MATSim version 11 outputs (in which case there a personAttributes file will be required).

**using_experienced_plans** *boolean {true, false}* *(default true)*

If using MATSim "experienced_plans" you should set this flag to 'true'.

**crs** *string* *(required)*

The EPSG code specifying which coordinate projection system the MATSim scenario inputs used. This is used to convert the results to WGS 84 as required.

**verbose** *string* *(required)*

Logging module [level](https://docs.python.org/3/library/logging.html#levels), for example, either ERROR, WARNING, INFO or DEBUG.

**[inputs]**

**inputs_directory** *path*
Path to a directory containing MATSim Outputs. A convenient override for setting paths to standard MATSim outputs using a single line. Handles network, plans, attributes, events, transit vehicles, transit schedule, output config files.

**events** *file*

Path to the MATSim events XML file. Can be absolute or relative to the invocation location.

**.network** *file*
Path to the MATSim network XML file. Can be absolute or relative to the invocation location.

**etc** *file*

Path to additional MATSim resources:

* transit_schedule
* transit_vehicles
* attributes
* plans
* output_config_path
* road_pricing

**[outputs]**

**path** *directory* *(required)*

Desired output directory. Can be absolute or relative to the invocation location. If the directory does not exist it will be created.

**[event_handlers]**

**[NAME]** *list of strings as below* *(all optional)*

Specification of the plan handlers to be run during processing. For example:

```{toml}
[event_handlers]
link_vehicle_counts = ["car", "bus"]
link_passenger_counts = ["bus"]
```

All available event handlers are described in [Output Handlers](https://github.com/arup-group/elara#output-handlers).

The associated list attached to each handler allows specification of the network modes to be processed using that handler. This allows certain handlers to be activated for specific modes.

* modes should be supplied as a list eg ``["car", "bus", "train", ...]`` or just ``["car"]``.
* note that ``waiting_times`` only supports the option of ``["all"]``.
* note that ``vehicle_departure_log`` supports any valid mode options, but is intended for public transit modes only. It will produce empty csv outputs for modes without events of type ``VehicleDepartsAtFacility``.

The above format of `HANDLER_NAME = ["car", "bus"]` is a shorthand way of passing options. These options can also be passed in a dictionary format, ie `HANDLER_NAMES = {modes=["car","bus"]}`. More complex configuration options using this dictionary option are described later.

**[plan_handlers]**

**[NAME]** *list of strings as below* *(optional)*

Specification of the plan handlers to be run during processing. For example:

```{toml}
[plan_handlers]
trip_modes = ["all"]
trip_logs
agent_highway_distance_logs = ["car"]
```

All available event handlers are described in [Output Handlers](https://github.com/arup-group/elara#output-handlers).

The associated list attached to each handler allows specification of the network modes to be processed using that handler. This allows certain handlers to be activated for specific modes.

* in many cases the only modes supported are ``["all"]`` (this is a default, demonstrated in the above example for `trip_logs`).
* highway_distances only supports the mode ``car``

**[post_processors]**

**[NAME]** *list of strings* *(optional)*

Specification of the event handlers to be run post processing. For example:

```{toml}
[post_processors]
vkt = ["car"]
plan_summary = ["all"]
trip_duration_breakdown = ["all"]
```

The associated list attached to each handler allows specification of which modes of transport
should be processed using that handler.

* modes should be supplied as a list eg ``["car", "bus", "train", ...]`` or just ``["car"]``.
* note that ``plan_summary`` and `breakdowns` only support the option of ``["all"]``.

### Additional Options

The above configurations typically pass a modes option, eg:

```{.toml}
[event_handlers]
link_vehicle_counts = ["car", "bus"]
```

*Note that if no mode is defined a defualt of `["all"]` will be passed which is not a valid mode for most handlers.*

This is equivalent to passing a dictionary with the `modes` option as a key, eg:

```{.toml}
[event_handlers]
link_vehicle_counts = {modes=["car", "bus"]}
```

This more verbose method, of using a dictionary of handler options, allows more options to be passed to handlers. Allowing them to have more complex functionality.

Most handlers support a groupby operation that groups outputs such as counts based on person attribute. This can be enabled by passing the `groupby_person_attributes` option as follows:

```{.toml}
[event_handlers]
link_vehicle_counts = {modes = ["car","bus"], groupby_person_attributes = ["income", "age"]}
```

The above config will produce a number of link counts for car and bus with breakdowns by the person attributes named 'income' and 'age'. Note that cross tabulations are not supported and that specifying an unavailable attribute key will result in a warning and outputs being assigned to the 'None' group.

An example config using the `groupby_person_attributes` option is included: `./example_configs/complex_options.toml`.

Some handlers require additional options to be passed, for example `activity_mode_shares` which calculated mode share only for trips with given destination activity types:

```{.toml}
trip_activity_mode_shares = {destination_activity_filters = ["work"]}
```

You can also name your handler, which will be added to the output file names:

```{.toml}
trip_activity_mode_shares = {name = "commuters", destination_activity_filters = ["work"]}
```

Tabular (csv) outputs of the event and plan handlers can optionally be exported to a compressed format, by passing the `compression` option:

```{.toml}
[event_handlers]
link_vehicle_counts = {modes = ["car","bus"], compression = "gzip"}
```

### Letting Elara Deal With Dependancies

The `modes` and `groupby_person_attributes` options are understood by elara as dependancies. Therefore if you ask for a postprocessor such as VKT (or a benchmark - described later below), elara will make sure that any handlers required to produce these outputs will also be created and given the correct options.

This means that a config can be very minimal. As an example, the following configurations are equivalent:

*A complex config, using a post processor that requires outputs from the event handlers:*

```{.toml}
...
[event_handlers]
link_vehicle_counts = {modes=["car","bus"], groupby_person_attributes=["age"]}

[post_processors]
vkt = {modes=["car"], groupby_person_attributes=["age"]}
...
```

*An equivalent config, allowing Elara to deal with the dependancies:*

```{.toml}
...
[event_handlers]
link_vehicle_counts = ["bus"]

[post_processors]
vkt = {modes=["car"], groupby_person_attributes=["age"]}
...
```

Care should be taken not to specify unnecesary options as they can add significant compute time. An example configuration that passes complex dependancies is included: `./example_configs/complex_dependancies.toml`

### Experienced Plans

Elara assumes your MATSim run is configured to output agents' "experienced" plans (rather than "planned" plans).  If you do **not** wish to use experienced plans, you can set the following option in the config:

```{.toml}
[scenario]
...
using_experienced_plans = false
```

This option is set to `true` by default, and does not need to be included it in the config, although you may set it explicitly if you wish.

When using MATSim "experienced" plans from versions 12 and up you will find that these (currently) do not include the person attributes. In turn, this prevents elara from providing outputs grouped by person attributes (ie those that use the `groupby_person_attributes` option).

If such outputs are required when using experienced plans, provide the standard MATSim 'output_plans' as the attributes input:

```{.toml}
[inputs]
...
attributes = "./tests/test_fixtures/output_plans_v12.xml"
plans= "./tests/test_fixtures/output_experienced_plans.xml"
...
```

This allows elara to access person attributes from the standard output_plans, while taking plans from the output_experienced_plans. This is not necessary if groupby_person_attributes are not required.

This guidance does apply when using MATSim version 11 and below, because attributes must always be set explicitly to the relevant `output_personAttributes.xml` file.

## Tests

### Run the tests (from the elara root dir)

```{sh}
pytest -v
```

### Generate a code coverage report

To generate XML & HTML coverage reports to `reports/coverage`:

```{sh}
./scripts/code-coverage.sh
```

## Debug

Logging level can be set in the config or via the cli, or otherwise defaults to False (INFO). We
currently support the following levels: DEBUG, INFO.

Note that the configured or default logging level can be overwritten to debug using an env variable:

```{sh}
export ELARA_LOGGINGLEVEL='True'
```

## Technical Details

Elara is designed to work out it's own dependencies and be somewhat efficient with compute using `elara.factory`. This allows some flexibility and succinctness in application and is particularly suited to extracting batches of outputs as defined in a config.

This design allows Elara to create arbitrarily complex pipelines of output processing, postprocessing and benchmarking as defined by a configuration file or via the CLI.

Elara defines a graph of connected `WorkStations`, each responsible for building certain types of output or intermediate data requirements (`Tools`). These workstations are connected to their respective dependees and dependencies (managers and suppliers) to form a **DAG**.

 ![dag](images/dag.png)

Elara uses this **DAG** to provide:

* **Minimal** dependency processing
* **Early validation** of all intermediate requirements
* **Early Failure**/Ordering

Elara does this by traversing the DAG in three stages:

1) Longest path search to ensure correct order.

2) Breadth first initiation of `.tools` as `.resources` and supplier validation.

3) Reverse Breadth first build all tools in `.resources` and build workstation.

**Is it fast/sensible?**

Sometimes:

![dag](images/design.png)

## Adding Features

**NOTE**: Pushing code to this repository is temporarily restricted while we undergo some spring cleaning during April/May 2022. If you wish to contribute code, please contact one of the owners for permisison.

Elara is designed to be extendable, primarily with new tools such as handlers or benchmarks.

Where new tools are new classes that implement some process that fits within the implementation (both in terms of code and also abstractly) of a WorkStation. New tools must be added to the relevant workstations 'roster' of tools (ie `.tools`).

New tool classes must correctly inherit and implement a number of values and methods so that they
play well with their respective workstation:

* ``.__init__(self, config, mode)``: Used for early validation of mode and subsequent requirements.
* ``.build(self, resource=None)``: Used to process required output (assumes required resources are available).

A good place to start when adding a new handler/tool is to copy or adapt an existing one.

It may also be required to add or connect new workstations, where new workstations are new types of process that might contain a new or multiple new tools. New workstations must be defined and connected to their respective suppliers and managers in `main`.

### Conventions

Where possible please name new handlers/tools based on the following:

`<Where><What>`

For example: `LinkPassengerCounts` or `StopPassengerCounts`.

## Todo
* Add support for all available handlers to CLI
* More descriptive generated column headers in handler results. Right now column headers are simply numbers mapped to the particular time slice during the modelled day.
* Try and move away from test_data towards unit tests.
* More outputs, ie mode distances/times/animations.
* S3 integration (read and write).
* Clean up docstrings and add "read the docs" style documentation.

## What does the name mean?

[Elara]("https://en.wikipedia.org/wiki/Elara_(moon)") is a moon of Jupiter. There's not much else interesting to say about it, other than that the wikipedia article states that it orbits 11-13 *gigametres* from the planet. Gigametres is a cool unit of measurement.
