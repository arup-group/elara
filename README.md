# Elara

A command line utility for processing (in big batches or bit by bit) MATSim outputs:

* **Event Based Outputs** (for example transport network 'flows' or agent waiting times)
* **Plan Based Outputs** (for example mode shares or final agent plan records)
* **Post Processing** of Outputs (for example for vehicle kms)
* **Benchmarking** of Outputs (for example comparison to measured cordon counts)

The outputs from elara can be used for further analysis, visualisation or calibration. Outputs 
are typically available as csv and/or geojson. Spatial representations are converted to 
EPSG:4326, which works in kepler.

## Contents
* [Introduction](https://github.com/arup-group/elara#introduction)
* [Installation](https://github.com/arup-group/elara#installation)
* [Command Line Reference](https://github.com/arup-group/elara#command-line-reference)
* [Example CLI Commands](https://github.com/arup-group/elara#example-cli-commands)
* [Configuration](https://github.com/arup-group/elara#configuration)
* [Tests](https://github.com/arup-group/elara#tests)
* [Debug](https://github.com/arup-group/elara#debug)
* [About](https://github.com/arup-group/elara#about)
* [Adding Features](https://github.com/arup-group/elara#adding-features)
* [Todo](https://github.com/arup-group/elara#todo)
* [What does the name mean?](https://github.com/arup-group/elara#what-does-the-name-mean)

## Introduction
Elara uses a Command Line Interface (CLI), but within this interface you can choose to run elara 
via a config or purely through the CLI. The CLI is preferred for producing single outputs, whereas 
the config can handle big batches of outputs in one go. So if you are after a quick output - best 
to use the CLI, whereas if you need a large output and/or reproducibility, a config is preferred.

The CLI and configs share naming conventions and structure so we recommend reading about both below.

## Installation
Clone or download this repository. Once available locally, navigate to the folder and run:
```
pip3 install -e .
elara --help
```

The ``GeoPandas`` library requires ``pyproj`` as a dependency. This can be a bit of a pain to 
install. For Mac OSX, activate the environment Elara lives in and run the following commands 
before installing the tool:
```
pip3 install cython
pip3 install git+https://github.com/jswhit/pyproj.git
```

On Windows, pre-compiled wheels of ``pyproj`` can be found on
[this page](https://www.lfd.uci.edu/~gohlke/pythonlibs/). Manually install the correct ``pyproj``
 wheel within your environment using pip.  

We require pyproj=='2.4.0' because older version have proven to be very slow/hang for converting 
between coordinate reference systems.

## Command Line Reference

Elara should be pretty discoverable, once installed, try the command `elara` in your terminal to 
find out about the available options:

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
  -p, --time_periods INTEGER  Time period breakdown, defaults to 24 (hourly.
  -o, --outputs_path PATH     Outputs path, defaults to './elara_out'.
  -i, --inputs_path PATH      Inputs path location, defaults to current root.
  -n, --name TEXT             Scenario name, defaults to root dir name.
  -d, --debug                 Switch on debug verbosity.
  --help                      Show this message and exit.
```

**Note that defaults will not always be suitable for your scenario. `-s` `--scale_factor` in 
particular, should be updated accordingly.**

Similarly, not that the CLI assumes inputs will have standard (at the time of writing) MATSim 
names, ie `output_plans.xml.gz`, `output_personAttributes.xml.gz`, `output_config.xml` and so on.

## Example CLI Commands

Produce **car VKT** for a London scenario, all the defaults are correct but you'd like to read the 
inputs from a different location to your current location:

`elara post-processors vkt car -inputs_path ~/DIFFERENT/DATA/LOCATION`

or, slightly more succinctly:

`elara post-processors vkt car -i ~/DIFFERENT/DATA/LOCATION`

Produce **volume counts for cars and buses** using a New Zealand projection (2113). The scenario was 
a 1% sample. You'd like to prefix the outputs as 'nz_test' in a new directory '~/Data/nz_test':

`elara event-handlers volume-counts car bus -epsg EPSG:2113 -scale_factor .01 -name nz_test 
-outputs_path ~/Data/nz_test`

or, much more succinctly:

`elara event-handlers volume-counts car bus -e EPSG:2113 -s .01 -n nz_test -o ~/Data/nz_test`

Produce a **benchmark**, in this case we will assume that a benchmark has already been created 
called `dublin_cordon` and that it works for buses, cars and trains.

`elara benchmarks dublin_cordon car bus`

Note that commands with dependencies (such as post-processors and benchmarks) will have these 
dependencies automatically fulfilled. In the case of the dublin benchmark, this means that 
outputs for car and bus volume-counts will also be procuced.

## Configuration

For reproducibility or to process a range of outputs, configuration is the most sensible and 
processing efficient way to proceed.

Configuration is accessed via the CLI:

`elara run <CONFIG PATH>` 

Config files must be `.toml` format and be roughly formatted as follows. The various fields are 
detailed further below and examples are also included in the repo:

```
 [scenario]
name = "test_town"
time_periods = 24
scale_factor = 0.01
crs = "EPSG:27700"
verbose = INFO

[inputs]
events = "./tests/test_fixtures/output_events.xml"
network = "./tests/test_fixtures/output_network.xml"
transit_schedule = "./tests/test_fixtures/output_transitSchedule.xml"
transit_vehicles = "./tests/test_fixtures/output_transitVehicles.xml"
attributes = "./tests/test_fixtures/output_personAttributes.xml"
plans= "./tests/test_fixtures/output_plans.xml"
output_config_path = "./tests/test_fixtures/output_config.xml"

[outputs]
path = "./tests/test_outputs"
contract = true

[event_handlers]
volume_counts = ["car"]
passenger_counts = ["bus", "train"]
stop_interactions = ["bus", "train"]
waiting_times = ["all"]

[plan_handlers]
mode_share = ["all"]
agent_logs = ["all"]
highway_distances = ["car"]

[post_processors]
vkt = ["car"]

[benchmarks]
test_pt_interaction_counter = ["bus"]
test_link_cordon = ["car"]

```

**#** scenario.**name** *string* *(required)*

The name of the scenario being processed, using when naming output files.

**#** scenario.**time_periods** *integer* *(required)*

The number of time slices used to split a 24-hour period for the purposes of reporting. A value 
of ``24`` will produce summary metrics for each our of the day. Similarly, a value of ``96`` will
 produce 15-minute summaries.

**#** scenario.**scale_factor** *float* *(required)*

The sample size used in the originating MATSim scenario run. This is used to scale metrics such 
as volume counts. For example, if the underlying scenario was run with a 25% sample size, a value
 of ``0.25`` in this field will ensure that all calculated volume counts are scaled by 4 times.

**#** scenario.**crs** *string* *(required)*

The EPSG code specifying which coordinate projection system the MATSim scenario inputs used. This
 is used to convert the results to WGS 84 as required. 

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

* ``volume_counts``: Produce link volume counts and volume capacity ratios by time slice.
* ``passenger_counts``: Produce vehicle occupancy by time slice.
* ``stop_interactions``: Boardings and Alightings by time slice.
* ``waiting_times``: Agent waiting times for unique pt interaction events.

The associated list attached to each handler allows specification of which options (typically modes
 of transport) should be processed using that handler. This allows certain handlers to be activated 
for public transport modes but not private vehicles for example. Possible modes currently include:

* eg ``car, bus, train, tram, ferry, ...``.
* note that ``waiting_times`` only supports the option of ``["all"]``.

**#** plan_handlers.**[handler name]** *list of strings* *(optional)*

Specification of the plan handlers to be run during processing. Currently available handlers 
include:

* ``mode_share``: Produce global modeshare of final plans using a mode hierarchy.
* ``agent_logs``: Produce flat output of agent activity logs and leg logs, including times, 
sequences, durations and categories.
* ``agent_plans``: Produce flat output of agent plans (logs and activities) including unselected 
plans and scores, 
including times, 
sequences, durations and categories.
* ``highway_distances``: Produce flat output of agent distances by car on different road 
types (as described by the input network osm:way).

The associated list attached to each handler allows specification of additional options:

* in most cases ``all``
* agent_plans support subpopulation selection, eg ``rich, poor``
* highway_distances only supports ``car``

**#** post_processors.**[post-processor name]** *list of strings* *(optional)*

Specification of the event handlers to be run post processing. Currently available handlers include:

* ``vkt``: Produce link volume vehicle kms by time slice.
* ``trip_logs``: Produce record of all agent trips using mode hierarchy to reveal mode of trips 
with multiple leg modes.

The associated list attached to each handler allows specification of which modes of transport 
should be processed using that handler. This allows certain handlers to be activated for public 
transport modes but not private vehicles for example. Possible modes currently include:

* eg ``car, bus, train, ...``
* note that ``trip_logs`` only supports the option of ``["all"]``.

**#** benchmarks.**[benchmarks name]** *list of strings* *(optional)*

Specification of the benchmarks to be run. These include a variety of highway counters, 
cordons and mode share benchmarks for specific projects. **Benchmarks calculated using preprocessed
 data unique to the given scenario network. This means that a given benchmark will not work for a
  given scenario (say 'London') unless the same network and/or schedule are in use.** Where a 
  network or schedule has been changed, the project [bench](https://github.com/arup-group/bench) 
  has been created to pre-process this data. 

Currently available benchmarks include:

_newer formats (produced using `bench`):_

* ``london_rods``
* ``london_central_cordon``
* ``london_inner_cordon``
* ``london_outer_cordon``
* ``london_thames_screen``
* ``test_pt_interaction_counter``
* ``test_link_cordon``

_older formats:_

* ``ireland_highways``
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

## Tests

### Run the tests (from the elara root dir)

    python -m pytest -vv tests

### Generate a code coverage report

To generate XML & HTML coverage reports to `reports/coverage`:

    ./scripts/code-coverage.sh
    
## Debug

Logging level can be set in the config or via the cli, or otherwise defaults to False (INFO). We 
currently support the following levels: DEBUG, INFO, WARN/WARNING.

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
 
* ``.__init__(self, config, option)``: Used for early validation of option and 
subsequent requirements.
* ``.build(self, resource=None)``: Used to process required output (assumes required resources are 
available).

`tool_templates` provides some templates and notes for adding new tools.
 
 It may also be required to add or connect new workstations, where new workstations are new types
  of process that might contain a new or multiple new tools. New workstations must be defined and
   connected to their respective suppliers and managers in `main`.

## Todo

* More descriptive generated column headers in handler results. Right now column headers are 
simply numbers mapped to the particular time slice during the modelled day. 
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
