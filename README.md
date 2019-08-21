# Elara

A command line utility for processing MATSim output XML files:

* **Event Based Outputs** (for example traffic 'flow')
* **Plan Based Outputs** (for example mode shares)
* **Post Processing** of Outputs (for example for Vehicle kms)
* **Benchmarking** of Outputs (for example comparison to measured cordon counts)

## Contents
* [Installation](#markdown-header-installation)
* [About](#markdown-header-about)
* [Command line reference](#markdown-header-command-line-reference)
* [Configuration format](#markdown-header-configuration-format)
* [Adding features](#markdown-header-adding-features)
* [Todo](#markdown-header-todo)
* [What does the name mean?](#markdown-header-what-does-the-name-mean)

## Installation
Clone or download the repository from the [downloads section](https://bitbucket.org/arupdigital/elara/downloads/). Once available locally, navigate to the folder and run:
```
pip3 install -e .
elara --help
```

The ``GeoPandas`` library requires ``pyproj`` as a dependency. This can be a bit of a pain to install. For Mac OSX, activate the environment Elara lives in and run the following commands before installing the tool:
```
pip3 install cython
pip3 install git+https://github.com/jswhit/pyproj.git
```

On Windows, pre-compiled wheels of ``pyproj`` can be found on [this page](https://www.lfd.uci.edu/~gohlke/pythonlibs/). Manually install the correct ``pyproj`` wheel within your environment using pip.  

Like (bad) tests?
```
cd elara
pytest
```

## About
Elara is designed to undertake arbitrarily complex pipelines of output processing, postprocessing
 and benchmarking as defined by a configuration file, such as `elara.scenario.toml`.
 
Elara defines a graph of connected `WorkStations`, each responsible for building certain types of
 output or intermediate data requirements (`Tools`). These workstations are connected to their 
 respective 
 dependees and dependencies (managers and suppliers) to form a DAG.
 
Elara uses this DAG to provide:

* **Minimal** dependency processing
* **Early validation** of all intermediate requirements 
* **Early Failure**/Ordering

Elara does this by traversing the DAG in three stages:

1) Longest path search to ensure correct order.

2) Breadth first initiation and supplier validation.

3) Reverse Breadth first build all requirements.

## Command line reference
```
Usage: elara [OPTIONS] CONFIG_PATH

  Command line tool for processing a MATSim
  scenario events output. :param config_path:
  Configuration file path

Options:
  -h, --help  Show this message and exit.
```

Given the path to a suitable configuration TOML file (see [here](#markdown-header-configuration-format)), processes a MATSim events file and produces the desired
 summary metric files. For example: `elara scenario.toml`.

## Configuration format
This utility uses a TOML configuration format to specify input, output and metric generation options. For example:
```
 [scenario]
name = "test_town"
time_periods = 24
scale_factor = 0.01
crs = "EPSG:27700"
verbose = true

[inputs]
events = "./tests/test_fixtures/output_events.xml"
network = "./tests/test_fixtures/output_network.xml"
transit_schedule = "./tests/test_fixtures/output_transitSchedule.xml"
transit_vehicles = "./tests/test_fixtures/output_transitVehicles.xml"
attributes = "./tests/test_fixtures/output_personAttributes.xml"
plans= "./tests/test_fixtures/output_plans.xml"

[event_handlers]
volume_counts = ["car"]
passenger_counts = ["bus", "train"]
stop_interactions = ["bus", "train"]

[plan_handlers]
mode_share = ["all"]

[post_processors]
vkt = ["car"]

[benchmarks]
test_town_cordon = ["car"]

[outputs]
path = "./tests/test_outputs"
contract = true
```

The following fields are available:

**#** scenario.**name** *string* *(required)*

The name of the scenario being processed, using when naming output files.

**#** scenario.**time_periods** *integer* *(required)*

The number of time slices used to split a 24-hour period for the purposes of reporting. A value of ``24`` will produce summary metrics for each our of the day. Similarly, a value of ``96`` will produce 15-minute summaries.

**#** scenario.**scale_factor** *float* *(required)*

The sample size used in the originating MATSim scenario run. This is used to scale metrics such as volume counts. For example, if the underlying scenario was run with a 25% sample size, a value of ``0.25`` in this field will ensure that all calculated volume counts are scaled by 4 times.

**#** scenario.**crs** *string* *(required)*

The EPSG code specifying which coordinate projection system the MATSim scenario inputs used. This
 is used to convert the results to WGS 84 as required. 

**#** inputs.**events** *file*

Path to the MATSim events XML file. Can be absolute or relative to the invocation location.

**#** inputs.**network** *file*

Path to the MATSim network XML file. Can be absolute or relative to the invocation location.

**#** inputs.**etc** *file*

Path to additional MATSim resources.

**#** event_handlers.**[handler name]** *list of strings* *(optional)*

Specification of the event handlers to be run during processing. Currently available handlers include:

* ``volume_counts``: Produce link volume counts and volume capacity ratios by time slice.
* ``passenger_counts``: Produce vehicle occupancy by time slice.
* ``stop_interactions``: Boardings and Alightings by time slice.

The associated list attached to each handler allows specification of which modes of transport should be processed using that handler. This allows certain handlers to be activated for public transport modes but not private vehicles for example. Possible modes currently include:

* eg ``car, bus, train``

**#** plan_handlers.**[handler name]** *list of strings* *(optional)*

Specification of the plan handlers to be run during processing. Currently available handlers 
include:

* ``mode_share``: Produce global modeshare of final plans using a mode hierarchy.

The associated list attached to each handler allows specification of additional options:

* eg ``all, <NOT IMPLEMENTED>``

**#** post_processors.**[post-processor name]** *list of strings* *(optional)*

Specification of the event handlers to be run post processing. Currently available handlers include:

* ``vkt``: Produce link volume vehicle kms by time slice.

The associated list attached to each handler allows specification of which modes of transport should be processed using that handler. This allows certain handlers to be activated for public transport modes but not private vehicles for example. Possible modes currently include:

* eg ``car, bus, train``

**#** benchmarks.**[benchmarks name]** *list of strings* *(optional)*

Specification of the benchmarks to be run post processing. Currently available benchmarks include:

* ``london_inner_cordon_car``
* ``dublin_canal_cordon_car``
* ``ireland_commuter_modeshare``
* ``test_town_cordon``
* ``test_town_peak_cordon``
* ``test_town_modeshare``

The associated list attached to each handler allows specification of which modes of transport should be processed using that handler. This allows certain handlers to be activated for public transport modes but not private vehicles for example. Possible modes currently include:

* eg ``car, bus, train``

**#** outputs.**path** *directory* *(required)*

Desired output directory. Can be absolute or relative to the invocation location. If the directory does not exist it will be created.

**#** outputs.**contract** *boolean*

If set to *true*, removes rows containing only zero values from the generated output files. 

## Adding Features

Elara is designed to be extendable, primarily with new tools such as handlers or benchmarks. 
New tool classes must correctly inherit and implement a number of values and methods so that they 
play well with their respective WorkStation:
 
* ``.__init__(self, config, option)``: Used for early validation of option and 
subsequent requirements.
* ``.build(self, resource=None)``: Used to process required output (assumes required resources are 
available).

`tool_templates` provides some templates and notes for adding new tools. Additionally remember to
 add new tools to their `WorkStation` `.tools` dictionary and the docs.

## Todo

**Priority**

* Addition of additional handlers, including average link travel time, node boardings/alightings, etc. This will be relatively easy to achieve once sample outputs are available to test against - the logic already exists in old scripts. 
* Add tests and refactor to make testing easier! 

**Nice to have**

* More descriptive generated column headers in handler results. Right now column headers are simply numbers mapped to the particular time slice during the modelled day. 
* Introduction of a --verbose option for more descriptive terminal outputs.

## What does the name mean?
[Elara]("https://en.wikipedia.org/wiki/Elara_(moon)") is a moon of Jupiter. There's not much else interesting to say about it, other than that the wikipedia article states that it orbits 11-13 *gigametres* from the planet. Gigametres is a cool unit of measurement. 
