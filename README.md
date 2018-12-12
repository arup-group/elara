# Elara

A command line utility for processing a MATSim scenario events output XML file. Generates flat CSV files summarising key metrics by time slice for links or nodes across the input network. 

## Contents
* [Installation](#markdown-header-installation)
* [About](#markdown-header-about)
* [Command line reference](#markdown-header-command-line-reference)
* [Configuration format](#markdown-header-configuration-format)
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

## About
A MATSim scenario run generates a number of different output files, including the events record. This particular XML file enumerates every discrete event that happened during the final iteration of the model. As such, it contains most of the information required to generate reporting for the particular model run. 

Given a configuration file pointing to this file, as well as a number of other auxiliary files, this utility will produce a customisable selection of summary metric CSVs. These CSVs can then be joined to network geometry (through Shapefiles, GeoJSON etc.) or themselves processed downstream to present scenario results. 

## Command line reference
```
Usage: elara [OPTIONS] CONFIG_PATH

  Command line tool for processing a MATSim
  scenario events output. :param config_path:
  Configuration file path

Options:
  -h, --help  Show this message and exit.
```

Given the path to a suitable configuration TOML file (see [here](#markdown-header-configuration-format)), processes a MATSim events file and produces the desired summary metric files. Example usage:

* ``elara scenario.toml``  
Processes according to the settings present in the ``scenario.toml`` file.

**#** mimas **-h** | **--help**

Output usage information.

## Configuration format
This utility uses a TOML configuration format to specify input, output and metric generation options. For example:
```
[scenario]
name = "test_scenario"
time_periods = 24
scale_factor = 0.25
crs = "EPSG:27700"

[inputs]
events = "./fixtures/test-events.xml"
network = "./fixtures/test-network.xml"

[outputs]
path = "./outputs"

[handlers]
volume_counts = ["car"]
```

The following fields are available:

**#** scenario.**name** *string* *(required)*

The name of the scenario being processed, using when naming output files.

**#** scenario.**time_periods** *integer* *(required)*

The number of time slices used to split a 24-hour period for the purposes of reporting. A value of ``24`` will produce summary metrics for each our of the day. Similarly, a value of ``96`` will produce 15-minute summaries.

**#** scenario.**scale_factor** *float* *(required)*

The sample size used in the originating MATSim scenario run. This is used to scale metrics such as volume counts. For example, if the underlying scenario was run with a 25% sample size, a value of ``0.25`` in this field will ensure that all calculated volume counts are scaled by 4 times.

**#** scenario.**crs** *string* *(required)*

The EPSG code specifying which coordinate projection system the MATSim scenario inputs used. This is used to convert the results to WGS 84. 

**#** inputs.**events** *file* *(required)*

Path to the MATSim events XML file. Can be absolute or relative to the invocation location.

**#** inputs.**network** *file* *(required)*

Path to the MATSim network XML file. Can be absolute or relative to the invocation location.

**#** outputs.**path** *directory* *(required)*

Desired output directory. Can be absolute or relative to the invocation location. If the directory does not exist it will be created.

**#** handlers.**[handler name]** *list of strings* *(required)*

Specification of the event handlers to be run during processing. Currently available handlers include:

* ``volume_counts``: Produce link volume counts and volume capacity ratios by time slice.

The associated list attached to each handler allows specification of which modes of transport should be processed using that handler. This allows certain handlers to be activated for public transport modes but not private vehicles for example. Possible modes currently include:

* ``car``

## Todo

**Priority**

* Implementation of proper mode detection. The mode of transport related to an event can only be determined based on the specific vehicle ID. In the case of the TfL mode, logic needs to be developed to separate private cars from public transport vehicles. The Melbourne model used a lookup table generated from the transit vehicles XML file - a similar approach could be implemented.
* Addition of additional handlers, including average link travel time, node boardings/alightings, etc. This will be relatively easy to achieve once sample outputs are available to test against - the logic already exists in old scripts. 
* Add tests and refactor to make testing easier! 

**Nice to have**

* More descriptive generated column headers in handler results. Right now column headers are simply numbers mapped to the particular time slice during the modelled day. 
* Introduction of a --verbose option for more descriptive terminal outputs.

## What does the name mean?
[Elara]("https://en.wikipedia.org/wiki/Elara_(moon)") is a moon of Jupiter. There's not much else interesting to say about it, other than that the wikipedia article states that it orbits 11-13 *gigametres* from the planet. Gigametres is a cool unit of measurement. 
