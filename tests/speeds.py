import sys
import os
import timeit


sys.path.append(os.path.abspath('../elara'))
from elara.config import Config, PathFinderWorkStation
from elara import inputs
sys.path.append(os.path.abspath('../tests'))


# Config
config_path = os.path.join('tests/speed_gzip_scenario.toml')
config = Config(config_path)

# input paths
paths = PathFinderWorkStation(config)
paths.connect(managers=None, suppliers=None)
paths.load_all_tools()
paths.build()


# elem_parse_speed
# xml
print('> Get_elems speed tests:')
print('xml path:')
network_path = "./fixtures/fixtures_01perc/baseline/output_network.xml"
elems = inputs.get_elems(network_path, 'node')
start_time = timeit.default_timer()
for e in elems:
    pass
print('XML get_elems() for nodes: ', timeit.default_timer() - start_time)

# xml.gz
print('gzip path:')
network_path = "./fixtures/fixtures_01perc/baseline/output_network.xml.gz"
elems = inputs.get_elems(network_path, 'node')
start_time = timeit.default_timer()
for e in elems:
    pass
print('GZIP get_elems() for nodes: ', timeit.default_timer() - start_time)


print('\t>Test network build:')
# network load
start_time = timeit.default_timer()
network = inputs.Network(config)
print('Network init: ', timeit.default_timer() - start_time)

# network build no crs
print('post crs transform build:')
start_time = timeit.default_timer()
network.build(paths.resources)
print('Network no crs build: ', timeit.default_timer() - start_time)
print(len(network.link_gdf))
print(len(network.node_gdf))
