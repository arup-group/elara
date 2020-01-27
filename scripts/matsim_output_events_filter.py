import argparse
import sys
import time

import pyproj
import tqdm
from lxml import etree


def filter_vehicle_file(vehicle_file_path, vehicle_types):
    print('\tParsing vehicle file at {}, looking for vehicle types {}'.format(vehicle_file_path, vehicle_types))
    vehicle_context = etree.iterparse(vehicle_file_path, events=('end',),
                                      tag='{http://www.matsim.org/files/dtd}vehicle')
    vehicle_table = {}
    vehicle_count = 0
    for event, elem in vehicle_context:
        if elem.get('type') in vehicle_types:
            vehicle_id = elem.get('id')
            vehicle_type = elem.get('type')
            vehicle_table[vehicle_id] = vehicle_type
        vehicle_count += 1
    print('\t\tFound {} vehicles matching specified types from {} vehicles in total'.format(len(vehicle_table),
                                                                                            vehicle_count))
    return vehicle_table


def build_event_table(matsim_event_file_path, event_types, vehicle_ids):
    print('\tParsing event file at {}, looking for events for {} specified vehicles'.format(matsim_event_file_path,
                                                                                            len(vehicle_ids)))
    event_context = etree.iterparse(matsim_event_file_path, events=('end',), tag='event')
    event_table = {}
    event_count = 0
    matching_event_count = 0
    event_links = set([])
    for event, elem in event_context:
        event_count += 1
        # <event time="1.0" type="left link" vehicle="veh_6208_bus" link="187221"/>
        matsim_event_type = elem.get('type')
        vehicle_id = elem.get('vehicle')
        if matsim_event_type in event_types and vehicle_id in vehicle_ids:
            matching_event_count += 1
            event_link = elem.get('link')
            event_links.add(event_link)
            event_summary = {'t': matsim_event_type, 'l': event_link, 'ts': elem.get('time')}
            if vehicle_id in event_table.keys():
                event_table[vehicle_id].append(event_summary)
            else:
                event_table[vehicle_id] = [event_summary]
    print('\t\tFound {} events matching specified types and vehicle IDs from {} events in total'
          .format(matching_event_count, event_count))
    return event_table, event_links


def make_link_table(link_ids, matsim_output_network_path):
    print('\tCreating link table from network file at {}; looking for {} links of interest'
          .format(matsim_output_network_path, len(link_ids)))
    link_context = etree.iterparse(matsim_output_network_path, events=('end',), tag='link')
    link_table = {}
    relevant_nodes_ids = []
    for event, elem in tqdm.tqdm(link_context):
        if elem.get("id") in link_ids:
            from_node_id = elem.get("from")
            to_node_id = elem.get("to")
            relevant_nodes_ids += [from_node_id, to_node_id]
            link_table[elem.get("id")] = {"from": from_node_id, "to": to_node_id}
    return link_table, set(relevant_nodes_ids)


def make_node_table(node_ids, matsim_output_network_path):
    print('\tCreating node table from network file at {}; looking for {} nodes of interest'
          .format(matsim_output_network_path, len(node_ids)))
    node_context = etree.iterparse(matsim_output_network_path, events=('end',), tag='node')
    node_table = {}
    for event, elem in tqdm.tqdm(node_context):
        if elem.get("id") in node_ids:
            node_table[elem.get("id")] = {"x": elem.get("x"), "y": elem.get("y")}
    return node_table


def event_as_csv_line(vehicle_id, vehicle_type, event_summary_dict, link_database, node_database, crs_transformer):
    # vehicle ID, vehicle type, time, lat, lon
    csv_line = ''
    csv_line += vehicle_id
    csv_line += ','
    csv_line += vehicle_type
    csv_line += ','
    csv_line += event_summary_dict['ts']
    csv_line += ','

    link_id = event_summary_dict['l']
    matsim_event_type = event_summary_dict['t']
    closest_node = 'from'
    if matsim_event_type == 'left link':
        closest_node = 'to'
    node_id = link_database.get(link_id).get(closest_node)
    x = node_database.get(node_id).get('x')
    y = node_database.get(node_id).get('y')
    long, lat = crs_transformer.transform(x, y)
    csv_line += str(lat)
    csv_line += ','
    csv_line += str(long)

    return csv_line + '\n'


if __name__ == '__main__':
    start_time = time.time()

    arg_parser = argparse.ArgumentParser(description='Filter and transform events from a MATSim output events file')

    arg_parser.add_argument('-e',
                            '--events_file',
                            help='Full path to the MATSim event file location',
                            required=True)
    arg_parser.add_argument('-n',
                            '--network_file',
                            help='Full path to the MATSim output network file location',
                            required=True)
    arg_parser.add_argument('-o',
                            '--output_file',
                            help='Full path to create the filtered events file',
                            required=True)
    arg_parser.add_argument('-v',
                            '--vehicle_file',
                            action='append',
                            help='Optional. Full path to the MATSim vehicles config file. Multiple instances of this '
                                 'argument are allowed')
    arg_parser.add_argument('-t',
                            '--vehicle_type',
                            action='append',
                            help='Optional. Type of vehicle we are interested in. Multiple instances of this argument '
                                 'are allowed')
    args = vars(arg_parser.parse_args())

    events_file_path = args['events_file']
    network_file_path = args['network_file']
    output_file_path = args['output_file']
    vehicle_file_paths = args['vehicle_file']
    vehicle_types = args['vehicle_type']

    print('Filtering raw events from MATSim events file at {}, referencing links in network file {}.\nWill write output'
          ' to {}'.format(events_file_path, network_file_path, output_file_path))

    vehicle_lookup_table = {}
    if vehicle_types:
        if not vehicle_file_paths:
            print('!!! ERROR: You must supply paths to at least one vehicle file if you want to filter events based on '
                  'vehicle type')
            sys.exit(1)
        print('Filtering events to only vehicle types {} defined in vehicle files {}'.format(vehicle_types,
                                                                                             vehicle_file_paths))
        for vehicle_file in vehicle_file_paths:
            vehicle_lookup_table = {**vehicle_lookup_table, **filter_vehicle_file(vehicle_file, vehicle_types)}
        print('\tFound {} vehicle IDs matching specified types'.format(len(vehicle_lookup_table)))

    filtered_event_summaries, link_ids = build_event_table(events_file_path,
                                                           ["entered link"],
                                                           vehicle_lookup_table.keys())
    print("Got raw event summaries, transforming to CSV")

    link_table, node_ids = make_link_table(link_ids, network_file_path)
    node_table = make_node_table(node_ids, network_file_path)

    print("Creating output CSV file at {}".format(output_file_path))
    geometry_transformer = pyproj.Transformer.from_proj(pyproj.Proj(init='epsg:27700'), pyproj.Proj(init='epsg:4326'))
    with open(output_file_path, 'w') as out_file:
        out_file.write('vehicle ID,vehicle type,time,lat,lon\n')
        for vehicle in filtered_event_summaries.keys():
            vehicle_type = vehicle_lookup_table[vehicle]
            vehicle_events = filtered_event_summaries[vehicle]
            for vehicle_event in vehicle_events:
                out_file.write(event_as_csv_line(vehicle,
                                                 vehicle_type,
                                                 vehicle_event,
                                                 link_table,
                                                 node_table,
                                                 geometry_transformer))

    print("\nFinished data extraction in {} seconds".format(time.time() - start_time))
