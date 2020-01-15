import sys
import time

import geojson
import pyproj
from lxml import etree


def filter_matching_legs(plans_xml_path, mode):
    print("Parsing plans file....")
    plans_xml = etree.parse(plans_xml_path)
    people = plans_xml.findall('person')
    print('Found {} people with plans'.format(len(people)))
    matching_legs = []
    for person in people:
        person_id = person.get('id')
        plans = person.findall('plan')
        print("\tPerson {} has {} plans".format(person_id, len(plans)))
        for plan in plans:
            if plan.get('selected') == 'yes':
                legs = plan.findall('leg')
                print('\t\tPlan has {} legs'.format(len(legs)))
                for leg in legs:
                    if leg.get('mode') == mode:
                        print("\t\t\tFound a leg matching mode '{}'".format(mode))
                        matching_legs.append(
                            {
                                'person_id': person_id,
                                'plan_score': plan.get('score'),
                                'leg': leg,
                                'activity': leg.getnext()
                            })
    return matching_legs


def leg_as_tsv(leg_dict, link_database, node_database, crs_transformer):
    tsv_line = ''
    tsv_line += leg_dict['person_id']
    tsv_line += '\t'
    tsv_line += leg_dict['plan_score']
    tsv_line += '\t'
    tsv_line += leg_dict['activity'].get('type')
    tsv_line += '\t'
    tsv_line += leg_dict['leg'].get('mode')
    tsv_line += '\t'
    tsv_line += leg_dict['leg'].get('dep_time')
    tsv_line += '\t'
    tsv_line += leg_dict['leg'].get('trav_time')
    tsv_line += '\t'

    route_element = leg_dict['leg'].find('route')
    tsv_line += route_element.get('distance')
    tsv_line += '\t'
    tsv_line += make_polyline(route_element.text.split(' '),
                              link_database,
                              node_database,
                              crs_transformer)
    return tsv_line + '\n'


def make_polyline(route_point_ids, link_database, node_database, crs_transformer):
    print("\t\tMaking polyline from {} points".format(len(route_point_ids)))
    route_points = []
    for route_point in route_point_ids:
        from_node_id = link_database.get(route_point).get('from')
        from_x = node_database.get(from_node_id).get('x')
        from_y = node_database.get(from_node_id).get('y')
        from_long, from_lat = crs_transformer.transform(from_x, from_y)
        route_points.append(geojson.Point((from_long, from_lat)))
        to_node_id = link_database.get(route_point).get('to')
        to_x = node_database.get(to_node_id).get('x')
        to_y = node_database.get(to_node_id).get('y')
        to_long, to_lat = crs_transformer.transform(to_x, to_y)
        route_points.append(geojson.Point((to_long, to_lat)))
    return geojson.dumps(geojson.LineString(route_points))


def extract_link_ids(leg_elements):
    link_ids = []
    for leg_dict in leg_elements:
        route_element = leg_dict['leg'].find('route')
        route_point_ids = route_element.text.split(' ')
        link_ids += route_point_ids
    return set(link_ids)


def make_link_table(link_ids, matsim_output_network_path):
    link_context = etree.iterparse(matsim_output_network_path, events=('end',), tag='link')
    link_table = {}
    relevant_nodes_ids = []
    for event, elem in link_context:
        if elem.get("id") in link_ids:
            from_node_id = elem.get("from")
            to_node_id = elem.get("to")
            relevant_nodes_ids += [from_node_id, to_node_id]
            link_table[elem.get("id")] = {"from": from_node_id, "to": to_node_id}
    return link_table, set(relevant_nodes_ids)


def make_node_table(link_node_ids, matsim_output_network_path):
    node_context = etree.iterparse(matsim_output_network_path, events=('end',), tag='node')
    node_table = {}
    for event, elem in node_context:
        if elem.get("id") in link_node_ids:
            node_table[elem.get("id")] = {"x": elem.get("x"), "y": elem.get("y")}
    return node_table


if __name__ == '__main__':
    start_time = time.time()

    matsim_output_plans_path = sys.argv[1]
    matsim_output_network_path = sys.argv[2]
    mode = sys.argv[3]
    output_path = sys.argv[4]

    print("\nExtracting all '{}' legs from MATSim output plans file at {} using network file at {} for reference"
          .format(mode, matsim_output_plans_path, matsim_output_network_path))
    matching_leg_elements = filter_matching_legs(matsim_output_plans_path, mode)
    print("\tFound {} legs with mode {}".format(len(matching_leg_elements), mode))
    link_ids = extract_link_ids(matching_leg_elements)
    print("\tFound {} link IDs across all matching legs".format(len(link_ids)))

    print("Reading in network from {}...".format(matsim_output_network_path))

    link_database, link_node_ids = make_link_table(link_ids, matsim_output_network_path)
    print("\tCreated link database with {} links".format(len(link_database)))
    print("\tFound {} node IDs referenced by links of interest".format(len(link_node_ids)))

    node_database = make_node_table(link_node_ids, matsim_output_network_path)
    print("\tCreated node database with {} nodes".format(len(node_database)))

    print("Finished reading in network from {}".format(matsim_output_network_path))

    print("Creating output TSV file at {}".format(output_path))
    with open(output_path, 'w') as out_file:
        out_file.write('person_id\t'
                       'plan_score\t'
                       'activity\t'
                       'mode\t'
                       'departure_time\t'
                       'travel_time\t'
                       'distance\t'
                       'polyline\n')
        leg_count = 1
        epsg_27700 = pyproj.Proj(init='epsg:27700')
        epsg_4326 = pyproj.Proj(init='epsg:4326')
        geometry_transformer = pyproj.Transformer.from_proj(epsg_27700, epsg_4326)
        for leg in matching_leg_elements:
            print('\tExporting leg {} : {}'.format(leg_count, leg))
            out_file.write(leg_as_tsv(leg, link_database, node_database, geometry_transformer))
            leg_count += 1

    print("\nFinished exporting plan data in {} seconds".format(time.time() - start_time))
