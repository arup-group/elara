import lxml.etree as etree


def get_vehicle_capacity_from_vehicles_xml_file(vehicle_file_path, vehicle_type):
    root_elem = etree.parse(vehicle_file_path).getroot()
    schema_version = root_elem.attrib['{http://www.w3.org/2001/XMLSchema-instance}schemaLocation'].split()[-1]
    vehicle_elem = root_elem.find('{{*}}vehicleType[@id="{}"]'.format(vehicle_type))
    return schema_version, vehicle_elem, get_vehicle_capacity_from_xml_element(vehicle_elem, schema_version)


def get_vehicle_capacity_from_xml_element(vehicle_type_elem, schema_version):
    capacity_elem = vehicle_type_elem.find("{*}capacity")
    if schema_version.endswith('v1.0.xsd'):
        seated_capacity = int(capacity_elem.find("{*}seats").attrib['persons'])
        standing_capacity = int(capacity_elem.find("{*}standingRoom").attrib['persons'])
    else:
        seated_capacity = int(capacity_elem.attrib['seats'])
        standing_capacity = int(capacity_elem.attrib['standingRoomInPersons'])
    return seated_capacity + standing_capacity
