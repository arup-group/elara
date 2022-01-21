# exit when any command fails
set -e

# Script that moves data from columbus to elara for post-processing

# Get version of matsim run from command line arguments

while getopts v:c:d: flag
do
    case "${flag}" in
        v) VERSION=${OPTARG};;
        c) CONTAINER=${OPTARG};;
        d) VOLUME=${OPTARG};;
    esac
done

# Check that version exists otherwise set it to v0_0_1
VERSION=${VERSION:-"v0_0_1"}
# Check that docker container name exists
CONTAINER=${CONTAINER:-"cambridge_${VERSION}_my_cont"}
# Check that docker volume name exists
VOLUME=${VOLUME:-"/mnt/${VERSION}/RunMultimodalMatsim_outputs/"}

# Create output file
OUTPUT_FILE="../elara/cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs"
mkdir -p $OUTPUT_FILE

# Create output dir for elara outputs
OUTPUT_DIR="../elara/cambridge-abm/${VERSION}/elara_outputs/"
mkdir -p $OUTPUT_DIR


# Go to columbus directory
cd ..


# Check if columbus directory exists
if [ -d "./columbus/" ]; then
  cd ./columbus/
else
  echo 'columbus/ directory does not exist!'
  exit 1
fi

echo "Copying data from container volume to local directory"

#${OUTPUT_FILE}
# Extract data from docker volume
docker cp $CONTAINER:$VOLUME $OUTPUT_FILE

# Go back to elara directory
cd ../elara

echo "Unzipping MATSim outputs"

# Unzip files that are needed for Elara
gzip --force -dk "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_plans.xml.gz" \
              "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_vehicles.xml.gz" \
              "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_transitVehicles.xml.gz" \
              "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_transitSchedule.xml.gz" \
              "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_network.xml.gz" \
              "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_events.xml.gz" \
              "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_experienced_plans.xml.gz"

# Move files
mv  "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_plans.xml" "./cambridge-abm/${VERSION}/matsim_outputs/output_plans.xml"
mv  "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_vehicles.xml" "./cambridge-abm/${VERSION}/matsim_outputs/output_vehicles.xml"
mv "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_transitVehicles.xml" "./cambridge-abm/${VERSION}/matsim_outputs/output_transitVehicles.xml"
mv "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_transitSchedule.xml" "./cambridge-abm/${VERSION}/matsim_outputs/output_transitSchedule.xml"
mv "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_network.xml" "./cambridge-abm/${VERSION}/matsim_outputs/output_network.xml"
mv "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_events.xml" "./cambridge-abm/${VERSION}/matsim_outputs/output_events.xml"
mv "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_config.xml" "./cambridge-abm/${VERSION}/matsim_outputs/output_config.xml"
mv "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs/RunMultimodalMatsim_outputs/output_experienced_plans.xml" "./cambridge-abm/${VERSION}/matsim_outputs/output_experienced_plans.xml"

# Remove directory with copied data from columbus
rm -rf "./cambridge-abm/${VERSION}/matsim_outputs/${VERSION}_outputs"

echo "Done"

exit 0
