import json
from pprint import pprint
import pandas as pd
import sys
import geopandas
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='SEE')

# Add the arguments
parser.add_argument('--input',type=str,help='the path to the input trips_logs_all_trips.csv')
parser.add_argument('--output',type=str,help='the path to the desired output folder')
parser.add_argument('--testing',type=bool,default=False,help='Run on a smaller subset of the data')

args = parser.parse_args()

trips = pd.read_csv(args.input)

if args.testing == True:
    print("TESTING MODE")
    print("Running on subset of 5000")
    trips = trips[:5000]

print("{} trips loaded from {}".format(len(trips),args.input))

if not os.path.exists(args.output):
    print("Specified output director didn't exist: {}, creating...")
    os.makedirs(args.output)

# for each agent and each innovation, we want to define the dominantMode (by longest trip)
# this is crude, it's only picking mode of the longest trip. Should probably be cumulative, good enough for now

def get_dominant(group):
    group['dominantModeDistance'] = group['distance'].max()
    group['dominantTripMode'] = group.loc[group['distance'].idxmax(), 'mode'] # this might return a series if multiple same max values ?
    return group

trips = trips.groupby(['agent',"innovation_hash"]).apply(get_dominant)

# for each agent and each innovation, we want to define the dominantMode (by frequency)
trips['dominantModeFreq'] = trips.groupby(['agent','innovation_hash'])['mode'].transform('max')
trips = trips.drop_duplicates(subset=['agent','innovation_hash'],keep='first')

print("Writing all trips csv to {} ".format(args.output))
trips.to_csv(args.output + "/allTrips.csv")

# we use the innovation_hash to identify a summary, per agent
plans = trips.groupby(['agent',"innovation_hash","utility","dominantTripMode"],as_index=False)['mode','selected'].agg(lambda x: ','.join(x.unique()))

# add relative score to selected score, gives indication of relative proximity
# This is naive - we assume max score is selected, which sometimes it isn't
# good enough for now
plans['relativeDisUtilityToSelected'] = plans['utility'] - plans.groupby(['agent'])['utility'].transform('max')
plans['relativeDisUtilityToSelectedPerc'] = plans['relativeDisUtilityToSelected'] / plans['utility'] * -100.0

# version where we get selected score, not maximum score, when calculating relative score
# m=plans['selected']=="yes"
# plans.loc[plans['selected']=="yes",'relativeDisUtilityToSelected'] = plans.loc[plans['selected']=="yes",'utility'] - plans[plans['selected']=="yes"].groupby('agent')['utility'].transform('max').values
# plans.loc[m,'relativeDisUtilityToSelected'] = plans.loc[m,'utility']- plans[m].groupby('agent')['utility'].transform('max')

# We find the home locations, based on the origin activity (home)
# we use home locations for visulisation purposes
homes = trips[trips.o_act=="home"][['ox','oy','agent']]
homes.drop_duplicates(subset=['agent'],keep='first')

# merge this table into the plans, giving us the ox and oy
plans = plans.merge(homes,on='agent',how='left')

# geopandas the df
gdf = geopandas.GeoDataFrame(plans, geometry=geopandas.points_from_xy(plans.ox, plans.oy))

# British east/northing
gdf.crs = {'init': 'epsg:27700'}

# re-project to 4326
gdf['geometry'] = gdf['geometry'].to_crs(epsg=4326)

# sort by utility
gdf = gdf.sort_values("utility")
# flatten, one row per innovation (removes duplicates from lazy processes above)
gdf = gdf.drop_duplicates(subset=['innovation_hash'],keep='first')

# Kepler weirdness. Moving from yes/no (matsim lingo) to a bool for whether or not it was selected
# enables this to be toggled on/off on kepler
gdf['selected'] = gdf['selected'].map({'yes':True ,'no':False})

# let's sort and label them in order (i.e. 1st (selected),  2nd (least worst etc), per plan
gdf = gdf.sort_values('utility')
gdf['scoreRank'] = gdf.groupby(['agent'])['utility'].rank(method='dense',ascending=False).astype(int)

# subselecting them into 2 different dfs
selectedPlans = gdf[gdf.selected==True]
unSelectedPlans = gdf[gdf.selected==False]

# Often we will have time/route innovations across n innovation strategies
# Since we care about mode shift, we can pick the best innovation per mode. 
# this is the 'best' option for a given mode
# since we have sorted based on utility, we can remove duplicates 

unSelectedPlans = unSelectedPlans.sort_values("utility")
unSelectedPlans = unSelectedPlans.drop_duplicates(['agent','mode'], keep='last')

# zip them back together again
gdf = pd.concat([selectedPlans,unSelectedPlans])

# kepler'able geojson
print("Writing all plans GeoJSON to {} ".format(args.output))
gdf.to_file(args.output + "/AllPlans.geojson", driver="GeoJSON")

# creation of a df where car is selected
# but PT exists in their unchosen plans
# "mode shift opportunity" gdf
PlanAgentsSel = gdf[gdf.selected==True]
carPlanAgentsSel = PlanAgentsSel[PlanAgentsSel.dominantTripMode=='car']

unSelectedPlansCarSelected = unSelectedPlans[unSelectedPlans.agent.isin(carPlanAgentsSel.agent.unique())]
unSelectedPlansCarSelected = unSelectedPlans[unSelectedPlans.dominantTripMode.isin(['bus','rail'])]
print("Writing CarModeShiftOpportunities GeoJSON to {} ".format(args.output))
unSelectedPlansCarSelected.to_file(args.output +  "/CarModeShiftOpportunities.geojson",driver='GeoJSON')