import json
from pprint import pprint
import pandas as pd
import sys
import geopandas
import numpy as np

# trips = pd.read_csv("./elara/20210525_01p/trip_logs_all_trips.csv")
# trips = pd.read_csv("./elara/20210526_10p/trip_logs_all_trips.csv")
trips = pd.read_csv("./elara/20210609_10p/experiment3_no_car_100times_cost/trip_logs_all_trips.csv")

# simplify for now
# trips = trips[:20000]

print("{} trips loaded".format(len(trips)))

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

trips.to_csv("./out/allTrips.csv")

# we use the innovation_hash to identify a summary, per agent
plans = trips.groupby(['agent',"innovation_hash","utility","dominantTripMode"],as_index=False)['mode','selected'].agg(lambda x: ','.join(x.unique()))

# add relative score to selected score, gives indication of relative proximity
# version where we get selected, not maximum score
# m=plans['selected']=="yes"
# plans.loc[plans['selected']=="yes",'relativeDisUtilityToSelected'] = plans.loc[plans['selected']=="yes",'utility'] - plans[plans['selected']=="yes"].groupby('agent')['utility'].transform('max').values
# plans.loc[m,'relativeDisUtilityToSelected'] = plans.loc[m,'utility']- plans[m].groupby('agent')['utility'].transform('max')

# This is wrong - we assume max score is selected, which it often isn't.
# good enough for now
plans['relativeDisUtilityToSelected'] = plans['utility'] - plans.groupby(['agent'])['utility'].transform('max')
plans['relativeDisUtilityToSelectedPerc'] = plans['relativeDisUtilityToSelected'] / plans['utility'] * -100.0

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

gdf = gdf.sort_values("utility")
gdf = gdf.drop_duplicates(subset=['innovation_hash'],keep='first')

# Kepler weirdness. Moving from yes/no (matsim lingo) to a bool for whether or not it was selected
gdf['selected'] = gdf['selected'].map({'yes':True ,'no':False})

# let's sort and label them in order (i.e. 1st (selected),  2nd (least worst etc)
gdf = gdf.sort_values('utility')
gdf['scoreRank'] = gdf.groupby(['agent'])['utility'].rank(method='dense',ascending=False).astype(int)

gdf.to_csv("./tmp.csv")

selectedPlans = gdf[gdf.selected==True]
unSelectedPlans = gdf[gdf.selected==False]

# Often we will have time/route innovations across n innovation strategies
# Since we care about mode shift, we can pick the best innovation per mode. 
# this is the 'best' option for a given mode
# since we have sorted based on utility, we can remove duplicates 

unSelectedPlans = unSelectedPlans.sort_values("utility")
unSelectedPlans = unSelectedPlans.drop_duplicates(['agent','mode'], keep='last')

gdf = pd.concat([selectedPlans,unSelectedPlans])

# kepler'able geojson
gdf.to_file("./out/AllPlans.geojson", driver="GeoJSON")

PlanAgentsSel = gdf[gdf.selected==True]
carPlanAgentsSel = PlanAgentsSel[PlanAgentsSel.dominantTripMode=='car']

unSelectedPlansCarSelected = unSelectedPlans[unSelectedPlans.agent.isin(carPlanAgentsSel.agent.unique())]
unSelectedPlansCarSelected = unSelectedPlans[unSelectedPlans.dominantTripMode.isin(['bus','rail'])]
unSelectedPlansCarSelected.to_file("./out/CarModeShiftOpportunities.geojson",driver='GeoJSON')


# aggregate all innovations
# per lsoa?
# per income group?
# per mosaic data?


# let's remove those selected, keep only those not selected
# gdfUnSelected = gdf[gdf['selected']==False]

# # let's order these unselected by their utility
# # let's keep the best (unselected closest plan)
# gdfUnSelectedTop = gdfUnSelected.sort_values('utility').drop_duplicates(['agent'],keep='last')

# # write to geojson
# gdfUnSelectedTop.to_file("./out/topUnselected.geojson",driver='GeoJSON')

# # let's find plans where they have an unselected pt (bus and or rail) option
# pt_misses = gdfUnSelectedTop[(gdfUnSelectedTop['mode'].str.contains("rail|bus"))]

# # write to geojson
# pt_misses.to_file("./out/topUnselectedPT.geojson",driver="GeoJSON")