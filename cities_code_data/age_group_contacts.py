# Population age groups and contact matrices
# These are aggregated data for Italy extracted from the SOCRATES app: https://lwillem.shinyapps.io/socrates_rshiny/
# The data are formatted to be imported in python with a standard import statement such as "from age_group_contacts.py import *"
GROUP_NAMES  = ['children', 'young', 'adults', 'elderly']
AGE_GROUPS   = [18, 35, 65, 100] 
GROUP2GROUP  = {'children':{'children':10.3139591, 'young':1.4095328, 'adults':3.004145, 'elderly':0.3194044}, 'young':{'children':1.1164900, 'young':5.0501872, 'adults':2.671315, 'elderly':0.5556512}, 'adults':{'children':1.4129868, 'young':1.5862171, 'adults':3.777106, 'elderly':0.8300762}, 'elderly':{'children':0.2498358, 'young':0.5487018, 'adults':1.380431, 'elderly':1.2556003}}
GROUP_FREQ   = {'children':0.15, 'young':0.25, 'adults':0.4, 'elderly':0.2}
