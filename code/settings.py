"""Settings for the oscillation methods project."""

from copy import deepcopy
from fooof import Bands

###################################################################################################
###################################################################################################

# Define oscillations bands of interest
BANDS = Bands({'delta' : [2, 4],
               'theta' : [4, 8],
               'alpha' : [8, 13],
               'beta' : [13, 30]})

BANDS_FULL = Bands({'delta' : [2, 4],
                    'theta' : [4, 8],
                    'alpha' : [8, 13],
                    'beta' : [13, 30],
                    'gamma' : [30, 80]})

ALPHA_RANGE = (8, 12)


# Define band colors, for shading
BAND_COLORS = {'delta' : '#e8dc35',
               'theta' : '#46b870',
               'alpha' : '#1882d9',
               'beta'  : '#a218d9',
               'gamma' : '#e60026'}

ALPHA_COLOR = BAND_COLORS['alpha']
INDIV_COLOR = '#00a645'

# Define CONDITION COLORS
C1 = '#0000ff'
C2 = '#00693a'
COND_COLORS = [C1, C2]

# Figure save format
PLT_EXT = 'pdf'