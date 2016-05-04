import matplotlib, os, fiona, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
from pysal.esda.mapclassify import Natural_Breaks as nb
from descartes import PolygonPatch
from itertools import chain

DUBLIN_DED = '../../data/interim/DublinDED_2011/dublin_ed'

def build_basemap():
    shapefile = DUBLIN_DED
    shp = fiona.open(shapefile+".shp")
    bds = shp.bounds
    shp.close()
    extra = 0.01
    ll = (bds[0], bds[1])
    ur = (bds[2], bds[3])
    coords = list(chain(ll, ur))
    w, h = coords[2] - coords[0], coords[3] - coords[1]

    m = Basemap(
    projection='tmerc',
    lon_0=-2.,
    lat_0=49.,
    ellps = 'WGS84',
    llcrnrlon=coords[0] - extra * w,
    llcrnrlat=coords[1] - extra + 0.01 * h,
    urcrnrlon=coords[2] + extra * w,
    urcrnrlat=coords[3] + extra + 0.01 * h,
    lat_ts=0,
    resolution='i',
    suppress_ticks=True)

    m.readshapefile(
        shapefile,
        'dublin',
        color='none',
        zorder=2)

    return m, coords


# Convenience functions for working with colour ramps and bars
def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    This is a convenience function to stop you making off-by-one errors
    Takes a standard colour ramp, and discretizes it,
    then draws a colour bar with correctly aligned labels
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar

def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)

    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

def fixed_breaks(df_map, field):
    bins = pd.cut(df_map[field]*100, bins=[-1, 20, 40, 60, 80, 100])
    fb = pd.DataFrame({'bins': bins}, index=df_map[df_map[field].notnull()].index)
    #print(fb)
    df_map['bins'] = [int(re.search(", (\d{2,3})]$", v).group(1)) for v in fb['bins']]
    uppers = df_map.groupby('bins')[field].count()
    fixed_labels = ["  {}-{}% of units ({} EDs)".format(upper_bound - 20, upper_bound, count) for upper_bound, count in zip(uppers.keys(), uppers.values)]
    fixed_labels[0] = fixed_labels[0].replace("  0-", "<= ")
    return df_map, fixed_labels

def jenks_breaks(df_map, field):
    # Calculate Jenks natural breaks for density
    breaks = nb(
        df_map[df_map[field].notnull()][field].values,
        initial=300,
        k=5)
    # the notnull method lets us match indices when joining
    jb = pd.DataFrame({'bins': breaks.yb}, index=df_map[df_map[field].notnull()].index)
    df_map = df_map.join(jb)
    df_map.jenks_bins.fillna(-1, inplace=True)

    jenks_labels = ["up to %0.f%% (%s EDs)" % (b*100, c) for b, c in zip(
        breaks.bins, breaks.counts)]
    #jenks_labels.insert(0, 'No plaques (%s wards)' % len(df_map[df_map['density_km'].isnull()]))
    return df_map, jenks_labels

def plot_map(m, coords, df_map, field, labels, year):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)

    # use a blue colour ramp - we'll be converting it to a map using cmap()
    cmap = plt.get_cmap('Blues')
    # draw wards with grey outlines
    df_map['patches'] = df_map['geometry'].map(lambda x: PolygonPatch(x, ec='#555555', lw=.2, alpha=1., zorder=4))
    pc = PatchCollection(df_map['patches'], match_original=True)
    # impose our colour map onto the patch collection
    norm = Normalize()
    pc.set_facecolor(cmap(norm(df_map['bins'].values)))
    ax.add_collection(pc)

    # Add a colour bar
    cb = colorbar_index(ncolors=len(labels), cmap=cmap, shrink=0.5, labels=labels)
    cb.ax.tick_params(labelsize=25)

    field_label = field.replace("pc_", "").replace("_", " ")

    # Show highest densities, in descending order
    highest = '\n'.join("{} ({:0.0f}%)".format(value['ed_name'], value[field]*100) for _, value in df_map.sort_values(field, ascending=False)[:5].iterrows())

    highest = 'Highest EDs by percentile:\n' + highest
    # Subtraction is necessary for precise y coordinate alignment
    details = cb.ax.text(
        -1., 0 - 0.007,
        highest,
        ha='right', va='bottom',
        size=30,
        color='#555555')

    '''
    # Bin method, copyright and source data info
    smallprint = ax.text(
        0.9, 0,
        'Classification method: fixed breaks\nContains Ordnance Survey Ireland data\n © OSi 2012\\nCensus data from http://cso.ie and http://opac.oireachtas.ie',
        ha='right', va='bottom',
        size=4,
        color='#555555',
        transform=ax.transAxes)
        '''
    # Draw a map scale
    m.drawmapscale(
        coords[0] + 0.05, coords[1] + 0.001,
        coords[0], coords[1],
        5.,
        barstyle='fancy',
        labelstyle='simple',
        fillcolor1='w',
        fillcolor2='#555555',
        fontcolor='#555555',
        zorder=5)
    # this will set the image width to (a multiple of!) 722px at 100dpi
    plt.tight_layout()
    fig.set_size_inches(14.44*3, 10.5*3)
    plt.title("Percentage of {}\nby Dublin electoral division, {}".format(field_label, year), fontsize=30)
    plt.savefig('../../data/media/dublin_{}_{}.svg'.format(year, field),
                format='svg',
                dpi=100,
                frameon=True,
                bbox_inches='tight',
                pad_inches=0.5,
                facecolor='#F2F2F2')

def main():
    df_map = pd.read_pickle("../../data/processed/dublin_ed_basemap.pickle")
    m, coords = build_basemap()
    basedir = "../../data/processed/tenure_pickles"
    for file in sorted(f for f in os.listdir(basedir) if f.startswith('private')):
        #print(file)
        field = "pc_"+file.split("_dublin")[0]
        year = file.split("_")[-1].replace(".pickle", "")
        print(field, year)
        d_df = pd.read_pickle(os.path.join(basedir, file))
        #jk_map, labels = jenks_breaks(df_map.merge(d_df), field)
        jk_map, labels = fixed_breaks(df_map.merge(d_df), field)
        print(labels)
        plot_map(m, coords, jk_map, field, labels, year)

if __name__ == '__main__':
    main()
