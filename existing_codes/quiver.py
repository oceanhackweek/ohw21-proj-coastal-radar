#!/usr/bin/env python
"""
Adapt Mike Smith's code to
  call with arguments
  filter display vectors with GDOP or OI error thresholds
  work for Rutgers OI, Rutgers UWLS and National Network NetCDF files
@author Mike Smith / Teresa Updyke modified
@email michaesm@marine.rutgers.edu
@purpose  Plot NetCDF totals with quiver
"""

import xarray as xr
import numpy.ma as ma
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from codar_processing.src.common import create_dir
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from oceans.ocfis import uv2spdir, spdir2uv


def main(file, save_dir, regions):
    """
    Quiver plot for HFR NetCDF files
    :param file: NetCDF filename including path
    :param save_dir: Directory to save image files to
    :param regions: Dictionary of regions names and extents (lat/lon limits)
    """

    velocity_min = 0
    velocity_max = 60

    ds = xr.open_mfdataset(file)
    ds.load()
    nn = 0
    if ds.creator_name[0:10] == 'Mark Otero':
        nn = 1


    if ds.summary[0:24] == 'Unweighted Least Squares' or nn:
        error_threshold = 1.25
    else:
        error_threshold = 0.6

    LAND = cfeature.NaturalEarthFeature(
        'physical', 'land', '10m',
        edgecolor='face',
        facecolor='tan'
    )

    state_lines = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    fig = plt.figure()

    for t in ds.time.data:
        #if nn:
            #temp = ds.sel(time=t)
        #else:
             #temp = ds.sel(time=t, z=0)  # Works with CF/NCEI compliant netCDF files created in 2018
             #temp = ds.sel(time=t)
        #temp = temp.squeeze()
        temp = ds.squeeze()

        timestamp = pd.Timestamp(t).strftime('%Y%m%dT%H%M%SZ')

        for key, values in regions.items():
            extent = values
            tds = temp.sel(
                lon=(temp.lon > extent[0]) & (temp.lon < extent[1]),
                lat=(temp.lat < extent[3]) & (temp.lat > extent[2]),
            )
            if nn:
                tds = tds.where(
                    (tds.DOPx < error_threshold))

                tds = tds.where(
                    (tds.DOPy < error_threshold))

                u = tds['u'].data*100
                v = tds['v'].data*100

            else:

                if "qc_primary_flag" in tds.data_vars:

                    tds = tds.where(
                    (tds.qc_primary_flag < 4))

                    u = tds['u'].data*100
                    v = tds['v'].data*100
                else:
                    tds = tds.where(
                    (tds.u_err < error_threshold))

                    tds = tds.where(
                    (tds.v_err < error_threshold))

                    u = tds['u'].data
                    v = tds['v'].data


            lon = tds.coords['lon'].data
            lat = tds.coords['lat'].data
            time = tds.coords['time'].data

            u = ma.masked_invalid(u)
            v = ma.masked_invalid(v)

            angle, speed = uv2spdir(u, v)
            us, vs = spdir2uv(np.ones_like(speed), angle, deg=True)

            lons, lats = np.meshgrid(lon, lat)

            speed_clipped = np.clip(speed, velocity_min, velocity_max).squeeze()

            fig, (ax) = plt.subplots(figsize=(11, 8),
                                     subplot_kw=dict(projection=ccrs.PlateCarree()))


            #ax.set_xticks(np.arange(extent[0], extent[1], 0.5))
            #ax.set_yticks(np.arange(extent[2], extent[3], 0.5))
            #plt.grid(b=True, which='major', color='#666666', linestyle='--')

            # plot pcolor on map
            h = ax.pcolormesh(lons, lats,
                          speed_clipped,
                          vmin=velocity_min,
                          vmax=velocity_max,
                          cmap='jet')

            #h = ax.imshow(speed_clipped,
            #h = ax.imshow(speed,
            #              vmin=velocity_min,
            #              vmax=velocity_max,
            #              cmap='jet',
            #              interpolation='bilinear',
            #              extent=extent,
            #              origin='lower')

            # plot arrows over pcolor
            ax.quiver(lons, lats,
                      us, vs,
                      cmap='jet',
                      scale=55)

            # generate colorbar
            #plt.colorbar(h)
            cbar = plt.colorbar(h)
            cbar.ax.set_ylabel('Speed (cm/s)', fontsize = 20)
            cbar.ax.tick_params(labelsize=20)

            # plot reference point for Outer Banks
            #plt.plot([-75.0602960], [35.760360], marker='s', markersize=12, markeredgewidth=2, markerfacecolor='none', markeredgecolor='w')

            # Plot title
            #plt.title('{}\n{} - {} {}'.format(tds.title, ' '.join(key.split('_')), timestamp, error_threshold))
            plt.title('{}\n{} - {}'.format(tds.title, ' '.join(key.split('_')), timestamp))
            #plt.title('{}\n{}'.format('Surface Currents - V1 Processing', 'May 21 2021, 07:00 UTC'), fontsize = 20)

            #plt.show()
            # Gridlines and grid labels

            gl = ax.gridlines(draw_labels=True,
                               linewidth=1,
                                color='black',
                                alpha=0.5, linestyle='--')
            gl.xlocator = mticker.FixedLocator(np.arange(np.floor(extent[0]), np.ceil(extent[1])+1, 1))
            gl.ylocator = mticker.FixedLocator(np.arange(np.floor(extent[2]), np.ceil(extent[3])+1, 1))
            gl.xlabels_top = gl.ylabels_right = False
            gl.xlabel_style = {'size': 20, 'color': 'black'}
            gl.ylabel_style = {'size': 20, 'color': 'black'}
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER

            # Axes properties and features
            ax.set_extent(extent)
            ax.add_feature(LAND, zorder=0, edgecolor='black')
            ax.add_feature(cfeature.LAKES)
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(state_lines, edgecolor='black')

            # note = "Note: Filtered to exclude totals with Uerr and Verr Uncertainties < " + str(error_threshold)
            #ax.text(0.5, -1, 'Test', va='bottom', ha='center',
            #        rotation='horizontal', rotation_mode='anchor',
            #        transform=ax.transAxes)
            #plt.show()
            save_path = os.path.join(save_dir, key)
            create_dir(save_path)

            fig_name = '{}/MARACOOS_{}_{}_totals.png'.format(save_path, key, timestamp)
            fig_size = plt.rcParams["figure.figsize"]
            #fig_size[0] = 12
            #fig_size[1] = 8.5
            fig_size[0] = 6
            fig_size[1] = 4.25

            plt.rcParams["figure.figsize"] = fig_size

            plt.savefig(fig_name, dpi=150)
            plt.close('all')
            #plt.show()


if __name__ == '__main__':
    import os
    import glob

    # Define test inputs
    fns = glob.glob(os.path.join('/Users/teresa/Desktop/AnnotationExamples/nc/*.nc'))
    save_dir = '/Users/teresa/Desktop/AnnotationExamples/plots/'
    regions = dict(OuterBanks=[-76.5, -73, 34, 37], Massachusetts=[-70.5, -68, 40, 43],
                   Rhode_Island=[-72, -70, 39, 41.5],
                   New_Jersey=[-75, -72, 39, 41], EasternShores=[-76, -72, 37, 39])

    for file in fns:
        main(file, save_dir, regions)