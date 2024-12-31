import numpy as np
import pandas as pd
import itertools
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#--------------------------------------------
## MAP-RELATED METHODS

def calc_grid_hist(
    x_coords: np.array, y_coords: np.array, time: np.array, x_edges: list, y_edges: list, res: float = 1
) -> dict:
    """Takes x,y (e.g., lon, lat) coordinates of each profile or
    sensor deployment and calculates the average monthly number
    of profiles/deployments per grid cell.  Grid cells are defined
    by x_edges and y_edges (e.g., longitude and latitude coordinates).

    Parameters
    ----------
    x_coords : np.array
        Measurements' x_coordinates (e.g., longitude)
    y_coords : np.array
        Measurements' y_coordinates (e.g., latitude)
    time : np.array
        Measurements' times
    x_edges : list
        x-coordinates of grid cell edges
    y_edges : list
        y-coordinates of grid cell edges

    Returns
    -------
    tuple[list, list, list]
        Returns average monthly counts per grid cell (h) and grid
        cell centers (x,y), i.e. (lon,lat)
    """
    df = (
        pd.DataFrame(
            data={"xc": x_coords, "yc": y_coords, "time": pd.to_datetime(time)}
        )
        .drop_duplicates()
        .set_index("time")
    )

    x2 = []
    y2 = []
    h2 = []

    for xa, ya in itertools.product(x_edges, y_edges):
        yb = ya + res
        xb = xa + res
        in_cell = df.loc[
            (df.yc >= ya) & (df.yc < yb) & (df.xc >= xa) & (df.xc < xb)
        ].dropna()
        ctm = in_cell.resample("MS")["yc"].count().fillna(0).mean()
        x2.append((xa + xb) / 2)
        y2.append((ya + yb) / 2)
        h2.append(ctm)

    x = np.array(x2)
    y = np.array(y2)
    h = np.array(h2)

    return {'x':x, 'y':y, 'h':h}


def plot_map(
    argo: pd.DataFrame,
    moana: pd.DataFrame,
    res: float = 1,
    ms: int = 38,
    box: list = [161, 190, -52, -31],
    bounds: list = [0, 4, 8, 12, 16, 24, 28],
    lon_offset: float = 180
    ) -> matplotlib.figure.Figure:
    """Creates two lat/lon maps of gridded data given a lat-lon box and
    both Argo and Mangōpare data as pandas dataframes.  Kind of a 
    rough way to do it because it just uses square scatter plot markers
    to show the grid.

    Parameters
    ----------
    argo: pandas dataframe
        contains argo data from load_data.py (load_argo)
    moana: pandas dataframe
        contains Mangōpare data from load_data.py (load_moana_tds or load_moana_db)
    res: float
        horizontal (lat and lon) resolution of geospatial grid cells in degrees
    ms: int
        matplotlib markersize for scatter plot
    box: list
        lat/lon coordinate box for map limits, [lon_min, lon_max, lat_min, lat_max]
    bounds: list
        colorbar bounds in sensor deployments per month
    lon_offset: float
        cartpy's lon_offset for centering the map

    Returns
    -------
    fig: matplotlib figure 
    """

    # calculate box edges
    x_edges = np.arange(box[0] - res, box[1] + res, res)
    y_edges = np.arange(box[2] - res, box[3] + res, res)    

    # calculate gridded histogram for moana and argo data
    moana_grid = calc_grid_hist(
        x_coords=moana["lon"].values,
        y_coords=moana["lat"].values,
        time=moana["time"].values,
        x_edges=x_edges,
        y_edges=y_edges,
        res=res
    )

    argo_grid = calc_grid_hist(
        x_coords=argo["lon"].values,
        y_coords=argo["lat"].values,
        time=argo.index,
        x_edges=x_edges,
        y_edges=y_edges,
        res=res
    )

    # now do the plotting
    plt.rcParams.update(plt.rcParamsDefault)
    # Set up plots
    fig, (ax0, ax1) = plt.subplots(
        nrows=1,
        ncols=2,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=lon_offset)},
        figsize=(9, 6),
        dpi=120,
        facecolor="w",
        edgecolor="k",
    )

    ## Panel 1

    # plot Argo data

    colors = plt.get_cmap("Blues")(np.linspace(0, 1, len(bounds) + 1))
    cmap = mcolors.ListedColormap(colors[1:-1])
    cmap.set_over(colors[-1])
    cmap.set_under(colors[0])
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds) - 1)
    sc2 = ax0.scatter(
        argo_grid['x'] + lon_offset, argo_grid['y'], c=argo_grid['h'], s=ms, marker="s", cmap=cmap, norm=norm
    )

    ax0.plot([box[0], box[0], box[1], box[1]], [box[2], box[3], box[3], box[2]])

    cb = plt.colorbar(sc2, ax=ax0, extend="both", orientation="horizontal", pad=0.1)
    cb.set_label("Number of Argo profiles")

    # plot properties
    ax0.set_extent(box, crs=ccrs.PlateCarree())
    ax0.coastlines(resolution="10m", facecolor="grey")
    land_10m = cfeature.NaturalEarthFeature(
        "physical", "land", "10m", edgecolor="black", facecolor=cfeature.COLORS["land"]
    )
    ax0.add_feature(land_10m)

    gl = ax0.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(range(-180, 180, 5))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 15, "color": "black"}
    gl.xlabel_style = {"color": "black"}

    ## Panel 2

    # plot Argo data

    colors = plt.get_cmap("Blues")(np.linspace(0, 1, len(bounds) + 1))
    cmap = mcolors.ListedColormap(colors[1:-1])
    cmap.set_over(colors[-1])
    cmap.set_under(colors[0])
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds) - 1)
    sc2 = ax1.scatter(
        argo_grid['x'] + lon_offset, argo_grid['y'], c=argo_grid['h'], s=ms, marker="s", cmap=cmap, norm=norm
    )

    # plot Mangopare data
    colors = plt.get_cmap("Oranges")(np.linspace(0, 1, len(bounds) + 1))
    cmap = mcolors.ListedColormap(colors[1:-1])
    cmap.set_over(colors[-1])
    cmap.set_under(colors[0])
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(bounds) - 1)
    sc = ax1.scatter(
        moana_grid['x'] + lon_offset, moana_grid['y'], c=moana_grid['h'], s=ms, marker="s", cmap=cmap, norm=norm
    )

    ax1.plot([box[0], box[0], box[1], box[1]], [box[2], box[3], box[3], box[2]])

    # Colorbars
    cb = plt.colorbar(sc, ax=ax1, extend="both", orientation="horizontal", pad=0.1)
    cb.set_label(r"Number of Moana deployments")

    # plot properties
    ax1.set_extent(box, crs=ccrs.PlateCarree())
    ax1.coastlines(resolution="10m", facecolor="grey")
    land_10m = cfeature.NaturalEarthFeature(
        "physical", "land", "10m", edgecolor="black", facecolor=cfeature.COLORS["land"]
    )
    ax1.add_feature(land_10m)

    gl = ax1.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(range(-180, 180, 5))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 15, "color": "black"}
    gl.xlabel_style = {"color": "black"}
    # fig.text(0.5, 0.83, r'Average Argo Events and Mang$\mathrm{\bar{o}}$pare Deployments:'+' {} through {}'.format(datemin,datemax,len(locs['lat'])),ha='center')
    # fig.text(0.5, 0.83, r'Monthly Average Argo Profiles and Sensor Deployments:'+' {} through {}'.format(datemin,datemax,len(locs['lat'])),ha='center')
    plt.show()
    return fig

#---------------------------------------------------------
## BAR PLOT METHODS

def calc_bar_data(
        argo: pd.DataFrame,
        moana: pd.DataFrame) -> pd.DataFrame:
    """Calculates monthly profiles (Argo) and sensor deployments (Moana) 
    and returns a single dataframe with Argo and Moana columns.

    Parameters
    ----------
    argo: pandas dataframe
        contains argo data from load_data.py (load_argo)
    moana: pandas dataframe
        contains Mangōpare data from load_data.py (load_moana_tds or load_moana_db)

    Returns
    -------
    df_comb: pd.DataFrame
    """

    dft_argo = (
        argo.drop(columns="pres (decibar)")
        .drop_duplicates()
        .reset_index()
        .rename(columns={"time (UTC)": "time"})
    )

    df_comb = pd.concat(
        [
            moana.groupby(pd.Grouper(key="time", freq="ME")).size(),
            dft_argo.groupby(pd.Grouper(key="time", freq="ME")).size(),
        ],
        axis=1,
    ).reset_index()
    df_comb = df_comb.rename(columns={0: "Mangōpare", 1: "Argo"}).set_index("time")
    df_comb = df_comb[["Argo", "Mangōpare"]].dropna()
    return df_comb

def monthly_bar_plot(        
        argo: pd.DataFrame,
        moana: pd.DataFrame) -> pd.DataFrame:
    """Plots monthly profiles (Argo) and sensor deployments (Moana).

    Parameters
    ----------
    argo: pandas dataframe
        contains argo data from load_data.py (load_argo)
    moana: pandas dataframe
        contains Mangōpare data from load_data.py (load_moana_tds or load_moana_db)

    Returns
    -------
    fig: matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    df_comb = calc_bar_data(argo,moana)

    df_comb.plot.bar(ax=ax)

    set_plot_params(ax)
    x_dates = [vals.strftime("%b-%y") for vals in df_comb.index]
    ax.set_xticklabels(labels=x_dates, rotation=90, ha="center")
    ax.set_ylabel("Number of Deployments")
    ax.set_xlabel("Date [mmm-yy]")

    return fig

def set_plot_params(
    ax: matplotlib.axes._axes.Axes) -> matplotlib.axes._axes.Axes:
    """Takes a matplotlib axis and applies specific formatting for 
    a cleaner look.

    Parameters
    ----------
    ax: matplotlib axis

    Returns
    -------
    ax: formatted axis
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")

    # Second, remove the ticks as well.
    ax.tick_params(bottom=False, left=False, right=False)

    # Third, add a horizontal grid (but keep the vertical grid hidden).
    # Color the lines a light gray as well.
    ax.set_axisbelow(True)

    ax.yaxis.grid(True, color="#EEEEEE")
    ax.xaxis.grid(False)
    return ax
