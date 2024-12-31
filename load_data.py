import numpy as np
import pandas as pd
import xarray as xr
import validators
import glob2 as glob
from erddapy import ERDDAP
from siphon.catalog import TDSCatalog

def download_erddap(
    url: str,
    dataset_id: str,
    start_time: str,
    end_time: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    variables: list,
) -> pd.DataFrame:
    """Connects to ERDDAP, loads geospatial data into a pandas dataframe with
    time as the index.  Note that often you may need to query across a longitude
    of -180 using two separate queries using different lon_min and lon_max.

    Parameters
    ----------
    url : str
        ERDDAP url, for example, "http://www.ifremer.fr/erddap"
    dataset_id : str
        Comes from the particular ERDDAP server's documentation/metadata
    start_time : str
        Start of time range for query in format '%Y-%m-%dT%H:%M:%S'
    end_time : str
        End of time range for query in format '%Y-%m-%dT%H:%M:%S'
    lon_min : float
    lon_max : float
    lat_min : float
    lat_max : float
    variables : list
        Variables to return from ERDDAP query, example, ['latitude','longitude',
        'time','instrument_serial_number','depth','temp','qc_flag']

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing the variables in variable_list as columns and with time as the index
    """
    e = ERDDAP(server=url, protocol="tabledap")
    e.response = "nc"
    e.dataset_id = dataset_id
    e.variables = variables

    e.constraints = {
        "latitude>=": lat_min,
        "latitude<=": lat_max,
        "longitude>=": lon_min,
        "longitude<=": lon_max,
        "time>=": start_time,
        "time<=": end_time,
    }
    df = e.to_pandas(parse_dates=["time (UTC)"], index_col="time (UTC)").dropna()

    return df

def load_argo(
    url: str = "http://www.ifremer.fr/erddap",
    dataset_id: str = "ArgoFloats",
    start_date: np.datetime64 = np.datetime64("2024-01-01"),
    end_date: np.datetime64 = np.datetime64("2024-12-31"),
    box: list =[161, 190, -52, -31],
    variables: list = ["latitude", "longitude", "time", "float_serial_no", "pres"],
) -> pd.DataFrame:

    dmin = pd.to_datetime(start_date).strftime("%Y-%m-%dT%H:%M:%S")
    dmax = pd.to_datetime(end_date).strftime("%Y-%m-%dT%H:%M:%S")

    # Reformat "box" for use with ERDDAP, i.e. break into
    # two boxes on either side of the international dateline
    # if needed.

    if box[1] > 180:
        box1 = list(box)
        box2 = list(box)
        box1[1] = 180
        box2[0] = -180
        box2[1] = (box[1] + 180) % 360 - 180
    else:
        box1 = box
        box2 = False

    # one side of the international date line
    argo = download_erddap(
        url,
        dataset_id,
        start_time=dmin,
        end_time=dmax,
        lon_min=box1[0],
        lon_max=box1[1],
        lat_min=box1[2],
        lat_max=box1[3],
        variables=variables,
    )

    # other side of the international date line, if needed
    if box2:
        argo_1 = download_erddap(
            url,
            dataset_id,
            start_time=dmin,
            end_time=dmax,
            lon_min=box2[0],
            lon_max=box2[1],
            lat_min=box2[2],
            lat_max=box2[3],
            variables=variables,
        )

        # put the two sides of the international date line together
        argo = pd.concat([argo, argo_1])
        argo = argo.rename(
            columns={
                "latitude (degrees_north)": "lat",
                "longitude (degrees_east)": "lon",
            }
        )

        # convert longitude to 0-360
        argo["lon"] = argo["lon"] % 360

    return argo

def load_moana_tds(
    source: str = "http://thredds.moanaproject.org:6443/thredds/catalog/moana/Mangopare/public/catalog.html",
    start_date: np.datetime64 = np.datetime64("2024-01-01"),
    end_date: np.datetime64 = np.datetime64("2024-12-31")
) -> tuple[pd.DataFrame, dict, list]:
    """Loads public Mangōpare data from the Moana Project THREDDS server,
    or local directory, between start_date and end_date.  Calculates statistics
    including the number of measurements, max depth, and duration of each
    deployment.

    Parameters
    ----------
    source : str, optional
        THREDDS server url, by default "http://thredds.moanaproject.org:6443/thredds/catalog/moana/Mangopare/public/catalog.html"
        or directory to find files in, e.g., '/path_to_files/*.nc'
    start_date : np.datetime64, optional
        Start of desired date range, by default start_date
    end_date : np.datetime64, optional
        End of desired date range, by default end_date

    Returns
    -------
    tuple[pd.DataFrame, dict, list]
        Returns a dataframe of the initial latitude, longitude,
        and time of each deployment, a dictionary of the above statistics,
        and an array of the time of all measurements.
    """
    if validators.url(source):
        # load THREDDS catalog
        cat = TDSCatalog(source)
        filelist = sorted(cat.datasets)
    else:
        filelist = glob.glob(source)

    # initialise variables

    lat = []
    lon = []
    time = []
    deploy_time = []

    num_measurements = []
    max_depths = []
    durations = []

    for file in filelist:

        sdn = pd.to_datetime(file[6:14], format="%Y%m%d").to_numpy()
        if (sdn < start_date) or (sdn > end_date):
            continue
        if validators.url(source):
            ds = cat.datasets[file].remote_access(use_xarray=True)
        else:
            ds = xr.open_dataset(file)

        mask = ds["QC_FLAG"] < 4
        ds = ds.where(mask, drop=True)
        ds = ds.where(ds["TIME"] >= start_date, drop=True)
        ds = ds.where(ds["TIME"] <= end_date, drop=True)

        if len(ds.LATITUDE) < 1:
            ds.close()
            continue

        lat.append(float(ds.LATITUDE[0]))
        lon.append(float(ds.LONGITUDE[0]))
        deploy_time.append(ds.TIME[0].values)
        time.extend(ds.TIME.values)

        num_measurements.append(len(ds.TIME.values))
        max_depths.append(np.nanmax(ds.DEPTH.values))
        durations.append(np.nanmax(ds.TIME) - np.nanmin(ds.TIME))

        ds.close()

    moana_df = pd.DataFrame({"lat": lat, "lon": lon, "time": deploy_time}).dropna()
    moana_df["time"] = moana_df["time"].dt.tz_localize("UTC")

    stats_moana = {
        "num_measurements": num_measurements,
        "max_depths": max_depths,
        "durations": durations,
    }

    return moana_df, stats_moana, time

def load_moana_db(
    source: str = "sqlite:///moana.db",
    start_date: np.datetime64 = np.datetime64("2024-01-01"),
    end_date: np.datetime64 = np.datetime64("2024-12-31"),
    qc_flag_max: int = 3,
) -> tuple[pd.DataFrame, dict, list]:
    """Loads public Mangōpare data from the Moana Project THREDDS server,
    or local directory, between start_date and end_date.  Calculates statistics
    including the number of measurements, max depth, and duration of each
    deployment.

    Parameters
    ----------
    source : str, optional
        database file, by default "moana.db"
    start_date : np.datetime64, optional
        Start of desired date range, by default start_date
    end_date : np.datetime64, optional
        End of desired date range, by default end_date

    Returns
    -------
    tuple[pd.DataFrame, dict, list]
        Returns a dataframe of the initial latitude, longitude,
        and time of each deployment, a dictionary of the above statistics,
        and an array of the time of all measurements.
    """

    import sqlite3
    from sqlalchemy.engine import create_engine
    from sqlalchemy.engine.base import Engine

    disk_engine = create_engine(source)

    moana_df = pd.read_sql_query(
        "SELECT MIN(time) as time, AVG(lat) as lat, AVG(lon) as lon FROM observations WHERE qcflag<"
        + str(qc_flag_max)
        + " GROUP BY file",
        disk_engine,
    )

    time = pd.read_sql_query(
        "SELECT time FROM observations WHERE qcflag<" + str(qc_flag_max), disk_engine
    )

    # calculate number of measurements, maximum depth, and duration in hours of each deployment (or more accurately, file)
    stats_moana = pd.read_sql_query(
        "SELECT COUNT(*) as num_measurements, MAX(depth) as max_depth, CAST ((julianday(MAX(time),'utc')-julianday(MIN(time),'utc')) * 24 AS Float) as durations FROM observations WHERE qcflag<"
        + str(qc_flag_max)
        + " GROUP BY file",
        disk_engine,
    )

    moana_df["time"] = moana_df['time']=pd.to_datetime(moana_df['time'], format='%Y-%m-%d %H:%M:%S.000000', errors='coerce').dropna().dt.tz_localize("UTC")

    stats_moana = stats_moana.to_dict(orient="list")

    return moana_df, stats_moana, time