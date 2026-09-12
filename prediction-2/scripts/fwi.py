"""Canadian Forest Fire Weather Index (FWI) System.

A direct numpy port of the reference implementation in the CRAN `cffdrs` R
package (Wang, Cantin, Parisien, Wotton, Anderson, Flannigan), which itself
implements Van Wagner & Pickett (1985) and Van Wagner (1987). Equation
numbers in comments refer to Van Wagner & Pickett (1985).

Reference source ported from:
https://github.com/cran/cffdrs/blob/master/R/fine_fuel_moisture_code.r
https://github.com/cran/cffdrs/blob/master/R/duff_moisture_code.r
https://github.com/cran/cffdrs/blob/master/R/drought_code.r
https://github.com/cran/cffdrs/blob/master/R/initial_spread_index.r
https://github.com/cran/cffdrs/blob/master/R/buildup_index.r
https://github.com/cran/cffdrs/blob/master/R/fire_weather_index.r
"""
import numpy as np
import pandas as pd

FFMC_COEFFICIENT = 250.0 * 59.5 / 101.0

# Day-length adjustment factor for DMC, by month (Jan..Dec)
_DMC_ELL_46N = np.array([6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6])
_DMC_ELL_20N = np.array([7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8])
_DMC_ELL_20S = np.array([10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2])
_DMC_ELL_40S = np.array([11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10, 11.2, 11.8])

# Day-length factor for DC, by month
_DC_FL_20N = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5, 2.4, 0.4, -1.6, -1.6])
_DC_FL_20S = np.array([6.4, 5, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8])

DEFAULT_STARTUP = {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}


def fine_fuel_moisture_code(ffmc_yda, temp, rh, ws, prec):
    ffmc_yda = np.asarray(ffmc_yda, dtype=float)
    temp = np.asarray(temp, dtype=float)
    rh = np.asarray(rh, dtype=float)
    ws = np.asarray(ws, dtype=float)
    prec = np.asarray(prec, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Eq. 1
        wmo = FFMC_COEFFICIENT * (101 - ffmc_yda) / (59.5 + ffmc_yda)
        # Eq. 2 - rain reduction for canopy interception
        ra = np.where(prec > 0.5, prec - 0.5, prec)
        # Eqs. 3a & 3b
        rain_gt150 = (
            wmo
            + 0.0015 * (wmo - 150) ** 2 * np.sqrt(ra)
            + 42.5 * ra * np.exp(-100 / (251 - wmo)) * (1 - np.exp(-6.93 / ra))
        )
        rain_le150 = wmo + 42.5 * ra * np.exp(-100 / (251 - wmo)) * (1 - np.exp(-6.93 / ra))
        wmo = np.where(prec > 0.5, np.where(wmo > 150, rain_gt150, rain_le150), wmo)
        wmo = np.minimum(wmo, 250)

        # Eq. 4 - equilibrium moisture content from drying
        ed = 0.942 * rh**0.679 + 11 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (
            1 - 1 / np.exp(rh * 0.115)
        )
        # Eq. 5 - equilibrium moisture content from wetting
        ew = 0.618 * rh**0.753 + 10 * np.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (
            1 - 1 / np.exp(rh * 0.115)
        )

        # Eq. 6a/6b -> Eq. 8
        z = np.where(
            (wmo < ed) & (wmo < ew),
            0.424 * (1 - ((100 - rh) / 100) ** 1.7) + 0.0694 * np.sqrt(ws) * (1 - ((100 - rh) / 100) ** 8),
            0.0,
        )
        x = z * 0.581 * np.exp(0.0365 * temp)
        wm = np.where((wmo < ed) & (wmo < ew), ew - (ew - wmo) / (10**x), wmo)

        # Eq. 7a/7b -> Eq. 9
        z = np.where(
            wmo > ed,
            0.424 * (1 - (rh / 100) ** 1.7) + 0.0694 * np.sqrt(ws) * (1 - (rh / 100) ** 8),
            z,
        )
        x = z * 0.581 * np.exp(0.0365 * temp)
        wm = np.where(wmo > ed, ed + (wmo - ed) / (10**x), wm)

        # Eq. 10
        ffmc1 = 59.5 * (250 - wm) / (FFMC_COEFFICIENT + wm)
    return np.clip(ffmc1, 0, 101)


def duff_moisture_code(dmc_yda, temp, rh, prec, lat, mon, lat_adjust=True):
    dmc_yda = np.asarray(dmc_yda, dtype=float)
    temp = np.asarray(temp, dtype=float)
    rh = np.asarray(rh, dtype=float)
    prec = np.asarray(prec, dtype=float)
    lat = np.asarray(lat, dtype=float)
    mon = np.asarray(mon, dtype=int)

    with np.errstate(divide="ignore", invalid="ignore"):
        temp = np.where(temp < -1.1, -1.1, temp)
        # Eq. 16 - log drying rate (46N reference table)
        rk = 1.894 * (temp + 1.1) * (100 - rh) * _DMC_ELL_46N[mon - 1] * 1e-4

        if lat_adjust:
            rk = np.where(
                (lat <= 30) & (lat > 10),
                1.894 * (temp + 1.1) * (100 - rh) * _DMC_ELL_20N[mon - 1] * 1e-4,
                rk,
            )
            rk = np.where(
                (lat <= -10) & (lat > -30),
                1.894 * (temp + 1.1) * (100 - rh) * _DMC_ELL_20S[mon - 1] * 1e-4,
                rk,
            )
            rk = np.where(
                (lat <= -30) & (lat >= -90),
                1.894 * (temp + 1.1) * (100 - rh) * _DMC_ELL_40S[mon - 1] * 1e-4,
                rk,
            )
            rk = np.where(
                (lat <= 10) & (lat > -10),
                1.894 * (temp + 1.1) * (100 - rh) * 9 * 1e-4,
                rk,
            )

        # Eq. 12 (altered), 13a-c, 14, 15 (altered)
        wmi = 20 + 280 / np.exp(0.023 * dmc_yda)
        b = np.where(
            dmc_yda <= 33,
            100 / (0.5 + 0.3 * dmc_yda),
            np.where(dmc_yda <= 65, 14 - 1.3 * np.log(dmc_yda), 6.2 * np.log(dmc_yda) - 17.2),
        )
        rw = 0.92 * prec - 1.27
        wmr = wmi + 1000 * rw / (48.77 + b * rw)
        pr_rain = 43.43 * (5.6348 - np.log(wmr - 20))
        pr = np.where(prec <= 1.5, dmc_yda, pr_rain)
        pr = np.maximum(pr, 0)

        dmc1 = pr + rk
    return np.maximum(dmc1, 0)


def drought_code(dc_yda, temp, rh, prec, lat, mon, lat_adjust=True):
    dc_yda = np.asarray(dc_yda, dtype=float)
    temp = np.asarray(temp, dtype=float)
    prec = np.asarray(prec, dtype=float)
    lat = np.asarray(lat, dtype=float)
    mon = np.asarray(mon, dtype=int)

    with np.errstate(divide="ignore", invalid="ignore"):
        temp = np.where(temp < -2.8, -2.8, temp)
        # Eq. 22 - potential evapotranspiration
        pe = (0.36 * (temp + 2.8) + _DC_FL_20N[mon - 1]) / 2

        if lat_adjust:
            pe = np.where(lat <= -20, (0.36 * (temp + 2.8) + _DC_FL_20S[mon - 1]) / 2, pe)
            pe = np.where((lat > -20) & (lat <= 20), (0.36 * (temp + 2.8) + 1.4) / 2, pe)
        pe = np.maximum(pe, 0)

        # Eq. 18, 19, 21 (altered), 23 (altered)
        rw = 0.83 * prec - 1.27
        smi = 800 * np.exp(-dc_yda / 400)
        dr0 = dc_yda - 400 * np.log(1 + 3.937 * rw / smi)
        dr0 = np.maximum(dr0, 0)
        dr = np.where(prec <= 2.8, dc_yda, dr0)
        dc1 = dr + pe
    return np.maximum(dc1, 0)


def initial_spread_index(ffmc, ws, fbp_mod=False):
    ffmc = np.asarray(ffmc, dtype=float)
    ws = np.asarray(ws, dtype=float)
    # Eq. 10 - moisture content
    fm = FFMC_COEFFICIENT * (101 - ffmc) / (59.5 + ffmc)
    # Eq. 24 - wind effect
    if fbp_mod:
        fw = np.where(ws >= 40, 12 * (1 - np.exp(-0.0818 * (ws - 28))), np.exp(0.05039 * ws))
    else:
        fw = np.exp(0.05039 * ws)
    # Eq. 25 - fine fuel moisture effect
    ff = 91.9 * np.exp(-0.1386 * fm) * (1 + fm**5.31 / 49_300_000)
    # Eq. 26
    return 0.208 * fw * ff


def buildup_index(dmc, dc):
    dmc = np.asarray(dmc, dtype=float)
    dc = np.asarray(dc, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Eq. 27a
        bui1 = np.where((dmc == 0) & (dc == 0), 0.0, 0.8 * dc * dmc / (dmc + 0.4 * dc))
        # Eq. 27b
        p = np.where(dmc == 0, 0.0, (dmc - bui1) / dmc)
        cc = 0.92 + (0.0114 * dmc) ** 1.7
        bui0 = np.maximum(dmc - cc * p, 0)
        bui1 = np.where(bui1 < dmc, bui0, bui1)
    return bui1


def fire_weather_index(isi, bui):
    isi = np.asarray(isi, dtype=float)
    bui = np.asarray(bui, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Eqs. 28a, 28b
        bb = np.where(
            bui > 80,
            0.1 * isi * (1000 / (25 + 108.64 / np.exp(0.023 * bui))),
            0.1 * isi * (0.626 * bui**0.809 + 2),
        )
        # Eqs. 29, 30a, 30b
        fwi = np.where(bb <= 1, bb, np.exp(2.72 * (0.434 * np.log(bb)) ** 0.647))
    return fwi


def compute_fwi_series(weather_df, startup=None, fire_season_start_mmdd="03-01", strict=True):
    """Recursively compute daily FWI System outputs per grid cell.

    `weather_df` must contain one row per (cell_id, date) with columns
    [cell_id, date, lat, temp, rh, wind, precip], covering the full cell x
    date grid (every cell present on every date). FFMC/DMC/DC are reset to
    the standard startup values on `fire_season_start_mmdd` each year.

    If `strict=False`, cells missing any date (e.g. a batch that failed to
    fetch after retries) are dropped rather than raising, so a handful of
    gaps only shrinks the output instead of failing the whole computation --
    useful for an operational daily run where a single stuck upstream
    request shouldn't block the entire risk map.
    """
    startup = startup or DEFAULT_STARTUP
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    cell_ids = np.sort(df["cell_id"].unique())
    n_cells = len(cell_ids)
    dates = np.sort(df["date"].unique())
    n_dates = len(dates)

    df = df.sort_values(["date", "cell_id"])
    if len(df) != n_cells * n_dates:
        if not strict:
            counts = df.groupby("cell_id")["date"].nunique()
            complete_cells = counts[counts == n_dates].index
            dropped = n_cells - len(complete_cells)
            print(
                f"  compute_fwi_series: dropping {dropped}/{n_cells} cells with incomplete "
                f"date coverage (kept {len(complete_cells)})"
            )
            df = df[df["cell_id"].isin(complete_cells)]
            cell_ids = np.sort(complete_cells.to_numpy())
            n_cells = len(cell_ids)
        else:
            raise ValueError(
                f"weather_df is not a complete cell x date grid: got {len(df)} rows, "
                f"expected {n_cells * n_dates} ({n_cells} cells x {n_dates} dates)"
            )

    temp_arr = df["temp"].to_numpy().reshape(n_dates, n_cells)
    rh_arr = np.clip(df["rh"].to_numpy(), 0, 99.9999).reshape(n_dates, n_cells)
    ws_arr = df["wind"].to_numpy().reshape(n_dates, n_cells)
    prec_arr = df["precip"].to_numpy().reshape(n_dates, n_cells)
    lat_arr = df["lat"].to_numpy().reshape(n_dates, n_cells)

    out = {k: np.empty((n_dates, n_cells)) for k in ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI")}

    ffmc_prev = np.full(n_cells, startup["ffmc"], dtype=float)
    dmc_prev = np.full(n_cells, startup["dmc"], dtype=float)
    dc_prev = np.full(n_cells, startup["dc"], dtype=float)

    for i, date in enumerate(dates):
        ts = pd.Timestamp(date)
        if ts.strftime("%m-%d") == fire_season_start_mmdd:
            ffmc_prev = np.full(n_cells, startup["ffmc"], dtype=float)
            dmc_prev = np.full(n_cells, startup["dmc"], dtype=float)
            dc_prev = np.full(n_cells, startup["dc"], dtype=float)

        temp, rh, ws, prec, lat = temp_arr[i], rh_arr[i], ws_arr[i], prec_arr[i], lat_arr[i]
        mon = ts.month

        ffmc = fine_fuel_moisture_code(ffmc_prev, temp, rh, ws, prec)
        dmc = duff_moisture_code(dmc_prev, temp, rh, prec, lat, mon)
        dc = drought_code(dc_prev, temp, rh, prec, lat, mon)
        isi = initial_spread_index(ffmc, ws)
        bui = buildup_index(dmc, dc)
        fwi = fire_weather_index(isi, bui)

        out["FFMC"][i], out["DMC"][i], out["DC"][i] = ffmc, dmc, dc
        out["ISI"][i], out["BUI"][i], out["FWI"][i] = isi, bui, fwi

        ffmc_prev, dmc_prev, dc_prev = ffmc, dmc, dc

    result = pd.DataFrame(
        {
            "cell_id": np.tile(cell_ids, n_dates),
            "date": np.repeat(dates, n_cells),
            **{k: v.ravel() for k, v in out.items()},
        }
    )
    result["DSR"] = 0.0272 * result["FWI"] ** 1.77
    return result


# Reference test data: Van Wagner & Pickett (1985) standard FWI test dataset,
# as published (rounded to 1 decimal place, DSR to 2) via the CRAN `cffdrs` R
# package (`data("test_fwi")`, `init = data.frame(ffmc=85, dmc=6, dc=15, lat=40)`).
# Source: https://github.com/cffdrs/cffdrs_py/blob/main/cffdrs/tests/data/fwi_test_data.csv
_REFERENCE_TEST_DATA_CSV = """LONG,LAT,YR,MON,DAY,TEMP,RH,WS,PREC,FFMC,DMC,DC,ISI,BUI,FWI,DSR
-100,40,1985,4,13,17,42,25,0,87.6,8.5,19,10.8,8.5,10,1.61
-100,40,1985,4,14,20,21,25,2.4,86.2,10.4,23.6,8.8,10.4,9.2,1.39
-100,40,1985,4,15,8.5,40,17,0,86.9,11.8,26.1,6.5,11.7,7.5,0.97
-100,40,1985,4,16,6.5,25,6,0,88.8,13.2,28.2,4.9,13.1,6.1,0.67
-100,40,1985,4,17,13,34,24,0,89,15.4,31.5,12.5,15.3,14.7,3.18
-100,40,1985,4,18,6,40,22,0.4,88.6,16.5,33.5,10.6,16.4,13.4,2.67
-100,40,1985,4,19,5.5,52,6,0,87.3,17.2,35.4,3.9,17.1,5.8,0.61
-100,40,1985,4,20,8.5,46,16,0,87.3,18.5,37.9,6.5,18.4,9.5,1.46
-100,40,1985,4,21,9.5,54,20,0,86.7,19.7,40.6,7.3,19.6,10.9,1.86
-100,40,1985,4,22,7,93,14,9,29.8,10.1,29.5,0,10.9,0,0
-100,40,1985,4,23,6.5,71,17,1,49.4,10.7,31.6,0.4,11.6,0.2,0
-100,40,1985,4,24,6,59,17,0,67.2,11.4,33.7,1.3,12.3,0.9,0.02
-100,40,1985,4,25,13,52,4,0,77.7,13,37,1.1,13.9,0.8,0.02
-100,40,1985,4,26,15.5,40,11,0,85.4,15.4,40.7,3.9,15.9,5.5,0.55
-100,40,1985,4,27,23,25,9,0,91.5,19.8,45.8,8.3,19.8,12.1,2.24
-100,40,1985,4,28,19,46,16,0,89.9,22.5,50.2,9.4,22.4,14.2,2.98
-100,40,1985,4,29,18,41,20,0,89.9,25.2,54.4,11.5,25.1,17.5,4.31
-100,40,1985,4,30,14.5,51,16,0,88.4,27,57.9,7.6,27,13.2,2.61
-100,40,1985,5,1,14.5,69,11,0,85.6,28.3,63,4,28.2,7.9,1.07
-100,40,1985,5,2,15.5,42,8,0,87.3,30.8,68.2,4.4,30.8,9,1.33
-100,40,1985,5,3,21,37,8,0,89.3,34.5,74.3,5.8,34.4,12.2,2.27
-100,40,1985,5,4,23,32,16,0,90.9,38.8,80.9,11,38.7,20.9,5.9
-100,40,1985,5,5,23,32,14,0,91.2,43.1,87.4,10.2,43,20.9,5.93
-100,40,1985,5,6,27,33,12,0,91.6,48.1,94.7,9.9,47.9,21.6,6.24
-100,40,1985,5,7,28,17,27,0,95.1,54.5,102.1,34.3,54.3,52.3,29.96
-100,40,1985,5,8,23.5,54,20,0,89.7,57.4,108.8,11.2,57.2,25.7,8.53
-100,40,1985,5,9,16,50,22,12.2,62.2,29.9,91.8,1.4,33,3,0.19
-100,40,1985,5,10,11,58,20,0,76.4,31.3,96.2,2.3,34.5,5.3,0.53
-100,40,1985,5,11,16,54,16,0,83.3,33.3,101.5,3.8,36.6,8.8,1.27
-100,40,1985,5,12,21.5,37,9,0,88.6,37.1,107.7,5.5,39.9,12.6,2.42
-100,40,1985,5,13,14,61,22,0.2,86.6,38.6,112.7,8,41.6,17.2,4.18
-100,40,1985,5,14,15,30,27,0,89.6,41.6,117.8,15.7,44.2,28.7,10.32
-100,40,1985,5,15,20,23,11,0,92.1,45.9,123.8,10,47.6,21.7,6.32
-100,40,1985,5,16,14,95,3,16.4,21.3,20.1,96.9,0,26.5,0,0
-100,40,1985,5,17,20,53,4,2.8,51,18.2,102.9,0.2,25.3,0.2,0
-100,40,1985,5,18,19.5,30,16,0,82.2,22,108.8,3.3,29.3,6.8,0.8
-100,40,1985,5,19,25.5,51,20,6,75.3,16.4,106.3,2.1,23.6,3.8,0.29
-100,40,1985,5,20,10,38,24,0,84.3,18.2,110.5,6.4,25.8,11.2,1.96
-100,40,1985,5,21,19,27,16,0,90.3,22,116.4,9.9,29.9,17.1,4.14
-100,40,1985,5,22,26,46,11,4.2,77.5,18.7,117.7,1.6,26.7,2.9,0.18
-100,40,1985,5,23,30,38,22,0,90.2,23.7,125.5,13.3,32.2,21.9,6.42
-100,40,1985,5,24,25.5,67,19,12.6,65.3,13.1,108.4,1.4,20.2,1.9,0.08
-100,40,1985,5,25,12,53,28,11.8,55.4,7.7,91.6,1.2,12.8,0.8,0.02
-100,40,1985,5,26,21,38,8,0,80.8,11.3,97.8,1.9,17.6,2.5,0.14
-100,40,1985,5,27,13,70,20,3.8,61.7,8.4,97.8,1.2,13.8,0.9,0.02
-100,40,1985,5,28,9,78,24,1.4,64.4,9,101.8,1.7,14.7,1.9,0.09
-100,40,1985,5,29,11,54,16,0,77.6,10.5,106.2,2,16.8,2.8,0.16
-100,40,1985,5,30,15.5,39,9,0,85.4,13.1,111.4,3.5,20.3,5.7,0.6
"""


def _validate():
    """Validate this port against the published Van Wagner & Pickett (1985)
    test dataset, sequentially, the same way the reference `cffdrs` test
    suite does. Raises AssertionError on any mismatch beyond tolerance."""
    import io

    ref = pd.read_csv(io.StringIO(_REFERENCE_TEST_DATA_CSV))

    ffmc_prev, dmc_prev, dc_prev = 85.0, 6.0, 15.0
    max_err = {"FFMC": 0, "DMC": 0, "DC": 0, "ISI": 0, "BUI": 0, "FWI": 0}

    for _, row in ref.iterrows():
        ffmc = float(fine_fuel_moisture_code(ffmc_prev, row.TEMP, row.RH, row.WS, row.PREC))
        dmc = float(duff_moisture_code(dmc_prev, row.TEMP, row.RH, row.PREC, row.LAT, int(row.MON)))
        dc = float(drought_code(dc_prev, row.TEMP, row.RH, row.PREC, row.LAT, int(row.MON)))
        isi = float(initial_spread_index(ffmc, row.WS))
        bui = float(buildup_index(dmc, dc))
        fwi = float(fire_weather_index(isi, bui))

        computed = {"FFMC": ffmc, "DMC": dmc, "DC": dc, "ISI": isi, "BUI": bui, "FWI": fwi}
        for key, val in computed.items():
            err = abs(val - row[key])
            max_err[key] = max(max_err[key], err)
            assert err < 0.15, (
                f"{key} mismatch on {row.YR}-{row.MON}-{row.DAY}: "
                f"got {val:.3f}, expected {row[key]} (err {err:.3f})"
            )

        ffmc_prev, dmc_prev, dc_prev = ffmc, dmc, dc

    print(f"Validated {len(ref)} days against Van Wagner & Pickett (1985) reference data.")
    print("Max abs error per variable:", {k: round(v, 4) for k, v in max_err.items()})


if __name__ == "__main__":
    _validate()
