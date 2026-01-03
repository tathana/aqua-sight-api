# =========================================================
# Aqua Sight API — Code B (Production Safe)
# Fix scale issues + monthly coverage + cloud-aware nulls
# =========================================================

import os, json, base64, tempfile, re
from typing import Dict, List, Literal, Any, Optional

import ee
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------
# 0) Earth Engine Init
# ---------------------------------------------------------
cred_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
SA_EMAIL  = os.getenv("EE_SERVICE_ACCOUNT")
KEY_B64   = os.getenv("EE_KEY_B64")
KEY_PATH  = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

try:
    if cred_json:
        info = json.loads(cred_json)
        ee.Initialize(
            ee.ServiceAccountCredentials(
                info["client_email"], key_data=cred_json
            )
        )
    elif SA_EMAIL and KEY_B64:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.write(base64.b64decode(KEY_B64))
        tmp.flush()
        ee.Initialize(ee.ServiceAccountCredentials(SA_EMAIL, tmp.name))
    elif SA_EMAIL and KEY_PATH:
        ee.Initialize(ee.ServiceAccountCredentials(SA_EMAIL, KEY_PATH))
    else:
        raise RuntimeError("Missing Earth Engine credentials")
except Exception as e:
    raise RuntimeError(f"Earth Engine init failed: {e}")

# ---------------------------------------------------------
# App
# ---------------------------------------------------------
app = FastAPI(title="Aqua Sight API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# AOIs
# ---------------------------------------------------------
def poly(c): return ee.Geometry.Polygon(c)

AOIS = {
    "CP01": poly([[99.2468656,10.4457916],[99.2474450,10.4457071],[99.2473591,10.4423730],[99.2448486,10.4394187],[99.2461575,10.4396719],[99.2477025,10.4423308],[99.2480458,10.4457493],[99.2487324,10.4459603],[99.2484320,10.4480705],[99.2469300,10.4480494],[99.2468656,10.4457916]]),
    "LS01": poly([[99.1553557,9.9445364],[99.1452277,9.9403305],[99.1457856,9.9399712],[99.1516435,9.9409857],[99.1532099,9.9423172],[99.1560423,9.9431415],[99.1553557,9.9445364]]),
    "LS03": poly([[99.0621109,9.9539187],[99.0633769,9.9534115],[99.0649541,9.9530627],[99.0682371,9.9533164],[99.0680654,9.9536228],[99.0640421,9.9536651],[99.0622504,9.9542780],[99.0621109,9.9539187]]),
    "TP01": poly([[99.3740662,9.1892754],[99.3680795,9.1858863],[99.3712123,9.1840010],[99.3819841,9.1911607],[99.3784436,9.1937449],[99.3740662,9.1892754]]),
    "TP04": poly([[99.1745863,9.0884324],[99.1699407,9.0868009],[99.1675481,9.0849681],[99.1660139,9.0825103],[99.1674516,9.0821712],[99.1700158,9.0855720],[99.1720006,9.0867056],[99.1757235,9.0874366],[99.1745863,9.0884324]]),
    "TP11": poly([[99.6078125,8.5328002],[99.6101084,8.5345084],[99.6106127,8.5358028],[99.6106127,8.5363333],[99.6097651,8.5348373],[99.6075657,8.5335747],[99.6078125,8.5328002]]),
    "PN01": poly([[99.9084979,7.8915176],[99.9107939,7.8934305],[99.9111587,7.8943338],[99.9080795,7.8920490],[99.9084979,7.8915176]]),
    "SK01": poly([[100.1251495,7.7890046],[100.1251287,7.7889043],[100.1258073,7.7888525],[100.1258093,7.7888631],[100.1251522,7.7889202],[100.1251710,7.7890053],[100.1251495,7.7890046]]),
    "SK06": poly([[100.1577107,7.6251339],[100.1590518,7.6249371],[100.1596205,7.6244586],[100.1596151,7.6255645],[100.1585959,7.6253306],[100.1567505,7.6258038],[100.1577107,7.6251339]])
}

YEARS = list(range(2017, 2026))

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
SCALE = 20
MAXPX = 1e13

def get_window(year):
    ini = ee.Date.fromYMD(year,1,1)
    return ini, ini.advance(12,"month")

def water_mask(geom):
    col = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
           .filterBounds(geom)
           .filter(ee.Filter.lt("CLOUD_COVER",30))
           .select("SR_B6"))
    m = ee.Image(col.median().lt(300))
    return m.updateMask(m)

def s2_sr(geom, ini, end, cloud):
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(ini, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)))

# ---------------------------------------------------------
# Water quality indices (FIXED SCALE)
# ---------------------------------------------------------
def img_chl(img):
    ndci = img.normalizedDifference(["B5","B4"])
    return (ee.Image(14.039)
            .add(ee.Image(86.115).multiply(ndci))
            .add(ee.Image(194.325).multiply(ndci.pow(2)))
            .rename("chl_a")
            .copyProperties(img, ["system:time_start"]))

def img_zsd(img):
    blue_red = img.select("B2").divide(img.select("B4")).log()
    lnMOSD = ee.Image(1.4856).multiply(blue_red).add(0.2734)
    zsd = ee.Image(0.1777).multiply(ee.Image(10).pow(lnMOSD)).add(1.0813)
    # physical constraint
    zsd = zsd.updateMask(zsd.gt(0).And(zsd.lte(10)))
    return zsd.rename("secchi_m").copyProperties(img, ["system:time_start"])

def img_tsi(chl):
    return ee.Image(30.6).add(ee.Image(9.81).multiply(chl.log())).rename("tsi")

# ---------------------------------------------------------
# Coverage
# ---------------------------------------------------------
def coverage_pct(img, geom):
    valid = img.mask().reduceRegion(
        ee.Reducer.sum(), geom, SCALE, maxPixels=MAXPX
    ).values().get(0)
    total = ee.Image.constant(1).clip(geom).reduceRegion(
        ee.Reducer.count(), geom, SCALE, maxPixels=MAXPX
    ).values().get(0)
    return ee.Number(valid).divide(total).multiply(100)

# ---------------------------------------------------------
# API: summary_year (FIXED SCALE)
# ---------------------------------------------------------
@app.get("/summary_year")
def summary_year(
    station: Literal["CP01","LS01","LS03","TP01","TP04","TP11","PN01","SK01","SK06"],
    year: int = Query(..., ge=2017, le=2025),
    cloud_perc: int = 30
):
    geom = AOIS[station]
    ini, end = get_window(year)
    mask = water_mask(geom)

    base = s2_sr(geom, ini, end, cloud_perc).map(
        lambda im: im.select(
            ['B1','B2','B3','B4','B5','B8','B11','B12']
        ).divide(10000).multiply(mask)
    )

    chl = base.map(img_chl).mean()
    zsd = base.map(img_zsd).mean()
    tsi = chl.map(img_tsi).mean()

    def safe(x):
        try: return float(ee.Number(x).getInfo())
        except: return None

    return {
        "station": station,
        "year": year,
        "cloud_perc": cloud_perc,
        "mean": {
            "chl_a": safe(chl.reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get("chl_a")),
            "secchi_m": safe(zsd.reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get("secchi_m")),
            "tsi": safe(tsi.reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get("tsi"))
        }
    }

# ---------------------------------------------------------
# API: timeseries_monthly (NULL + COVERAGE)
# ---------------------------------------------------------
@app.get("/timeseries_monthly")
def timeseries_monthly(
    station: Literal["CP01","LS01","LS03","TP01","TP04","TP11","PN01","SK01","SK06"],
    year: int = Query(..., ge=2017, le=2025),
    cloud_perc: int = 30
):
    geom = AOIS[station]
    ini, end = get_window(year)
    mask = water_mask(geom)

    base = s2_sr(geom, ini, end, cloud_perc).map(
        lambda im: im.select(
            ['B1','B2','B3','B4','B5','B8','B11','B12']
        ).divide(10000).multiply(mask)
    )

    chl_ic = base.map(img_chl)
    zsd_ic = base.map(img_zsd)
    tsi_ic = chl_ic.map(img_tsi)

    def monthly(ic, band):
        out = []
        for m in range(1,13):
            start = ee.Date.fromYMD(year,m,1)
            endm  = start.advance(1,"month")
            mic = ic.filterDate(start,endm)

            if mic.size().getInfo() == 0:
                out.append({
                    "month": m,
                    "value": None,
                    "coverage_pct": 0,
                    "status": "NO_DATA_CLOUD"
                })
                continue

            img = mic.mean()
            cov = coverage_pct(img, geom).getInfo()

            val = img.reduceRegion(
                ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX
            ).get(band)

            val = None if val is None else float(ee.Number(val).getInfo())

            status = "OK" if cov >= 20 else "LOW_COVERAGE"

            out.append({
                "month": m,
                "value": val,
                "coverage_pct": round(cov,2),
                "status": status
            })
        return out

    return {
        "station": station,
        "year": year,
        "cloud_perc": cloud_perc,
        "monthly": {
            "chl_a": monthly(chl_ic,"chl_a"),
            "secchi_m": monthly(zsd_ic,"secchi_m"),
            "tsi": monthly(tsi_ic,"tsi")
        }
    }

# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Aqua Sight API (Code B – Fixed)",
        "endpoints": [
            "/summary_year",
            "/timeseries_monthly"
        ]
    }
