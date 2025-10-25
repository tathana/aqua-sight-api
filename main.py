# main.py — Aqua Sight API (fixed monthly_series KeyError + agg support)

import os, json, base64, tempfile
from typing import Dict, List, Literal, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ee

# ---------- 0) Earth Engine Init ----------
cred_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
SA_EMAIL = os.getenv("EE_SERVICE_ACCOUNT")
KEY_B64 = os.getenv("EE_KEY_B64")
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

try:
    if cred_json:
        info = json.loads(cred_json)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=cred_json)
        ee.Initialize(creds)
    elif SA_EMAIL and KEY_B64:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.write(base64.b64decode(KEY_B64))
        tmp.flush()
        creds = ee.ServiceAccountCredentials(SA_EMAIL, tmp.name)
        ee.Initialize(creds)
    elif SA_EMAIL and KEY_PATH:
        creds = ee.ServiceAccountCredentials(SA_EMAIL, KEY_PATH)
        ee.Initialize(creds)
    else:
        raise RuntimeError("Missing Earth Engine credentials.")
except Exception as e:
    raise RuntimeError(f"Earth Engine init failed: {e}")

app = FastAPI(title="Aqua Sight API", version="1.2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------- 1) AOIs ----------
def poly(coords): return ee.Geometry.Polygon(coords)

AOIS = {
    "CP01": poly([[99.24686562565613,10.445791568889858],[99.24744498280334,10.445707160671937],[99.24735915211487,10.442373017728443],[99.24484860447693,10.439418683974635],[99.2461575224762,10.439671913681973],[99.24770247486877,10.442330813158181],[99.24804579762268,10.445749364783767],[99.24873244313049,10.445960385256942],[99.24843203572082,10.448070582107523],[99.24692999867248,10.448049480209948],[99.24686562565613,10.445791568889858]]),
    "LS01": poly([[99.15535572624208,9.94453648425491],[99.14522770500184,9.94033053669308],[99.14578560447694,9.939971232131072],[99.15164354896547,9.940985738113076],[99.15320995903016,9.94231727243655],[99.15604237174989,9.94314155287098],[99.15535572624208,9.94453648425491]]),
}
YEARS = list(range(2017, 2026))

# ---------- 2) Helpers ----------
def get_window(year:int):
    ini = ee.Date.fromYMD(year,1,1); end = ini.advance(12,"month")
    return ini, end

def build_water_mask(geom: ee.Geometry) -> ee.Image:
    col = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
           .filterBounds(geom)
           .filter(ee.Filter.lt('CLOUD_COVER',30))
           .select('SR_B6'))
    m = ee.Image(col.median().lt(300))
    return m.updateMask(m)

def s2_sr(geom, ini, end, cloud_perc:int=30):
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom).filterDate(ini, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_perc)))

def add_scaled(img: ee.Image, mask: ee.Image) -> ee.Image:
    bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
    return (img.select(bands).divide(10000).multiply(mask)
            .copyProperties(img, ['system:time_start']))

# ---------- 3) Water-quality formulas ----------
def img_pH(img):   return ee.Image(8.339).subtract(ee.Image(0.827).multiply(img.select('B1').divide(img.select('B8')))).rename('pH').copyProperties(img, ['system:time_start'])
def img_turb(img): return ee.Image(100).multiply(ee.Image(1).subtract(img.normalizedDifference(['B8','B4']))).rename('turbidity').copyProperties(img, ['system:time_start'])
def img_sal(img):  return img.normalizedDifference(['B11','B12']).rename('salinity_idx').copyProperties(img, ['system:time_start'])
def img_do(img):   return (ee.Image(-0.0167).multiply(img.select('B8')).add(ee.Image(0.0067).multiply(img.select('B9'))).add(ee.Image(0.0083).multiply(img.select('B11'))).add(9.577)).rename('do_mgL').copyProperties(img, ['system:time_start'])
def img_chl(img):
    ndci = img.normalizedDifference(['B5','B4'])
    return ee.Image(14.039).add(ee.Image(86.115).multiply(ndci)).add(ee.Image(194.325).multiply(ndci.pow(2))).rename('chl_a').copyProperties(img, ['system:time_start'])
def img_zsd(img):
    blueRed = img.select('B2').divide(img.select('B4')).log()
    lnMOSD = ee.Image(1.4856).multiply(blueRed).add(0.2734)
    return (ee.Image(0.1777).multiply(ee.Image(10).pow(lnMOSD)).add(1.0813)).rename('secchi_m').copyProperties(img, ['system:time_start'])
def img_tsi_from_chl(chl_img): return ee.Image(30.6).add(ee.Image(9.81).multiply(chl_img.log())).rename('tsi').copyProperties(chl_img, ['system:time_start'])

def tsi_reclass(tsi_img: ee.Image) -> ee.Image:
    img = tsi_img
    mask1 = img.lt(30)
    mask2 = img.gte(30).And(img.lt(40))
    mask3 = img.gte(40).And(img.lt(50))
    mask4 = img.gte(50).And(img.lt(60))
    mask5 = img.gte(60).And(img.lt(70))
    mask6 = img.gte(70).And(img.lt(80))
    mask7 = img.gte(80)
    out = img.where(mask1, 1).where(mask2, 2).where(mask3, 3).where(mask4, 4).where(mask5, 5).where(mask6, 6).where(mask7, 7)
    return out.rename('tsi_class').copyProperties(tsi_img, ['system:time_start'])

# ---------- 4) Series helpers ----------
SCALE = 20
MAXPX = 1e13

def monthly_series(ic, geom, band, year):
    out = []
    for m in range(1, 13):
        start = ee.Date.fromYMD(year, m, 1)
        end = start.advance(1, "month")
        month_ic = ic.filterDate(start, end)

        def per_img(img):
            val = img.select(band).reduceRegion(
                ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX
            ).get(band)
            return ee.Feature(None, {"v": val})

        feats = ee.FeatureCollection(month_ic.map(per_img)).getInfo().get("features", [])
        vals = []
        for f in feats:
            props = f.get("properties") or {}
            if "v" in props and props["v"] is not None:
                try:
                    vals.append(float(props["v"]))
                except Exception:
                    pass

        vpy = (sum(vals) / len(vals)) if len(vals) else None
        out.append({"month": m, "value": vpy})
    return out

def scenes_series(ic, geom, band):
    def per_image(img):
        val = img.select(band).reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get(band)
        return ee.Feature(None, {"date": ee.Date(img.get('system:time_start')).format("YYYY-MM-dd"), "value": val})
    fc = ee.FeatureCollection(ic.map(per_image)).getInfo().get("features", [])
    out = []
    for f in fc:
        props = f.get("properties") or {}
        if "date" in props:
            val = props.get("value")
            out.append({"date": props["date"], "value": (float(val) if val is not None else None)})
    out.sort(key=lambda x: x["date"])
    return out

# ---------- 5) Endpoints ----------
@app.get("/timeseries_monthly")
def timeseries_monthly(
    station: Literal["CP01","LS01"],
    year: int = Query(..., ge=2017, le=2025),
    ac: Literal["none","full"] = "none"
):
    if station not in AOIS: raise HTTPException(404, "Unknown station")
    geom = AOIS[station]
    ini, end = get_window(year)
    mask = build_water_mask(geom)
    col = s2_sr(geom, ini, end).map(lambda im: add_scaled(im, mask))

    ph_ic = col.map(img_pH)
    tur_ic = col.map(img_turb)
    do_ic = col.map(img_do)
    chl_ic = col.map(img_chl)
    zsd_ic = col.map(img_zsd)
    tsi_ic = chl_ic.map(img_tsi_from_chl)
    tsi_cls_ic = tsi_ic.map(tsi_reclass)
    sal_ic = col.map(img_sal)

    return {
        "station": station,
        "year": year,
        "monthly": {
            "pH": monthly_series(ph_ic, geom, "pH", year),
            "turbidity": monthly_series(tur_ic, geom, "turbidity", year),
            "salinity_idx": monthly_series(sal_ic, geom, "salinity_idx", year),
            "do_mgL": monthly_series(do_ic, geom, "do_mgL", year),
            "chl_a": monthly_series(chl_ic, geom, "chl_a", year),
            "secchi_m": monthly_series(zsd_ic, geom, "secchi_m", year),
            "tsi": monthly_series(tsi_ic, geom, "tsi", year),
            "tsi_class": monthly_series(tsi_cls_ic, geom, "tsi_class", year)
        }
    }

@app.get("/health-ee")
def health_ee():
    try:
        return {"ok": True, "roots": ee.data.getAssetRoots()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/")
def root():
    return {"name": "Aqua Sight API",
            "examples": {
                "monthly": "/timeseries_monthly?station=CP01&year=2024"
            }}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
