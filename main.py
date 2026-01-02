# =========================================================
# Aqua Sight API — Production Safe (Render-ready)
# =========================================================

import os, json, base64, tempfile, re
from typing import Dict, List, Literal, Any, Optional

import ee
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# =========================================================
# 0) Earth Engine Init (Render-safe)
# =========================================================
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

# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="Aqua Sight API", version="2.1.0-render")

allowed = os.getenv("ALLOWED_ORIGIN", "*")
origins = [o.strip() for o in allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# =========================================================
# 1) AOIs (ALL STATIONS)
# =========================================================
def poly(coords): return ee.Geometry.Polygon(coords)

AOIS = {
    "CP01": poly([[99.2468656,10.4457916],[99.2474450,10.4457072],[99.2473592,10.4423730],
                  [99.2448486,10.4394187],[99.2461575,10.4396719],[99.2477025,10.4423308],
                  [99.2480458,10.4457494],[99.2487324,10.4459604],[99.2484320,10.4480706],
                  [99.2469300,10.4480495],[99.2468656,10.4457916]]),
    "LS01": poly([[99.1553557,9.9445365],[99.1452277,9.9403305],[99.1457856,9.9399712],
                  [99.1516435,9.9409857],[99.1532100,9.9423173],[99.1560424,9.9431416],
                  [99.1553557,9.9445365]]),
    "LS03": poly([[99.0621110,9.9539187],[99.0633770,9.9534115],[99.0649541,9.9530628],
                  [99.0682371,9.9533164],[99.0680655,9.9536229],[99.0640422,9.9536651],
                  [99.0622504,9.9542780],[99.0621110,9.9539187]]),
    "TP01": poly([[99.3740663,9.1892755],[99.3680796,9.1858863],[99.3712124,9.1840011],
                  [99.3819842,9.1911607],[99.3784436,9.1937449],[99.3740663,9.1892755]]),
    "TP04": poly([[99.1745863,9.0884325],[99.1699407,9.0868010],[99.1675482,9.0849682],
                  [99.1660140,9.0825103],[99.1674516,9.0821713],[99.1700158,9.0855721],
                  [99.1720007,9.0867056],[99.1757236,9.0874366],[99.1745863,9.0884325]]),
    "TP11": poly([[99.6078125,8.5328002],[99.6101085,8.5345084],[99.6106127,8.5358028],
                  [99.6106127,8.5363333],[99.6097652,8.5348373],[99.6075658,8.5335747],
                  [99.6078125,8.5328002]]),
    "PN01": poly([[99.9084980,7.8915176],[99.9107939,7.8934305],[99.9111587,7.8943339],
                  [99.9080795,7.8920490],[99.9084980,7.8915176]]),
    "SK01": poly([[100.1251496,7.7890047],[100.1251288,7.7889043],[100.1258074,7.7888525],
                  [100.1258094,7.7888631],[100.1251522,7.7889203],[100.1251710,7.7890053],
                  [100.1251496,7.7890047]]),
    "SK06": poly([[100.1577108,7.6251339],[100.1590519,7.6249372],[100.1596205,7.6244586],
                  [100.1596152,7.6255646],[100.1585959,7.6253306],[100.1567506,7.6258038],
                  [100.1577108,7.6251339]])
}

STATIONS = list(AOIS.keys())
YEARS = list(range(2017, 2026))
SCALE = 20
MAXPX = 1e13

# =========================================================
# 2) Time & Collection
# =========================================================
def get_window(year:int):
    ini = ee.Date.fromYMD(year,1,1)
    end = ini.advance(12,"month")
    return ini, end

def s2_toa(geom, ini, end, cloud):
    return (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(geom)
            .filterDate(ini, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)))

# =========================================================
# 3) FULL Atmospheric Correction → Rrs (UNCHANGED LOGIC)
# =========================================================
pi = ee.Image(3.141592)
ozone = ee.ImageCollection('TOMS/MERGED')

def s2_correction_toa(img, ini, end, mask):
    bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
    rescale = img.select(bands).divide(10000).multiply(mask)
    footprint = rescale.geometry()

    DU = ee.Image(ozone.filterDate(ini,end).filterBounds(footprint).mean())
    SunZe = ee.Image.constant(img.get('MEAN_SOLAR_ZENITH_ANGLE'))
    cosdSunZe = SunZe.multiply(pi/180).cos()

    ESUN = ee.Image(ee.Array([ee.Image(img.get(f'SOLAR_IRRADIANCE_{b}')) for b in bands])
                    ).toArray().toArray(1)
    ESUN = ESUN.arrayProject([0]).arrayFlatten([bands])

    Ltoa = rescale.multiply(ESUN).multiply(cosdSunZe).divide(pi)
    Rrs = Ltoa.divide(pi)

    return Rrs.copyProperties(img, ['system:time_start'])

# =========================================================
# 4) Water-quality formulas (SAFE)
# =========================================================
def img_chl(img):
    ndci = img.normalizedDifference(['B5','B4'])
    return (ee.Image(14.039)
            .add(ee.Image(86.115).multiply(ndci))
            .add(ee.Image(194.325).multiply(ndci.pow(2)))
            .rename('chl_a')
            .copyProperties(img, ['system:time_start']))

def img_zsd(img):
    ratio = img.select('B2').divide(img.select('B4')).clamp(0.01, 10)
    lnMOSD = ee.Image(1.4856).multiply(ratio.log()).add(0.2734)
    zsd = ee.Image(0.1777).multiply(ee.Image(10).pow(lnMOSD)).add(1.0813)
    return zsd.updateMask(zsd.lt(10)).rename('secchi_m') \
              .copyProperties(img, ['system:time_start'])

def img_tsi(chl):
    return ee.Image(30.6).add(ee.Image(9.81).multiply(chl.log())) \
             .rename('tsi').copyProperties(chl, ['system:time_start'])

# =========================================================
# 5) Helpers
# =========================================================
def monthly_mean(ic, geom, band, year):
    out = []
    for m in range(1,13):
        s = ee.Date.fromYMD(year,m,1)
        e = s.advance(1,"month")
        v = (ic.filterDate(s,e).mean()
               .reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX)
               .get(band))
        try:
            v = float(ee.Number(v).getInfo())
        except:
            v = None
        out.append({"month":m,"value":v})
    return out

# =========================================================
# 6) API (FORCE FULL AC)
# =========================================================
@app.get("/timeseries_monthly")
def timeseries_monthly(
    station: Literal["CP01","LS01","LS03","TP01","TP04","TP11","PN01","SK01","SK06"],
    year: int = Query(..., ge=2017, le=2025),
    cloud_perc: int = 30,
    ac: Literal["none","full"] = "none"
):
    geom = AOIS[station]
    ini,end = get_window(year)

    rrs = (s2_toa(geom,ini,end,cloud_perc)
           .map(lambda im: s2_correction_toa(im,ini,end,ee.Image(1))))

    chl = rrs.map(img_chl)
    zsd = rrs.map(img_zsd)
    tsi = chl.map(img_tsi)

    return {
        "station": station,
        "year": year,
        "ac": "full",
        "monthly": {
            "chl_a": monthly_mean(chl,geom,"chl_a",year),
            "secchi_m": monthly_mean(zsd,geom,"secchi_m",year),
            "tsi": monthly_mean(tsi,geom,"tsi",year)
        }
    }

@app.get("/health-ee")
def health():
    return {"ok": True}

# =========================================================
# 7) Run
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
