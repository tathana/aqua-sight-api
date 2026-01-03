# =========================================================
# Aqua Sight API — Production Safe (Render-ready)
# Logic aligned to GEE UI (Code B)
# =========================================================

import os, json, base64, tempfile, re
from typing import Dict, List, Literal, Any, Optional

import ee
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import requests

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
# App
# =========================================================
app = FastAPI(title="Aqua Sight API", version="1.2.0")

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
# 1) AOIs
# =========================================================
def poly(coords): return ee.Geometry.Polygon(coords)

AOIS = {
    "CP01": poly([[99.24686562565613,10.445791568889858],[99.24744498280334,10.445707160671937],[99.24735915211487,10.442373017728443],[99.24484860447693,10.439418683974635],[99.2461575224762,10.439671913681973],[99.24770247486877,10.442330813158181],[99.24804579762268,10.445749364783767],[99.24873244313049,10.445960385256942],[99.24843203572082,10.448070582107523],[99.24692999867248,10.448049480209948],[99.24686562565613,10.445791568889858]]),
    "LS01": poly([[99.15535572624208,9.94453648425491],[99.14522770500184,9.94033053669308],[99.14578560447694,9.939971232131072],[99.15164354896547,9.940985738113076],[99.15320995903016,9.94231727243655],[99.15604237174989,9.94314155287098],[99.15535572624208,9.94453648425491]]),
    "LS03": poly([[99.06211096162414,9.953918743528815],[99.06337696427917,9.953411511011419],[99.06495410317993,9.953062788198586],[99.06823712701416,9.95331640482666],[99.0680654656372,9.953622857989421],[99.06404215211487,9.953665127368613],[99.06225043649292,9.954278032751825],[99.06211096162414,9.953918743528815]]),
    "TP01": poly([[99.37406626983643,9.189275494872142],[99.3680795793152,9.185886312564055],[99.37121239944459,9.184001065872005],[99.3819841508484,9.191160713494348],[99.37844363494874,9.193744929453619],[99.37406626983643,9.189275494872142]]),
    "TP04": poly([[99.17458631484986,9.088432491599194],[99.16994072883607,9.08680098982789],[99.16754819839478,9.08496819001516],[99.16601397483826,9.082510316590984],[99.16745163887025,9.082171298244077],[99.17001583068848,9.08557206150935],[99.1720006653595,9.086705642091964],[99.17572357147218,9.08743664075224],[99.17458631484986,9.088432491599194]]),
    "TP11": poly([[99.60781251982117,8.532800202326145],[99.61010849073791,8.53450842130176],[99.61061274603271,8.535802843013302],[99.61061274603271,8.536333342446767],[99.60976516798401,8.534837332152573],[99.6075657565918,8.53357473734279],[99.60781251982117,8.532800202326145]]),
    "PN01": poly([[99.90849795568084,7.891517643958013],[99.91079392659759,7.89343054126273],[99.91115870702362,7.894333850801101],[99.90807953107452,7.8920490052084915],[99.90849795568084,7.891517643958013]]),
    "SK01": poly([[100.12514956733038,7.789004657536354],[100.12512878021052,7.788904338311305],[100.12580737909128,7.788852517775042],[100.12580939074805,7.788863147629168],[100.1251522495394,7.7889202830904],[100.1251710250025,7.7890053219020015],[100.12514956733038,7.789004657536354]]),
    "SK06": poly([[100.15771078416547,7.625133904695956],[100.15905188867292,7.62493717628797],[100.15962051698408,7.624458647350537],[100.15961515256605,7.625564580084231],[100.15859591314039,7.625330633013513],[100.15675055333814,7.625803844001549],[100.15771078416547,7.625133904695956]])
}

STATIONS_META = {
    "CP01":"CP01 Chumphon River","LS01":"LS01 Lower Lang Suan River","LS03":"LS03 Upper Lang Suan River",
    "TP01":"TP01 Lower Tapee River","TP04":"TP04 Phum Duang River","TP11":"TP011 Upper Tapee River",
    "PN01":"PN01 Pak Phanang River","SK01":"SK01 Thale Noi","SK06":"SK06 Thalaluang"
}

YEARS = list(range(2017, 2026))

# =========================================================
# 2) Helpers
# =========================================================
SCALE = 20
MAXPX = 1e13

def get_window(year:int):
    ini = ee.Date.fromYMD(year,1,1)
    end = ini.advance(12,"month")
    return ini, end

def build_water_mask(geom):
    SRP = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    def scale(img):
        opt = img.select('SR_B.*').multiply(0.0000275).add(-0.2)
        return img.addBands(opt, None, True)
    SRP = SRP.map(scale)
    m = SRP.filterBounds(geom)\
           .select('SR_B6')\
           .filter(ee.Filter.lt('CLOUD_COVER',30))\
           .filter(ee.Filter.calendarRange(1,1,'month'))\
           .filter(ee.Filter.calendarRange(2021,2025,'year'))\
           .median()\
           .lt(300)
    return m.updateMask(m)

def s2_sr(geom, ini, end, cloud):
    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
             .filterBounds(geom)\
             .filterDate(ini,end)\
             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud))

def s2_toa(geom, ini, end, cloud):
    return ee.ImageCollection("COPERNICUS/S2_HARMONIZED")\
             .filterBounds(geom)\
             .filterDate(ini,end)\
             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud))

def add_scaled(img, mask):
    bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
    return img.select(bands).divide(10000).multiply(mask)\
              .copyProperties(img, ['system:time_start'])

# =========================================================
# 3) Water Quality (MATCH Code B)
# =========================================================
def img_pH(img):
    ph = ee.Image(8.339).subtract(ee.Image(0.827).multiply(img.select('B1').divide(img.select('B8'))))
    return ph.updateMask(ph.lt(14)).rename('pH').copyProperties(img, ['system:time_start'])

def img_turb(img):
    ndvi = img.normalizedDifference(['B8','B4'])
    tur = ee.Image(100).multiply(ee.Image(1).subtract(ndvi))
    return tur.updateMask(tur.lt(100)).rename('turbidity').copyProperties(img, ['system:time_start'])

def img_sal(img):
    sal = img.normalizedDifference(['B11','B12'])
    return sal.updateMask(sal.abs().lt(1)).rename('salinity_idx').copyProperties(img, ['system:time_start'])

def img_do(img):
    do = (ee.Image(-0.0167).multiply(img.select('B8'))
          .add(ee.Image(0.0067).multiply(img.select('B9')))
          .add(ee.Image(0.0083).multiply(img.select('B11')))
          .add(9.577))
    return do.updateMask(do.lt(20)).rename('do_mgL').copyProperties(img, ['system:time_start'])

def img_chl(img):
    ndci = img.normalizedDifference(['B5','B4'])
    chl = ee.Image(14.039)\
          .add(ee.Image(86.115).multiply(ndci))\
          .add(ee.Image(194.325).multiply(ndci.pow(2)))
    return chl.updateMask(chl.lt(100)).rename('chl_a').copyProperties(img, ['system:time_start'])

def img_zsd(img):
    blueRed = img.select('B2').divide(img.select('B4')).log()
    lnMOSD = ee.Image(1.4856).multiply(blueRed).add(0.2734)
    zsd = ee.Image(0.1777).multiply(ee.Image(10).pow(lnMOSD)).add(1.0813)
    return zsd.updateMask(zsd.lt(10)).rename('secchi_m').copyProperties(img, ['system:time_start'])

def img_tsi_from_chl(chl):
    tsi = ee.Image(30.6).add(ee.Image(9.81).multiply(chl.log()))
    return tsi.updateMask(tsi.lt(200)).rename('tsi').copyProperties(chl, ['system:time_start'])

def tsi_reclass(tsi):
    img = tsi
    return img.where(img.lt(30),1)\
              .where(img.gte(30).And(img.lt(40)),2)\
              .where(img.gte(40).And(img.lt(50)),3)\
              .where(img.gte(50).And(img.lt(60)),4)\
              .where(img.gte(60).And(img.lt(70)),5)\
              .where(img.gte(70).And(img.lt(80)),6)\
              .where(img.gte(80),7)\
              .rename('tsi_class')\
              .copyProperties(tsi,['system:time_start'])

# =========================================================
# 4) Scene-based Monthly (MATCH GEE UI)
# =========================================================
def monthly_series(ic, geom, band, year):
    out = []
    for m in range(1,13):
        start = ee.Date.fromYMD(year,m,1)
        end   = start.advance(1,"month")
        month_ic = ic.filterDate(start,end)

        def per_img(img):
            r = ee.Reducer.mode() if band=='tsi_class' else ee.Reducer.mean()
            v = img.select(band).reduceRegion(r, geom, SCALE, maxPixels=MAXPX).get(band)
            return ee.Feature(None, {"v": v})

        fc = ee.FeatureCollection(month_ic.map(per_img))
        reducer = ee.Reducer.mode() if band=='tsi_class' else ee.Reducer.mean()
        key = 'mode' if band=='tsi_class' else 'mean'
        val = fc.reduceColumns(reducer, ['v']).get(key)

        try:
            vpy = float(ee.Number(val).getInfo()) if val else None
        except:
            vpy = None

        out.append({"month": m, "value": vpy})
    return out

# =========================================================
# 5) Endpoints (UNCHANGED)
# =========================================================
@app.get("/stations")
def stations():
    return [{"code": k, "name": STATIONS_META[k]} for k in AOIS]

@app.get("/years")
def years():
    return YEARS

@app.get("/timeseries_monthly")
def timeseries_monthly(
    station: str,
    year: int,
    cloud_perc: int = 30,
    ac: Literal["none","full"] = "none"
):
    geom = AOIS[station]
    ini,end = get_window(year)
    mask = build_water_mask(geom)

    sr = s2_sr(geom,ini,end,cloud_perc).map(lambda i: add_scaled(i,mask))
    toa = s2_toa(geom,ini,end,cloud_perc).map(lambda i: add_scaled(i,mask))

    base_chl = toa if ac=="full" else sr
    base_phy = sr

    ph  = base_phy.map(img_pH)
    tur = base_phy.map(img_turb)
    sal = base_phy.map(img_sal)
    dox = base_phy.map(img_do)
    chl = base_chl.map(img_chl)
    zsd = base_chl.map(img_zsd)
    tsi = chl.map(img_tsi_from_chl)
    tsi_cls = tsi.map(tsi_reclass)

    return {
        "station":station,"year":year,"cloud_perc":cloud_perc,"ac":ac,
        "monthly":{
            "pH":monthly_series(ph,geom,"pH",year),
            "turbidity":monthly_series(tur,geom,"turbidity",year),
            "salinity_idx":monthly_series(sal,geom,"salinity_idx",year),
            "do_mgL":monthly_series(dox,geom,"do_mgL",year),
            "chl_a":monthly_series(chl,geom,"chl_a",year),
            "secchi_m":monthly_series(zsd,geom,"secchi_m",year),
            "tsi":monthly_series(tsi,geom,"tsi",year),
            "tsi_class":monthly_series(tsi_cls,geom,"tsi_class",year)
        }
    }

@app.get("/summary_year")
def summary_year(
    station: str,
    year: int,
    cloud_perc: int = 30,
    ac: Literal["none","full"] = "none"
):
    geom = AOIS[station]
    ini,end = get_window(year)
    mask = build_water_mask(geom)

    sr = s2_sr(geom,ini,end,cloud_perc).map(lambda i: add_scaled(i,mask))
    toa = s2_toa(geom,ini,end,cloud_perc).map(lambda i: add_scaled(i,mask))

    base_chl = toa if ac=="full" else sr
    base_phy = sr

    def mean(ic, band):
        try:
            return float(ic.mean().reduceRegion(
                ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX
            ).get(band).getInfo())
        except:
            return None

    chl = base_chl.map(img_chl)
    tsi = chl.map(img_tsi_from_chl)

    return {
        "station":station,"year":year,"cloud_perc":cloud_perc,"ac":ac,
        "mean":{
            "pH":mean(base_phy.map(img_pH),'pH'),
            "turbidity":mean(base_phy.map(img_turb),'turbidity'),
            "salinity_idx":mean(base_phy.map(img_sal),'salinity_idx'),
            "do_mgL":mean(base_phy.map(img_do),'do_mgL'),
            "chl_a":mean(chl,'chl_a'),
            "secchi_m":mean(base_chl.map(img_zsd),'secchi_m'),
            "tsi":mean(tsi,'tsi')
        }
    }

@app.get("/")
def root():
    return {"name":"Aqua Sight API"}
