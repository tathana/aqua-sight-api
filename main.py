# =========================================================
# Aqua Sight API — FULL VERSION (100%)
# Production Safe | Web + Mobile + LINE + Academic
# =========================================================

import os, json, base64, tempfile, re
from typing import Dict, List, Literal, Optional, Any

import ee
import requests

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
        ee.Initialize(ee.ServiceAccountCredentials(
            info["client_email"], key_data=cred_json
        ))
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
app = FastAPI(title="Aqua Sight API", version="2.0.0")

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
    return ini, ini.advance(12,"month")

def build_water_mask(geom):
    SRP = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    SRP = SRP.map(lambda i: i.addBands(
        i.select('SR_B.*').multiply(0.0000275).add(-0.2), None, True))
    m = (SRP.filterBounds(geom)
         .select('SR_B6')
         .filter(ee.Filter.lt('CLOUD_COVER',30))
         .median()
         .lt(300))
    return m.updateMask(m)

def s2_sr(geom, ini, end, cloud):
    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
        .filterBounds(geom).filterDate(ini,end)\
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud))

def s2_toa(geom, ini, end, cloud):
    return ee.ImageCollection("COPERNICUS/S2_HARMONIZED")\
        .filterBounds(geom).filterDate(ini,end)\
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud))

def add_scaled(img, mask):
    bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
    return img.select(bands).divide(10000).multiply(mask)\
        .copyProperties(img,['system:time_start'])

def build_collections(geom, ini, end, cloud, mask, ac):
    sr = s2_sr(geom,ini,end,cloud).map(lambda i: add_scaled(i,mask))
    out = {"sr": sr}
    if ac=="full":
        out["toa"] = s2_toa(geom,ini,end,cloud).map(lambda i: add_scaled(i,mask))
    return out

# =========================================================
# 3) Water Quality
# =========================================================
def img_pH(i): return ee.Image(8.339).subtract(0.827*i.select('B1').divide(i.select('B8'))).rename('pH')
def img_turb(i): return ee.Image(100).multiply(ee.Image(1).subtract(i.normalizedDifference(['B8','B4']))).rename('turbidity')
def img_sal(i): return i.normalizedDifference(['B11','B12']).rename('salinity_idx')
def img_do(i): return (-0.0167*i.select('B8') + 0.0067*i.select('B9') + 0.0083*i.select('B11') + 9.577).rename('do_mgL')
def img_chl(i):
    ndci=i.normalizedDifference(['B5','B4'])
    return (14.039+86.115*ndci+194.325*ndci.pow(2)).rename('chl_a')
def img_zsd(i):
    ln=(1.4856*(i.select('B2').divide(i.select('B4')).log())+0.2734)
    return (0.1777*(ee.Image(10).pow(ln))+1.0813).rename('secchi_m')
def img_tsi(chl): return (30.6+9.81*chl.log()).rename('tsi')
def tsi_class(tsi):
    return tsi.where(tsi.lt(30),1).where(tsi.gte(30).And(tsi.lt(40)),2)\
        .where(tsi.gte(40).And(tsi.lt(50)),3).where(tsi.gte(50).And(tsi.lt(60)),4)\
        .where(tsi.gte(60).And(tsi.lt(70)),5).where(tsi.gte(70).And(tsi.lt(80)),6)\
        .where(tsi.gte(80),7).rename('tsi_class')

# =========================================================
# 4) Series
# =========================================================
def monthly_series(ic, geom, band, year):
    out=[]
    for m in range(1,13):
        s=ee.Date.fromYMD(year,m,1)
        e=s.advance(1,"month")
        v=ic.filterDate(s,e).mean().reduceRegion(
            ee.Reducer.mean(),geom,SCALE,MAXPX).get(band)
        try: v=float(ee.Number(v).getInfo())
        except: v=None
        out.append({"month":m,"value":v})
    return out

def scenes_series(ic, geom, band):
    def f(img):
        v=img.select(band).reduceRegion(
            ee.Reducer.mean(),geom,SCALE,MAXPX).get(band)
        return ee.Feature(None,{
            "date":ee.Date(img.get("system:time_start")).format("YYYY-MM-dd"),
            "value":v})
    fc=ee.FeatureCollection(ic.map(f)).getInfo()["features"]
    out=[]
    for r in fc:
        p=r["properties"]
        out.append({"date":p["date"],
                    "value":float(p["value"]) if p["value"] else None})
    return sorted(out,key=lambda x:x["date"])

# =========================================================
# 5) Endpoints
# =========================================================
@app.get("/")
def root():
    return {"name":"Aqua Sight API","version":"2.0.0"}

@app.get("/stations")
def stations():
    return [{"code":k,"name":STATIONS_META[k]} for k in AOIS]

@app.get("/years")
def years():
    return YEARS

@app.get("/timeseries_monthly")
def ts_monthly(station:str,year:int,cloud:int=30,ac:Literal["none","full"]="none"):
    if station not in AOIS: raise HTTPException(404,"Unknown station")
    geom=AOIS[station]; ini,end=get_window(year)
    mask=build_water_mask(geom)
    col=build_collections(geom,ini,end,cloud,mask,ac)
    base=col["toa"] if ac=="full" else col["sr"]
    chl=base.map(img_chl); tsi=chl.map(img_tsi)
    return {"station":station,"year":year,"monthly":{
        "chl_a":monthly_series(chl,geom,"chl_a",year),
        "secchi":monthly_series(base.map(img_zsd),geom,"secchi_m",year),
        "tsi":monthly_series(tsi,geom,"tsi",year)
    }}

@app.get("/timeseries_scenes")
def ts_scenes(station:str,year:int,cloud:int=30,ac:Literal["none","full"]="none"):
    if station not in AOIS: raise HTTPException(404,"Unknown station")
    geom=AOIS[station]; ini,end=get_window(year)
    mask=build_water_mask(geom)
    col=build_collections(geom,ini,end,cloud,mask,ac)
    base=col["toa"] if ac=="full" else col["sr"]
    chl=base.map(img_chl)
    return {"station":station,"year":year,"series":{
        "chl_a":scenes_series(chl,geom,"chl_a")
    }}

@app.get("/summary_year")
def summary(station:str,year:int,cloud:int=30,ac:Literal["none","full"]="none"):
    if station not in AOIS: raise HTTPException(404,"Unknown station")
    geom=AOIS[station]; ini,end=get_window(year)
    mask=build_water_mask(geom)
    col=build_collections(geom,ini,end,cloud,mask,ac)
    base=col["toa"] if ac=="full" else col["sr"]
    chl=base.map(img_chl).mean().reduceRegion(
        ee.Reducer.mean(),geom,SCALE,MAXPX).get("chl_a")
    try: chl=float(ee.Number(chl).getInfo())
    except: chl=None
    return {"station":station,"year":year,"mean":{"chl_a":chl}}

@app.get("/map/tiles")
def map_tiles(station:str,year:int,cloud:int=30,ac:Literal["none","full"]="none"):
    if station not in AOIS: raise HTTPException(404,"Unknown station")
    geom=AOIS[station]; ini,end=get_window(year)
    mask=build_water_mask(geom)
    col=build_collections(geom,ini,end,cloud,mask,ac)
    base=col["toa"] if ac=="full" else col["sr"]
    chl=base.map(img_chl).mean().clip(geom)
    tile=ee.data.getMapId({"image":chl.visualize(
        min=0,max=40,
        palette=['darkblue','blue','cyan','limegreen','yellow','orange','orangered','darkred']
    )})
    return {"tiles":{"chl":tile["tile_fetcher"].url_format}}

@app.get("/map/png")
def map_png(station:str,year:int):
    geom=AOIS[station]; ini,end=get_window(year)
    img=s2_sr(geom,ini,end,30).map(lambda i:add_scaled(i,build_water_mask(geom)))\
        .map(img_chl).mean().clip(geom)
    url=img.visualize(min=0,max=40).getThumbURL({"dimensions":"1024x1024","region":geom})
    return {"png":url}

@app.get("/map/png_proxy")
def map_png_proxy(station:str,year:int,scale:int=60):
    geom=AOIS[station]; ini,end=get_window(year)
    img=s2_sr(geom,ini,end,30).map(lambda i:add_scaled(i,build_water_mask(geom)))\
        .map(img_chl).mean().clip(geom)
    png=ee.data.getThumbnail({
        "image":img.visualize(min=0,max=40),
        "region":json.dumps(geom.bounds().getInfo()),
        "scale":scale,"format":"png"
    },60000)
    return Response(png,media_type="image/png")

@app.get("/health-ee")
def health():
    try:
        return {"ok":True,"assets":ee.data.getAssetRoots()}
    except Exception as e:
        return {"ok":False,"error":str(e)}
