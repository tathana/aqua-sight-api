# main.py — Aqua Sight API (monthly agg + min_images + safe parsing + tsi_class mode)

import os, json, base64, tempfile
from typing import Dict, List, Literal, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ee

# ---------- 0) Earth Engine Init ----------
cred_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
SA_EMAIL  = os.getenv("EE_SERVICE_ACCOUNT")
KEY_B64   = os.getenv("EE_KEY_B64")
KEY_PATH  = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

try:
    if cred_json:
        info  = json.loads(cred_json)
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

app = FastAPI(title="Aqua Sight API", version="1.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGIN", "*").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------- 1) AOIs ----------
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

# ---------- 2) Time windows & base helpers ----------
def get_window(year:int):
    ini = ee.Date.fromYMD(year,1,1); end = ini.advance(12,"month")
    return ini, end

def build_water_mask(geom: ee.Geometry) -> ee.Image:
    # Simple LS9 SR_B6 threshold mask (no scale) — consistent with earlier behavior
    SRP = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").select('SR_B6')
    col = (SRP.filterBounds(geom)
             .filter(ee.Filter.lt('CLOUD_COVER',30))
             .filter(ee.Filter.calendarRange(1,1,'month'))
             .filter(ee.Filter.calendarRange(2021,2025,'year')))
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
    out = (ee.Image(0)
           .where(mask1, 1).where(mask2, 2).where(mask3, 3)
           .where(mask4, 4).where(mask5, 5).where(mask6, 6).where(mask7, 7))
    return out.rename('tsi_class').copyProperties(tsi_img, ['system:time_start'])

# ---------- 4) Series helpers ----------
SCALE = 20
MAXPX = 1e13

def _safe_float(x) -> Optional[float]:
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None

def _reduce_value(img: ee.Image, geom: ee.Geometry, band: str):
    """ReduceRegion สำหรับ 1 รูป:
       - ถ้า band == 'tsi_class' ใช้ mode แล้วคืนค่าคีย์ 'mode'
       - อื่น ๆ ใช้ mean แล้วคืนค่าคีย์เป็นชื่อ band
    """
    if band == 'tsi_class':
        reduced = img.select(band).reduceRegion(ee.Reducer.mode(), geom, SCALE, maxPixels=MAXPX)
        key, val = 'mode', reduced.get('mode')
    else:
        reduced = img.select(band).reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX)
        key, val = band, reduced.get(band)
    return key, val

def monthly_series(
    ic: ee.ImageCollection,
    geom: ee.Geometry,
    band: str,
    year: int,
    agg: Literal["mean","median","scene"]="mean",
    min_images: int = 1
) -> List[Dict[str, Any]]:
    """
    agg = mean|median  -> รวมภาพทั้งเดือนแล้ว reduceRegion
    agg = scene        -> คำนวณค่าต่อ scene แล้วเฉลี่ยใน Python
    min_images        -> ถ้าจำนวน scene < min_images ให้คืน None
    """
    out: List[Dict[str, Any]] = []
    for m in range(1, 12+1):
        start = ee.Date.fromYMD(year, m, 1)
        end   = start.advance(1, "month")
        month_ic = ic.filterDate(start, end)

        # นับจำนวนภาพ
        try:
            count = int(ee.Number(month_ic.size()).getInfo())
        except Exception:
            count = 0

        if count < min_images:
            out.append({"month": m, "value": None})
            continue

        if agg in ("mean","median"):
            img = month_ic.mean() if agg == "mean" else month_ic.median()
            _, val = _reduce_value(img, geom, band)
            out.append({"month": m, "value": _safe_float(ee.Number(val).getInfo() if val is not None else None)})
        else:
            # agg = 'scene' : คิดค่าต่อภาพก่อน แล้วค่อยเฉลี่ยใน Python
            def per_img(image):
                # ใช้ reducer ให้สอดคล้องกับชนิด band
                if band == 'tsi_class':
                    v = image.select(band).reduceRegion(ee.Reducer.mode(), geom, SCALE, maxPixels=MAXPX).get('mode')
                else:
                    v = image.select(band).reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get(band)
                return ee.Feature(None, {"v": v})

            feats = ee.FeatureCollection(month_ic.map(per_img)).getInfo().get("features", [])
            vals: List[float] = []
            for f in feats:
                props = f.get("properties") or {}
                fv = _safe_float(props.get("v"))  # ป้องกัน KeyError
                if fv is not None:
                    vals.append(fv)
            out.append({"month": m, "value": (sum(vals)/len(vals) if len(vals) else None)})
    return out

def scenes_series(ic: ee.ImageCollection, geom: ee.Geometry, band: str) -> List[Dict[str, Any]]:
    def per_image(img):
        if band == 'tsi_class':
            v = img.select(band).reduceRegion(ee.Reducer.mode(), geom, SCALE, maxPixels=MAXPX).get('mode')
        else:
            v = img.select(band).reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get(band)
        return ee.Feature(None, {
            "date": ee.Date(img.get('system:time_start')).format("YYYY-MM-dd"),
            "value": v
        })
    fc = ee.FeatureCollection(ic.map(per_image)).getInfo().get("features", [])
    out: List[Dict[str, Any]] = []
    for f in fc:
        p = f.get("properties") or {}
        d = p.get("date")
        v = _safe_float(p.get("value"))
        if d is not None:
            out.append({"date": d, "value": v})
    out.sort(key=lambda x: x["date"])
    return out

# ---------- 5) Schemas ----------
class MonthlyPoint(BaseModel):
    month: int
    value: Optional[float] = None

class ScenePoint(BaseModel):
    date: str
    value: Optional[float] = None

class TSMonthlyResponse(BaseModel):
    station: str
    year: int
    cloud_perc: int
    ac: Literal["none","full"]
    agg: Literal["mean","median","scene"]
    min_images: int
    monthly: Dict[str, List[MonthlyPoint]]

class TSScenesResponse(BaseModel):
    station: str
    year: int
    cloud_perc: int
    ac: Literal["none","full"]
    series: Dict[str, List[ScenePoint]]

# ---------- 6) Build collections ----------
def build_collections(geom, ini, end, cloud_perc, mask):
    # ใช้ S2 SR + scale + water mask (โหมด AC แบบง่าย)
    sr_scaled = s2_sr(geom, ini, end, cloud_perc).map(lambda im: add_scaled(im, mask))
    return sr_scaled

# ---------- 7) Endpoints ----------
@app.get("/stations")
def stations():
    return [{"code": k, "name": STATIONS_META[k]} for k in AOIS.keys()]

@app.get("/years")
def years():
    return YEARS

@app.get("/timeseries_monthly", response_model=TSMonthlyResponse)
def timeseries_monthly(
    station: Literal["CP01","LS01","LS03","TP01","TP04","TP11","PN01","SK01","SK06"],
    year: int = Query(..., ge=2017, le=2025),
    cloud_perc: int = Query(30, ge=0, le=100),
    ac: Literal["none","full"] = "none",               # reserved (ยังไม่เปิดใช้ full AC ในไฟล์นี้)
    agg: Literal["mean","median","scene"] = "mean",
    min_images: int = Query(1, ge=1, le=50)
):
    if station not in AOIS: raise HTTPException(404, "Unknown station")
    geom = AOIS[station]; ini, end = get_window(year)
    mask = build_water_mask(geom)
    col  = build_collections(geom, ini, end, cloud_perc, mask)

    # สร้างคอลเลกชันตัวชี้วัด
    ph_ic  = col.map(img_pH)
    tur_ic = col.map(img_turb)
    do_ic  = col.map(img_do)
    chl_ic = col.map(img_chl)
    zsd_ic = col.map(img_zsd)
    tsi_ic = chl_ic.map(img_tsi_from_chl)
    tsi_cls_ic = tsi_ic.map(tsi_reclass)
    sal_ic = col.map(img_sal)

    return {
        "station": station, "year": year, "cloud_perc": cloud_perc, "ac": ac,
        "agg": agg, "min_images": min_images,
        "monthly": {
            "pH":           monthly_series(ph_ic,  geom, "pH",        year, agg, min_images),
            "turbidity":    monthly_series(tur_ic, geom, "turbidity", year, agg, min_images),
            "salinity_idx": monthly_series(sal_ic, geom, "salinity_idx", year, agg, min_images),
            "do_mgL":       monthly_series(do_ic,  geom, "do_mgL",    year, agg, min_images),
            "chl_a":        monthly_series(chl_ic, geom, "chl_a",     year, agg, min_images),
            "secchi_m":     monthly_series(zsd_ic, geom, "secchi_m",  year, agg, min_images),
            "tsi":          monthly_series(tsi_ic, geom, "tsi",       year, agg, min_images),
            "tsi_class":    monthly_series(tsi_cls_ic, geom, "tsi_class", year, agg, min_images)
        }
    }

@app.get("/timeseries_scenes", response_model=TSScenesResponse)
def timeseries_scenes(
    station: Literal["CP01","LS01","LS03","TP01","TP04","TP11","PN01","SK01","SK06"],
    year: int = Query(..., ge=2017, le=2025),
    cloud_perc: int = Query(30, ge=0, le=100),
    ac: Literal["none","full"] = "none"
):
    if station not in AOIS: raise HTTPException(404, "Unknown station")
    geom = AOIS[station]; ini, end = get_window(year)
    mask = build_water_mask(geom)
    col  = build_collections(geom, ini, end, cloud_perc, mask)

    ph_ic  = col.map(img_pH)
    tur_ic = col.map(img_turb)
    do_ic  = col.map(img_do)
    chl_ic = col.map(img_chl)
    zsd_ic = col.map(img_zsd)
    tsi_ic = chl_ic.map(img_tsi_from_chl)
    tsi_cls_ic = tsi_ic.map(tsi_reclass)
    sal_ic = col.map(img_sal)

    return {
        "station": station, "year": year, "cloud_perc": cloud_perc, "ac": ac,
        "series": {
            "pH":           scenes_series(ph_ic,  geom, "pH"),
            "turbidity":    scenes_series(tur_ic, geom, "turbidity"),
            "salinity_idx": scenes_series(sal_ic, geom, "salinity_idx"),
            "do_mgL":       scenes_series(do_ic,  geom, "do_mgL"),
            "chl_a":        scenes_series(chl_ic, geom, "chl_a"),
            "secchi_m":     scenes_series(zsd_ic, geom, "secchi_m"),
            "tsi":          scenes_series(tsi_ic, geom, "tsi"),
            "tsi_class":    scenes_series(tsi_cls_ic, geom, "tsi_class")
        }
    }

@app.get("/")
def root():
    return {"name":"Aqua Sight API",
            "examples":{
                "monthly":"/timeseries_monthly?station=CP01&year=2024&agg=scene&min_images=1",
                "scenes":"/timeseries_scenes?station=CP01&year=2024"}}

@app.get("/health-ee")
def health_ee():
    try:
        return {"ok": True, "roots": ee.data.getAssetRoots()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
