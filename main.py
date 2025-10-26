# ver แรก main.py — Aqua Sight API (TSI reclass + per-scene series + full atmospheric correction)

import os, json, base64, tempfile, re
from typing import Dict, List, Literal, Any, Optional

import requests
import ee

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel


# ---------- 0) Earth Engine Init ----------
cred_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
SA_EMAIL  = os.getenv("EE_SERVICE_ACCOUNT")         # เช่น xxx@project.iam.gserviceaccount.com
KEY_B64   = os.getenv("EE_KEY_B64")                 # คีย์ทั้งไฟล์ (base64)
KEY_PATH  = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # path ไฟล์คีย์ (.json)

try:
    if cred_json:
        # วิธีที่ง่ายและพลาดยากสุด: วาง JSON ทั้งไฟล์ลง env เดียว
        info  = json.loads(cred_json)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=cred_json)
        ee.Initialize(creds)
    elif SA_EMAIL and KEY_B64:
        # ทางเลือก: อีเมล + คีย์ base64
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.write(base64.b64decode(KEY_B64))
        tmp.flush()
        creds = ee.ServiceAccountCredentials(SA_EMAIL, tmp.name)
        ee.Initialize(creds)
    elif SA_EMAIL and KEY_PATH:
        # ทางเลือก: อีเมล + path ไฟล์คีย์
        creds = ee.ServiceAccountCredentials(SA_EMAIL, KEY_PATH)
        ee.Initialize(creds)
    else:
        # ไม่มี credential ที่ใช้ได้ -> แจ้งเตือนให้ตั้งค่า env
        raise RuntimeError(
            "Missing Earth Engine credentials. "
            "Set GOOGLE_APPLICATION_CREDENTIALS_JSON (recommended) "
            "หรือ EE_SERVICE_ACCOUNT + EE_KEY_B64 / GOOGLE_APPLICATION_CREDENTIALS."
        )
except Exception as e:
    raise RuntimeError(f"Earth Engine init failed: {e}")

app = FastAPI(title="Aqua Sight API", version="1.2.0")

# CORS: ตั้งได้หลายโดเมนด้วยจุลภาค
allowed = os.getenv("ALLOWED_ORIGIN", "*")
origins = [o.strip() for o in allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
    # Landsat-9 SR (ม.ค. 2021–2025), CLOUD_COVER < 30, SR_B6 < 300
    SRP = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    def apply_scale(img):
        optical = img.select('SR_B.*').multiply(0.0000275).add(-0.2)
        thermal = img.select('ST_B.*').multiply(0.00341802).add(149.0)
        return img.addBands(optical, None, True).addBands(thermal, None, True)
    SRP = SRP.map(apply_scale)
    col = (SRP.filterBounds(geom).select('SR_B6')
             .filter(ee.Filter.lt('CLOUD_COVER',30))
             .filter(ee.Filter.calendarRange(1,1,'month'))
             .filter(ee.Filter.calendarRange(2021,2025,'year')))
    m = ee.Image(col.median().lt(300))
    return m.updateMask(m)

def s2_sr(geom, ini, end, cloud_perc:int=30):
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom).filterDate(ini, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_perc)))

def s2_toa(geom, ini, end, cloud_perc:int=30):
    return (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(geom).filterDate(ini, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_perc)))

def add_scaled(img: ee.Image, mask: ee.Image) -> ee.Image:
    bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
    return (img.select(bands).divide(10000).multiply(mask)
            .copyProperties(img, ['system:time_start']))

# ---------- 3) Atmospheric correction (full) ----------
pi = ee.Image(3.141592)
ozone = ee.ImageCollection('TOMS/MERGED')

def _s2_correction_common(img: ee.Image, bands: List[str], ini: ee.Date, end: ee.Date, mask: ee.Image):
    rescale = img.select(bands).divide(10000).multiply(mask)
    footprint = rescale.geometry()

    DEM = ee.Image('USGS/SRTMGL1_003').clip(footprint)
    DU  = ee.Image(ozone.filterDate(ini,end).filterBounds(footprint).mean())

    imgDate = ee.Date(img.get('system:time_start'))
    FOY = ee.Date.fromYMD(imgDate.get('year'),1,1)
    JD  = imgDate.difference(FOY,'day').int().add(1)

    myCos = ((ee.Image(0.0172).multiply(ee.Image(JD).subtract(ee.Image(2)))).cos()).pow(2)
    cosd  = myCos.multiply(pi.divide(ee.Image(180))).cos()
    d     = ee.Image(1).subtract(ee.Image(0.01673)).multiply(cosd).clip(footprint)

    SunAz = ee.Image.constant(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')).clip(footprint)
    SunZe = ee.Image.constant(img.get('MEAN_SOLAR_ZENITH_ANGLE')).clip(footprint)
    cosdSunZe = SunZe.multiply(pi.divide(ee.Image(180))).cos()
    sindSunZe = SunZe.multiply(pi.divide(ee.Image(180))).sin()

    SatZe = ee.Image.constant(img.get('MEAN_INCIDENCE_ZENITH_ANGLE_B5')).clip(footprint)
    cosdSatZe = SatZe.multiply(pi.divide(ee.Image(180))).cos()
    sindSatZe = SatZe.multiply(pi.divide(ee.Image(180))).sin()

    SatAz = ee.Image.constant(img.get('MEAN_INCIDENCE_AZIMUTH_ANGLE_B5')).clip(footprint)
    RelAz = SatAz.subtract(SunAz)
    cosdRelAz = RelAz.multiply(pi.divide(ee.Image(180))).cos()

    P  = (ee.Image(101325).multiply(ee.Image(1).subtract(ee.Image(0.0000225577).multiply(DEM)).pow(5.25588)).multiply(0.01)).multiply(mask)
    Po = ee.Image(1013.25)

    return {
        "rescale":rescale, "footprint":footprint, "DU":DU, "d":d,
        "SunZe":SunZe, "cosdSunZe":cosdSunZe, "sindSunZe":sindSunZe,
        "SatZe":SatZe, "cosdSatZe":cosdSatZe, "sindSatZe":sindSatZe,
        "cosdRelAz":cosdRelAz, "P":P, "Po":Po
    }

def s2_correction_toa(img: ee.Image, ini: ee.Date, end: ee.Date, mask: ee.Image) -> ee.Image:
    # ใช้กับ COPERNICUS/S2 (TOA)
    bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
    C = _s2_correction_common(img, bands, ini, end, mask)

    ESUN = ee.Image(ee.Array([ee.Image(img.get('SOLAR_IRRADIANCE_B1')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B2')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B3')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B4')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B5')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B6')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B7')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B8')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B8A')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B11')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B12'))])).toArray().toArray(1)
    ESUN = ESUN.multiply(ee.Image(1))
    ESUNImg = ESUN.arrayProject([0]).arrayFlatten([bands])

    imgArr = C["rescale"].select(bands).toArray().toArray(1)
    Ltoa = imgArr.multiply(ESUN).multiply(C["cosdSunZe"]).divide(pi.multiply(C["d"].pow(2)))

    bandCenter = ee.Image(443).divide(1000).addBands(ee.Image(490).divide(1000)) \
        .addBands(ee.Image(560).divide(1000)).addBands(ee.Image(665).divide(1000)) \
        .addBands(ee.Image(705).divide(1000)).addBands(ee.Image(740).divide(1000)) \
        .addBands(ee.Image(783).divide(1000)).addBands(ee.Image(842).divide(1000)) \
        .addBands(ee.Image(865).divide(1000)).addBands(ee.Image(1610).divide(1000)) \
        .addBands(ee.Image(2190).divide(1000)).toArray().toArray(1)

    koz = ee.Image(0.0039).addBands(ee.Image(0.0213)).addBands(ee.Image(0.1052)) \
        .addBands(ee.Image(0.0505)).addBands(ee.Image(0.0205)).addBands(ee.Image(0.0112)) \
        .addBands(ee.Image(0.0075)).addBands(ee.Image(0.0021)).addBands(ee.Image(0.0019)) \
        .addBands(ee.Image(0)).addBands(ee.Image(0)).toArray().toArray(1)
    Toz = koz.multiply(C["DU"]).divide(ee.Image(1000))
    Lt = Ltoa.multiply((Toz).multiply((ee.Image(1).divide(C["cosdSunZe"])).add(ee.Image(1).divide(C["cosdSatZe"])) ).exp())

    Tr = (C["P"].divide(C["Po"])).multiply(ee.Image(0.008569).multiply(bandCenter.pow(-4))) \
        .multiply((ee.Image(1).add(ee.Image(0.0113).multiply(bandCenter.pow(-2))).add(ee.Image(0.00013).multiply(bandCenter.pow(-4)))))

    theta_neg = ((C["cosdSunZe"].multiply(ee.Image(-1))).multiply(C["cosdSatZe"])) \
                .subtract((C["sindSunZe"]).multiply(C["sindSatZe"]).multiply(C["cosdRelAz"]))
    theta_neg_inv = theta_neg.acos().multiply(ee.Image(180).divide(pi))
    theta_pos = C["cosdSunZe"].multiply(C["cosdSatZe"]) \
        .subtract(C["sindSunZe"].multiply(C["sindSatZe"]).multiply(C["cosdRelAz"]))
    theta_pos_inv = theta_pos.acos().multiply(ee.Image(180).divide(pi))
    cosd_tni = theta_neg_inv.multiply(pi.divide(180)).cos()
    cosd_tpi = theta_pos_inv.multiply(pi.divide(180)).cos()
    Pr_neg = ee.Image(0.75).multiply(ee.Image(1).add(cosd_tni.pow(2)))
    Pr_pos = ee.Image(0.75).multiply(ee.Image(1).add(cosd_tpi.pow(2)))
    R_theta_SZ = ee.Image(0)  # ปล่อย 0 (ส่วน Fresnel ให้ค่าต่ำมาก)
    R_theta_V  = ee.Image(0)
    Pr = Pr_neg.add((R_theta_SZ.add(R_theta_V)).multiply(Pr_pos))
    denom = ee.Image(4).multiply(pi).multiply(C["cosdSatZe"])
    Lr = (ESUN.multiply(Tr)).multiply(Pr.divide(denom))
    Lrc = Lt.subtract(Lr)
    LrcImg = Lrc.arrayProject([0]).arrayFlatten([bands])

    bands_nm = ee.Image(443).addBands(ee.Image(490)).addBands(ee.Image(560)) \
        .addBands(ee.Image(665)).addBands(ee.Image(705)).addBands(ee.Image(740)) \
        .addBands(ee.Image(783)).addBands(ee.Image(842)).addBands(ee.Image(865)) \
        .addBands(ee.Image(0)).addBands(ee.Image(0)).toArray().toArray(1)

    Lam_10 = LrcImg.select('B11'); Lam_11 = LrcImg.select('B12')
    eps = (((Lam_11.divide(ESUNImg.select('B12'))).log()).subtract((Lam_10.divide(ESUNImg.select('B11'))).log())) \
          .divide(ee.Image(2190).subtract(ee.Image(1610)))
    Lam = (Lam_11).multiply((ESUN).divide(ESUNImg.select('B12'))).multiply((eps.multiply(ee.Image(-1))).multiply((bands_nm.divide(ee.Image(2190)))).exp())
    trans = Tr.multiply(ee.Image(-1)).divide(ee.Image(2)).multiply(ee.Image(1).divide(C["cosdSatZe"])).exp()
    Lw = Lrc.subtract(Lam).divide(trans)
    pw = (Lw.multiply(pi).multiply(C["d"].pow(2)).divide(ESUN.multiply(C["cosdSunZe"])))
    Rrs = (pw.divide(pi).arrayProject([0]).arrayFlatten([bands]).slice(0,12))
    return Rrs.set('system:time_start',img.get('system:time_start'))

def s2_correction_sr(img: ee.Image, ini: ee.Date, end: ee.Date, mask: ee.Image) -> ee.Image:
    # ใช้กับ COPERNICUS/S2_SR (รวม B9)
    bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
    C = _s2_correction_common(img, bands, ini, end, mask)

    ESUN = ee.Image(ee.Array([ee.Image(img.get('SOLAR_IRRADIANCE_B1')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B2')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B3')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B4')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B5')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B6')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B7')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B8')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B8A')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B9')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B11')),
                              ee.Image(img.get('SOLAR_IRRADIANCE_B12'))])).toArray().toArray(1)
    ESUN = ESUN.multiply(ee.Image(1))
    ESUNImg = ESUN.arrayProject([0]).arrayFlatten([bands])
    imgArr = C["rescale"].select(bands).toArray().toArray(1)
    Ltoa = imgArr.multiply(ESUN).multiply(C["cosdSunZe"]).divide(pi.multiply(C["d"].pow(2)))

    bandCenter = ee.Image(443).divide(1000).addBands(ee.Image(490).divide(1000)) \
        .addBands(ee.Image(560).divide(1000)).addBands(ee.Image(665).divide(1000)) \
        .addBands(ee.Image(705).divide(1000)).addBands(ee.Image(740).divide(1000)) \
        .addBands(ee.Image(783).divide(1000)).addBands(ee.Image(842).divide(1000)) \
        .addBands(ee.Image(865).divide(1000)).addBands(ee.Image(945).divide(1000)) \
        .addBands(ee.Image(1610).divide(1000)).addBands(ee.Image(2190).divide(1000)).toArray().toArray(1)

    koz = ee.Image(0.0039).addBands(ee.Image(0.0213)).addBands(ee.Image(0.1052)) \
        .addBands(ee.Image(0.0505)).addBands(ee.Image(0.0205)).addBands(ee.Image(0.0112)) \
        .addBands(ee.Image(0.0075)).addBands(ee.Image(0.0021)).addBands(ee.Image(0.0019)) \
        .addBands(ee.Image(0.0011)).addBands(ee.Image(0)).addBands(ee.Image(0)).toArray().toArray(1)
    Toz = koz.multiply(C["DU"]).divide(ee.Image(1000))
    Lt = Ltoa.multiply((Toz).multiply((ee.Image(1).divide(C["cosdSunZe"])).add(ee.Image(1).divide(C["cosdSatZe"])) ).exp())

    Tr = (C["P"].divide(C["Po"])).multiply(ee.Image(0.008569).multiply(bandCenter.pow(-4))) \
        .multiply((ee.Image(1).add(ee.Image(0.0113).multiply(bandCenter.pow(-2))).add(ee.Image(0.00013).multiply(bandCenter.pow(-4)))))

    theta_neg = ((C["cosdSunZe"].multiply(ee.Image(-1))).multiply(C["cosdSatZe"])) \
                .subtract((C["sindSunZe"]).multiply(C["sindSatZe"]).multiply(C["cosdRelAz"]))
    theta_neg_inv = theta_neg.acos().multiply(ee.Image(180).divide(pi))
    theta_pos = C["cosdSunZe"].multiply(C["cosdSatZe"]) \
        .subtract(C["sindSunZe"].multiply(C["sindSatZe"]).multiply(C["cosdRelAz"]))
    theta_pos_inv = theta_pos.acos().multiply(ee.Image(180).divide(pi))
    cosd_tni = theta_neg_inv.multiply(pi.divide(180)).cos()
    cosd_tpi = theta_pos_inv.multiply(pi.divide(180)).cos()
    Pr_neg = ee.Image(0.75).multiply(ee.Image(1).add(cosd_tni.pow(2)))
    Pr_pos = ee.Image(0.75).multiply(ee.Image(1).add(cosd_tpi.pow(2)))
    R_theta_SZ = ee.Image(0); R_theta_V = ee.Image(0)
    Pr = Pr_neg.add((R_theta_SZ.add(R_theta_V)).multiply(Pr_pos))
    denom = ee.Image(4).multiply(pi).multiply(C["cosdSatZe"])
    Lr = (ESUN.multiply(Tr)).multiply(Pr.divide(denom))
    Lrc = Lt.subtract(Lr)
    LrcImg = Lrc.arrayProject([0]).arrayFlatten([bands])

    bands_nm = ee.Image(443).addBands(ee.Image(490)).addBands(ee.Image(560)) \
        .addBands(ee.Image(665)).addBands(ee.Image(705)).addBands(ee.Image(740)) \
        .addBands(ee.Image(783)).addBands(ee.Image(842)).addBands(ee.Image(865)) \
        .addBands(ee.Image(945)).addBands(ee.Image(0)).addBands(ee.Image(0)).toArray().toArray(1)

    Lam_10 = LrcImg.select('B11'); Lam_11 = LrcImg.select('B12')
    eps = (((Lam_11.divide(ESUNImg.select('B12'))).log()).subtract((Lam_10.divide(ESUNImg.select('B11'))).log())) \
          .divide(ee.Image(2190).subtract(ee.Image(1610)))
    Lam = (Lam_11).multiply((ESUN).divide(ESUNImg.select('B12'))).multiply((eps.multiply(ee.Image(-1))).multiply((bands_nm.divide(ee.Image(2190)))).exp())
    trans = Tr.multiply(ee.Image(-1)).divide(ee.Image(2)).multiply(ee.Image(1).divide(C["cosdSatZe"])).exp()
    Lw = Lrc.subtract(Lam).divide(trans)
    pw = (Lw.multiply(pi).multiply(C["d"].pow(2)).divide(ESUN.multiply(C["cosdSunZe"])))
    Rrs = (pw.divide(pi).arrayProject([0]).arrayFlatten([bands]).slice(0,12))
    return Rrs.set('system:time_start',img.get('system:time_start'))

# ---------- 4) Water-quality formulas ----------
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

# ---------- 5) TSI reclass (1–7) ----------
def tsi_reclass(tsi_img: ee.Image) -> ee.Image:
    img = tsi_img
    mask1 = img.lt(30)
    mask2 = img.gte(30).And(img.lt(40))
    mask3 = img.gte(40).And(img.lt(50))
    mask4 = img.gte(50).And(img.lt(60))
    mask5 = img.gte(60).And(img.lt(70))
    mask6 = img.gte(70).And(img.lt(80))
    mask7 = img.gte(80)

    img1 = img.where(mask1.eq(1), 1).mask(mask1)
    img2 = img.where(mask2.eq(1), 2).mask(mask2)
    img3 = img.where(mask3.eq(1), 3).mask(mask3)
    img4 = img.where(mask4.eq(1), 4).mask(mask4)
    img5 = img.where(mask5.eq(1), 5).mask(mask5)
    img6 = img.where(mask6.eq(1), 6).mask(mask6)
    img7 = img.where(mask7.eq(1), 7).mask(mask7)

    out = img1.unmask(img2).unmask(img3).unmask(img4).unmask(img5).unmask(img6).unmask(img7)
    return out.rename('tsi_class').copyProperties(tsi_img, ['system:time_start'])

# ---------- 6) Series helpers ----------
SCALE = 20
MAXPX = 1e13

def monthly_series(ic: ee.ImageCollection, geom: ee.Geometry, band: str, year: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in range(1, 13):
        start = ee.Date.fromYMD(year, m, 1)
        end   = start.advance(1, "month")
        month_ic = ic.filterDate(start, end)

        safe_img = ee.Image(
            ee.Algorithms.If(
                month_ic.size().gt(0),
                (month_ic.mode() if band == 'tsi_class' else month_ic.mean()).clip(geom),
                ee.Image.constant(0).rename(band).updateMask(ee.Image(0)).clip(geom)
            )
        )

        reducer = ee.Reducer.mode() if band == 'tsi_class' else ee.Reducer.mean()
        val = ee.Image(safe_img).select(band).reduceRegion(reducer, geom, SCALE, maxPixels=MAXPX).get(band)
        vpy = None
        try:
            if val is not None:
                vpy = float(ee.Number(val).getInfo())
        except Exception:
            vpy = None
        out.append({"month": m, "value": vpy})
    return out

def scenes_series(ic: ee.ImageCollection, geom: ee.Geometry, band: str) -> List[Dict[str, Any]]:
    def per_image(img):
        reducer = ee.Reducer.mode() if band == 'tsi_class' else ee.Reducer.mean()
        val = img.select(band).reduceRegion(reducer, geom, SCALE, maxPixels=MAXPX).get(band)
        return ee.Feature(None, {"date": ee.Date(img.get('system:time_start')).format("YYYY-MM-dd"), "value": val})
    fc = ee.FeatureCollection(ic.map(per_image)).getInfo().get("features", [])
    out: List[Dict[str, Any]] = []
    for f in fc:
        p = f.get("properties", {})
        if p.get("date") is not None:
            v = p.get("value")
            out.append({"date": p["date"], "value": (float(v) if v is not None else None)})
    out.sort(key=lambda x: x["date"])
    return out

# ---------- 7) Schemas ----------
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
    monthly: Dict[str, List[MonthlyPoint]]

class TSScenesResponse(BaseModel):
    station: str
    year: int
    cloud_perc: int
    ac: Literal["none","full"]
    series: Dict[str, List[ScenePoint]]

class SummaryYearResponse(BaseModel):
    station: str
    year: int
    cloud_perc: int
    ac: Literal["none","full"]
    mean: Dict[str, Optional[float]]

class TileResponse(BaseModel):
    name: str
    tiles: Dict[str, str]

# ---------- 8) Build collections by mode ----------
def build_collections(geom, ini, end, cloud_perc, mask, ac: Literal["none","full"]):
    sr_scaled = s2_sr(geom, ini, end, cloud_perc).map(lambda im: add_scaled(im, mask))
    out = {"sr_scaled": sr_scaled}
    if ac == "full":
        toa_rrs = s2_toa(geom, ini, end, cloud_perc).map(lambda im: s2_correction_toa(im, ini, end, mask))
        sr_rrs  = s2_sr(geom, ini, end, cloud_perc).map(lambda im: s2_correction_sr(im,  ini, end, mask))
        out.update({"toa_rrs": toa_rrs, "sr_rrs": sr_rrs})
    return out

# ---------- 9) Endpoints ----------
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
    ac: Literal["none","full"] = "none"
):
    if station not in AOIS: raise HTTPException(404, "Unknown station")
    geom = AOIS[station]; ini, end = get_window(year)
    mask = build_water_mask(geom)
    cols = build_collections(geom, ini, end, cloud_perc, mask, ac)

    base_for_chl_sd = cols["toa_rrs"] if ac=="full" else cols["sr_scaled"]
    base_for_ph_tur_do = cols["sr_rrs"] if ac=="full" else cols["sr_scaled"]
    base_for_sal = cols["sr_scaled"]

    ph_ic  = base_for_ph_tur_do.map(img_pH)
    tur_ic = base_for_ph_tur_do.map(img_turb)
    do_ic  = base_for_ph_tur_do.map(img_do)
    chl_ic = base_for_chl_sd.map(img_chl)
    zsd_ic = base_for_chl_sd.map(img_zsd)
    tsi_ic = chl_ic.map(img_tsi_from_chl)
    tsi_cls_ic = tsi_ic.map(tsi_reclass)
    sal_ic = base_for_sal.map(img_sal)

    return {
        "station": station, "year": year, "cloud_perc": cloud_perc, "ac": ac,
        "monthly": {
            "pH":           monthly_series(ph_ic,  geom, "pH",        year),
            "turbidity":    monthly_series(tur_ic, geom, "turbidity", year),
            "salinity_idx": monthly_series(sal_ic, geom, "salinity_idx", year),
            "do_mgL":       monthly_series(do_ic,  geom, "do_mgL",    year),
            "chl_a":        monthly_series(chl_ic, geom, "chl_a",     year),
            "secchi_m":     monthly_series(zsd_ic, geom, "secchi_m",  year),
            "tsi":          monthly_series(tsi_ic, geom, "tsi",       year),
            "tsi_class":    monthly_series(tsi_cls_ic, geom, "tsi_class", year)
        }
    }

@app.get("/timeseries_scenes", response_model=TSScenesResponse)
def timeseries_scenes(
    station: Literal["CP01","LS01","LS03","TP01","TP04","TP11","PN01","SK01","SK06"],
    year: int = Query(..., ge=2017, le=2025),
    cloud_perc: int = Query(30, ge=0, le=100),
    ac: Literal["none","full"] = "none"
):
    """ซีรีส์ตามทุกฉากภาพ (จุดตามวันที่ภาพจริง)"""
    if station not in AOIS: raise HTTPException(404, "Unknown station")
    geom = AOIS[station]; ini, end = get_window(year)
    mask = build_water_mask(geom)
    cols = build_collections(geom, ini, end, cloud_perc, mask, ac)

    base_for_chl_sd = cols["toa_rrs"] if ac=="full" else cols["sr_scaled"]
    base_for_ph_tur_do = cols["sr_rrs"] if ac=="full" else cols["sr_scaled"]
    base_for_sal = cols["sr_scaled"]

    ph_ic  = base_for_ph_tur_do.map(img_pH)
    tur_ic = base_for_ph_tur_do.map(img_turb)
    do_ic  = base_for_ph_tur_do.map(img_do)
    chl_ic = base_for_chl_sd.map(img_chl)
    zsd_ic = base_for_chl_sd.map(img_zsd)
    tsi_ic = chl_ic.map(img_tsi_from_chl)
    tsi_cls_ic = tsi_ic.map(tsi_reclass)
    sal_ic = base_for_sal.map(img_sal)

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

@app.get("/summary_year", response_model=SummaryYearResponse)
def summary_year(
    station: Literal["CP01","LS01","LS03","TP01","TP04","TP11","PN01","SK01","SK06"],
    year: int = Query(..., ge=2017, le=2025),
    cloud_perc: int = 30,
    ac: Literal["none","full"] = "none"
):
    if station not in AOIS: raise HTTPException(404, "Unknown station")
    geom = AOIS[station]; ini, end = get_window(year)
    mask = build_water_mask(geom)
    cols = build_collections(geom, ini, end, cloud_perc, mask, ac)

    base_for_chl_sd = cols["toa_rrs"] if ac=="full" else cols["sr_scaled"]
    base_for_ph_tur_do = cols["sr_rrs"] if ac=="full" else cols["sr_scaled"]
    base_for_sal = cols["sr_scaled"]

    ph  = base_for_ph_tur_do.map(img_pH).mean().reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get("pH")
    tur = base_for_ph_tur_do.map(img_turb).mean().reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get("turbidity")
    dox = base_for_ph_tur_do.map(img_do).mean().reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get("do_mgL")
    chl = base_for_chl_sd.map(img_chl).mean().reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get("chl_a")
    zsd = base_for_chl_sd.map(img_zsd).mean().reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get("secchi_m")
    tsi = base_for_chl_sd.map(img_chl).map(img_tsi_from_chl).mean().reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get("tsi")
    sal = base_for_sal.map(img_sal).mean().reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=MAXPX).get("salinity_idx")

    def safe(x):
        try: return float(ee.Number(x).getInfo())
        except: return None

    return {
        "station": station, "year": year, "cloud_perc": cloud_perc, "ac": ac,
        "mean": {"pH": safe(ph), "turbidity": safe(tur), "salinity_idx": safe(sal),
                 "do_mgL": safe(dox), "chl_a": safe(chl), "secchi_m": safe(zsd), "tsi": safe(tsi)}
    }

@app.get("/map_png")
def map_png(
    station: Literal["CP01","LS01","LS03","TP01","TP04","TP11","PN01","SK01","SK06"],
    year: int = Query(..., ge=2017, le=2025),
    layer: Literal["chl_a", "secchi", "tsi"] = "chl_a",
    cloud_perc: int = 30,
    ac: Literal["none","full"] = "none",
    width: int = 1024, height: int = 1024
):
    """PNG สำหรับส่งเข้า LINE (เฉลี่ยทั้งปีของเลเยอร์ที่เลือก)"""
    if station not in AOIS: raise HTTPException(404, "Unknown station")
    geom = AOIS[station]; ini, end = get_window(year)
    mask = build_water_mask(geom)
    cols = build_collections(geom, ini, end, cloud_perc, mask, ac)
    base_for_chl_sd = cols["toa_rrs"] if ac=="full" else cols["sr_scaled"]

    if layer == "chl_a":
        img = base_for_chl_sd.map(img_chl).mean().clip(geom)
        vis = {"min":0,"max":40,"palette":['darkblue','blue','cyan','limegreen','yellow','orange','orangered','darkred']}
    elif layer == "secchi":
        img = base_for_chl_sd.map(img_zsd).mean().clip(geom)
        vis = {"min":0,"max":2,"palette":['800000','FF9700','7BFF7B','0080FF','000080']}
    else:
        img = base_for_chl_sd.map(img_chl).map(img_tsi_from_chl).mean().clip(geom)
        vis = {"min":30,"max":80,"palette":['darkblue','blue','cyan','limegreen','yellow','orange','orangered','darkred']}

    thumb = img.visualize(**vis).getThumbURL({
        "dimensions": f"{width}x{height}",
        "region": geom,
        "format": "png"
    })
    return {"station": station, "year": year, "layer": layer, "ac": ac, "png": thumb}

@app.get("/tiles", response_model=TileResponse)
def tiles(
    station: Literal["CP01","LS01","LS03","TP01","TP04","TP11","PN01","SK01","SK06"],
    year: int = Query(..., ge=2017, le=2025),
    cloud_perc: int = 30,
    ac: Literal["none","full"] = "none"
):
    if station not in AOIS: raise HTTPException(404, "Unknown station")
    geom = AOIS[station]; ini, end = get_window(year)
    mask = build_water_mask(geom)
    cols = build_collections(geom, ini, end, cloud_perc, mask, ac)
    base_for_chl_sd = cols["toa_rrs"] if ac=="full" else cols["sr_scaled"]

    chl = base_for_chl_sd.map(img_chl).mean().clip(geom)
    zsd = base_for_chl_sd.map(img_zsd).mean().clip(geom)
    tsi = base_for_chl_sd.map(img_chl).map(img_tsi_from_chl).mean().clip(geom)

    chl_map = ee.data.getMapId({"image": chl.visualize(min=0, max=40, palette=['darkblue','blue','cyan','limegreen','yellow','orange','orangered','darkred'])})
    zsd_map = ee.data.getMapId({"image": zsd.visualize(min=0, max=2,  palette=['800000','FF9700','7BFF7B','0080FF','000080'])})
    tsi_map = ee.data.getMapId({"image": tsi.visualize(min=30,max=80, palette=['darkblue','blue','cyan','limegreen','yellow','orange','orangered','darkred'])})
    return {
        "name": f"{station}-{year}",
        "tiles": {
            "chl_a_mean": chl_map["tile_fetcher"].url_format,
            "secchi_mean": zsd_map["tile_fetcher"].url_format,
            "tsi_mean": tsi_map["tile_fetcher"].url_format
        }
    }

@app.get("/")
def root():
    return {"name":"Aqua Sight API",
            "examples":{
                "monthly":"/timeseries_monthly?station=CP01&year=2024&ac=full",
                "scenes":"/timeseries_scenes?station=CP01&year=2024&ac=full",
                "summary":"/summary_year?station=CP01&year=2024&ac=full",
                "png":"/map_png?station=CP01&year=2024&layer=chl_a&ac=full",
                "tiles":"/tiles?station=CP01&year=2024&ac=full"}}

@app.get("/health-ee")
def health_ee():
    try:
        return {"ok": True, "roots": ee.data.getAssetRoots()}
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "has_EE_SERVICE_ACCOUNT": bool(os.getenv("EE_SERVICE_ACCOUNT")),
            "has_EE_KEY_B64": bool(os.getenv("EE_KEY_B64")),
            "has_GOOGLE_APPLICATION_CREDENTIALS_JSON": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")),
        }

# --- Proxy for EE thumbnails (:getPixels) ---
EE_THUMB_PATTERN = re.compile(
    r"^https://earthengine(?:-highvolume)?\.googleapis\.com/.+:getPixels(?:\?.*)?$"
)

@app.get("/proxy_png")
def proxy_png(url: str):
    """
    รับ URL แบบ Earth Engine thumbnails `...:getPixels`
    → POST ไปหา EE แล้วส่ง bytes กลับเป็น image/png พร้อม Content-Length
    (เหมาะกับ LINE ที่ต้องการ HTTPS public GET)
    """
    if not EE_THUMB_PATTERN.match(url):
        raise HTTPException(status_code=400, detail="Invalid EE thumbnail URL")

    try:
        r = requests.post(
            url,
            timeout=60,
            headers={"User-Agent": "AquaSight-Proxy/1.0"},
        )
        r.raise_for_status()

        data = r.content
        return Response(
            content=data,
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=300",
                "Content-Length": str(len(data)),
                "X-Content-Type-Options": "nosniff",
            },
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Proxy fetch failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
