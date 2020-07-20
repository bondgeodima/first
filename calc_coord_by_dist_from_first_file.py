"""
Файл строит угол смотрения камеры
Т.е. берутся координаты х, у и угол с срт файла и строится предпосмотр
куда смотрит камера согласно срт файла
"""

import io
import psycopg2
import json

data = {
    "type": "FeatureCollection",
    "name": "test",
    "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
    "features": []
}

feature = []

conn = psycopg2.connect(dbname='dzk', user='postgres',
                         password='gagra321', host='192.168.33.57')
cursor = conn.cursor()

ff = io.open(r'D:\deep_learning\video_4\VID_20200609_112258_out_repair_angle.txt', mode="r", encoding="utf-8")
fff = io.open(r'D:\deep_learning\video_4\VID_20200609_112258_out_repair_angle_view.txt', mode="w+", encoding="utf-8")
ffff = io.open(r'D:\deep_learning\video_4\VID_20200609_112258_out_repair_angle_view.geojson', mode="w+", encoding="utf-8")

ii = 1
for st in ff:

    lang = st.split(';')[2].strip()
    lat = st.split(';')[1].strip()
    angle = st.split(';')[3].strip()
    dist = 5
    num = st.split(';')[4].strip()
    i = 1

    print (i)

    if i == 1 and angle != 'nan' and angle != 'None':
        point = []
        properties = {}
        sql_statement = "SELECT ST_asText(ST_transform(ST_transform(ST_SetSRID(st_project(st_setsrid(" \
                        "st_makepoint({}, {}),4326), " \
            "{}, radians({}))::geometry,4326),900913),4326))".format(lang, lat, dist, angle)

        cursor.execute(sql_statement)
        records = cursor.fetchall()
        point = records[0][0].split()

        features = {
            "type": "Feature",
            "properties": {},
            "geometry": {}
        }

        geometry = {
            "type": "MultiLineString",
            "coordinates": []
        }

        ish = []
        e = []

        if records:
            print (point[0].replace("POINT(",""), point[1][:-1])

            ish = [float(lang), float(lat)]
            e = [float(point[0].replace("POINT(","")), float(point[1][:-1])]
            properties = {
                "id": ii
            }
            geometry["coordinates"].append([ish, e])

            features["properties"] = properties
            features["geometry"] = geometry
            feature.append(features)
            ii += 1

            ll = str(e[0]) + ';' + str(e[1]) + ';' + str(num) + '\n'
            fff.write(ll)

        # print (features)

data["features"] = feature

# print(data)
json.dump(data, ffff)
