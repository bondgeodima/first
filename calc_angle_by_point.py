"""

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

ff = io.open(r'D:\deep_learning\video\VID_20200603_121541_out.txt', mode="r", encoding="utf-8")
# fff = io.open(r'D:\deep_learning\video_5\VID_20200610_193856_out_view.txt', mode="w+", encoding="utf-8")
# ffff = io.open(r'D:\deep_learning\video_5\VID_20200610_193856_out_view.geojson', mode="w+", encoding="utf-8")
ffff = io.open(r'D:\deep_learning\video\VID_20200603_121541_out_repair_angle.txt', mode="w+", encoding="utf-8")
# Не пойму как работает GIT

ii = 1

langs = []
lats = []
angles = []
dists = []
nums = []
names = []

for st in ff:

    lats.append(st.split(';')[1].strip())
    langs.append(st.split(';')[2].strip())
    angles.append(st.split(';')[3].strip())
    dists.append(20)
    names.append(st.split(';')[4].strip())
    nums.append(st.split(';')[5].strip())


for z in range(1, len(langs)):
    sql_statement = "SELECT round(degrees(ST_Azimuth(st_transform(st_setsrid(st_point({}, {}),4326),900913), " \
                    "st_transform(st_setsrid(st_point({}, {}),4326),900913)))::numeric,1)"\
        .format(langs[z-1], lats[z-1], langs[z], lats[z])
    print (sql_statement)
    cursor.execute(sql_statement)
    records = cursor.fetchall()
    a = records[0][0]
    angles[z-1] = a
    # print(a)
    ll = str(z) + ';' + str(lats[z-1]) + ';' + str(langs[z-1]) + ';' + \
         str(angles[z-1]) + ';' + str(names[z-1]) + ';' + str(nums[z-1]) + '\n'
    ffff.write(ll)

exit()
