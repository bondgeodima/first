import requests
import os
import io
from PIL import Image
import skimage.io

from maptiler import GlobalMercator

lat1 = 52.200643
long1 = 20.974275

lat2 = 52.203171
long2 = 20.980568

# Работать надо с 19 зумом гугла
# z = 19

z = 19

g1 = GlobalMercator()

x1, y1 = g1.LatLonToMeters(lat1, long1)
t1 = g1.MetersToTile(x1, y1, z)
t1 = g1.GoogleTile(t1[0], t1[1], z)

x2, y2 = g1.LatLonToMeters(lat1, long2)
t2 = g1.MetersToTile(x2, y2, z)
t2 = g1.GoogleTile(t2[0], t2[1], z)

x3, y3 = g1.LatLonToMeters(lat2, long2)
t3 = g1.MetersToTile(x3, y3, z)
t3 = g1.GoogleTile(t3[0], t3[1], z)

x4, y4 = g1.LatLonToMeters(lat2, long1)
t4 = g1.MetersToTile(x4, y4, z)
t4 = g1.GoogleTile(t4[0], t4[1], z)

tx = [t1[0], t2[0], t3[0], t4[0]]
ty = [t1[1], t2[1], t3[1], t4[1]]

# print (t1)
# print (t2)
# print (t3)
# print (t4)

# print ("")

# dx = [t1[0], t2[0]]
# dy = [t3[1], t2[1]]

# print (dx)
# print (dy)

# url = 'https://khms0.google.com/kh/v=894?x=242601&y=393871&z=20'
IMAGE_DIR_OUT = 'F:/tmp/'
filename_out = 'tmp.jpg'
your_path = os.path.join(IMAGE_DIR_OUT, filename_out)

count = (max(tx) - min(tx))*(max(ty)-min(ty))

print("count tile = " + str(count))

t = 1
for i in range(min(tx), max(tx)):
    for j in range(min(ty), max(ty)):

        print (str(i) + " " + str(j) + " " + str(t) + " from " + str(count))
        t = t+1
        tms_tile = g1.GoogleTile(i, j, z)
        b = g1.TileBounds(tms_tile[0], tms_tile[1], z)

        dx = b[2] - b[0]

        x_coord = b[0]
        y_coord = b[3]

        url = 'https://khms0.google.com/kh/v=894?x='+str(i)+'&y='+str(j)+'&z='+str(z)+''

        r = requests.get(url, stream=True)
        im = Image.open(io.BytesIO(r.content))
        nx, ny = im.size

        scale = round(dx / nx, 10)

        im.save(IMAGE_DIR_OUT + str(i) + "_" + str(j) + '.jpg', dpi=(70, 70))

        f = open(IMAGE_DIR_OUT + str(i) + "_" + str(j) + '.jgw', "w+")
        f.write(str(scale) + '\r\n')
        f.write(str("0.0000000000") + '\r\n')
        f.write(str("0.0000000000") + '\r\n')
        f.write("-" + str(scale) + '\r\n')
        f.write(str(x_coord) + '\r\n')
        f.write(str(y_coord) + '\r\n')
        f.close()

        f = open(IMAGE_DIR_OUT + str(i) + "_" + str(j) + '.prj', "w+")
        f.write('PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
                'SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
                'UNIT["Degree",0.0174532925199433]],PROJECTION["Mercator_Auxiliary_Sphere"],'
                'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
                'PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",0.0],'
                'PARAMETER["Auxiliary_Sphere_Type",0.0],UNIT["Meter",1.0]]')
        f.close()

        for n in range(0, 256, 128):
            x_coord_new = x_coord + n * scale
            for m in range(0, 256, 128):
                im_cropped = im.crop((n, m, n+128, m+128))
                y_coord_new = y_coord - m * scale

                nx, ny = im_cropped.size

                im2 = im_cropped.resize((int(nx * 8), int(ny * 8)), Image.BICUBIC)
                scale_new = scale/8

                im2.save(IMAGE_DIR_OUT + str(i) + "_" + str(j) + "_" + str(n) + "_" + str(m) + '.jpg', dpi=(576, 576))

                # im3 = Image.open(new_patch)

                f = open(IMAGE_DIR_OUT + str(i) + "_" + str(j) + "_" + str(n) + "_" + str(m) + '.jgw', "w+")
                f.write(str(scale_new) + '\r\n')
                f.write(str("0.0000000000") + '\r\n')
                f.write(str("0.0000000000") + '\r\n')
                f.write("-" + str(scale_new) + '\r\n')
                f.write(str(x_coord_new) + '\r\n')
                f.write(str(y_coord_new) + '\r\n')
                f.close()

                f = open(IMAGE_DIR_OUT + str(i) + "_" + str(j) + "_" + str(n) + "_" + str(m) + '.prj', "w+")
                f.write('PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
                        'SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],'
                        'UNIT["Degree",0.0174532925199433]],PROJECTION["Mercator_Auxiliary_Sphere"],'
                        'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
                        'PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",0.0],'
                        'PARAMETER["Auxiliary_Sphere_Type",0.0],UNIT["Meter",1.0]]')
                f.close()

print("END CALCULATION")


