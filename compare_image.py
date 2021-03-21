import cv2
import difflib
import os

ROOT_DIR = os.path.abspath("../")
IMAGE_DIR_1 = os.path.join(ROOT_DIR, r'F:\car_project\time_comparsion\2004\black')
IMAGE_DIR_2 = os.path.join(ROOT_DIR, r'F:\car_project\time_comparsion\2019\black')


# Функция вычисления хэша
def CalcImageHash(FileName):
    image = cv2.imread(FileName)  # Прочитаем картинку
    resized = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)  # Уменьшим картинку
    # resized = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)  # Уменьшим картинку
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # Переведем в черно-белый формат
    avg = gray_image.mean()  # Среднее значение пикселя
    ret, threshold_image = cv2.threshold(gray_image, avg, 255, 0)  # Бинаризация по порогу

    # Рассчитаем хэш
    _hash = ""
    for x in range(8):
        for y in range(8):
            val = threshold_image[x, y]
            if val == 255:
                _hash = _hash + "1"
            else:
                _hash = _hash + "0"

    return _hash


def CompareHash(hash1, hash2):
    l = len(hash1)
    i = 0
    count = 0
    while i < l:
        if hash1[i] != hash2[i]:
            count = count + 1
        i = i + 1
    return count


f = open("F:/car_project/time_comparsion/compare_black_8.txt", "a")
f.write('name : compare' + '\n')


for filename in os.listdir(IMAGE_DIR_1):
    if filename.split(".")[1] == 'tif':
        filename_1 = filename
        filename_2 = '2019' + filename[4:]
        hash1 = CalcImageHash(os.path.join(IMAGE_DIR_1, filename_1))
        hash2 = CalcImageHash(os.path.join(IMAGE_DIR_2, filename_2))
        print(filename_2.split(".")[0], CompareHash(hash1, hash2))
        f.write(str(filename_2.split(".")[0]) + ' : ' + str(CompareHash(hash1, hash2)) + '\n')

f.close()



"""
hash1 = CalcImageHash("F:/car_project/time_comparsion/2004_185.tif")
hash2 = CalcImageHash("F:/car_project/time_comparsion/2019_185.tif")
print(hash1)
print(hash2)
print(CompareHash(hash1, hash2))
"""

