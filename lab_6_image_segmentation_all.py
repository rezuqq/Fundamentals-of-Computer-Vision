# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:38:53 2023

@author: AM4
"""

import sys
sys.path.append('../')
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import data
from scipy import ndimage
import matplotlib.pyplot as plt
import segmentation_utils
from sklearn.cluster import MiniBatchKMeans


image = cv.imread('./ape.bmp')
image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

## Основанные на регионах (Region-based)
# Разрастание областей (region growing)
# определяем координаты начальных точек
seeds = [(100, 100), (50, 125), (50, 225), (210, 119),(170, 190),(230, 210)]
# координаты для графика
x = list(map(lambda x: x[1], seeds))
y = list(map(lambda x: x[0], seeds))
# порог похожести цвета региона
threshold = 100
# находим сегментацию используя метод из segmentation_utils
segmented_region = segmentation_utils.region_growingHSV(image_hsv, seeds, threshold)
# накладываем маску - отображаем только участки попавшие в какой-либо сегмент
result = cv.bitwise_and(image, image, mask=segmented_region)
# отображаем полученное изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.scatter(x, y, marker="x", color="red", s=200)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()


# Разделение областей
qt = segmentation_utils.QTree(stdThreshold = 0.25, minPixelSize = 4,img = image.copy()) 
qt.subdivide()
tree_image = qt.render_img(thickness=1, color=(0,0,0))

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(tree_image, cv.COLOR_BGR2RGB))
plt.show()


# Алгоритм водораздела
# Бинаризируем изображение
binary_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
# Определяем карту расстояний
distance_map = ndimage.distance_transform_edt(binary_image)
# Определяем локальные максимумы
local_max = peak_local_max(
    distance_map,
    min_distance=20,
    labels=binary_image
)

local_max_mask = np.zeros(distance_map.shape, dtype=bool)
local_max_mask[tuple(local_max.T)] = True

# 4 Каждому минимуму присваивается метка и начинается заполнение бассейнов метками
markers = ndimage.label(
    local_max_mask,
    structure=np.ones((3, 3))
)[0]

labels = watershed(
    -distance_map,
    markers,
    mask=binary_image > 0
)
# построим результаты работы алгоритма
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(binary_image, cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(np.uint8(distance_map + 50), cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(np.uint8(labels))
plt.show()

# Найдем границы контуров и положим в маску все кроме метки 0
mask1 = np.zeros(image.shape[0:2], dtype="uint8")
total_area = 0
for label in np.unique(labels):
    if label < 2:
        continue
    # Create a mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    mask1 = mask1 + mask

    # Find contours and determine contour area
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv.contourArea)
    area = cv.contourArea(c)
    total_area += area
    cv.drawContours(image, [c], -1, (36,255,12), 1)

result = cv.bitwise_and(image, image, mask=mask1)

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(mask1, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()

## Методы кластеризации. K-средних
# Преобразуем изображение в оттенках серого в одномерный массив
pixels = gray.reshape(-1, 1)
# Задаем число кластеров для сегментации
K = 3
# С помощью библиотеки sklearn.cluster import KMeans проводим кластеризацию по яркости
kmeans = KMeans(n_clusters=K, random_state=0)
labels = kmeans.fit_predict(pixels)
cluster_centers = kmeans.cluster_centers_
print (np.uint8(cluster_centers))
# Каждому пикселю назначаем значение из центра кластера
segments = np.uint8(cluster_centers[labels].reshape(gray.shape))
# Удалим самые яркие пиксели
segments[segments==167] = 0
result = cv.bitwise_and(gray, gray, mask=segments)
# Отобразим избражения 
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.imshow(segments, cmap='Set3')
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.show()


## Методы кластеризации. Сдвиг среднего (Mean shift)
# Сглаживаем чтобы уменьшить шум
blur_image = cv.medianBlur(image, 3)
# Выстраиваем пиксели в один ряд и переводим в формат с правающей точкой
flat_image = np.float32(blur_image.reshape((-1,3)))

# Используем meanshift из библиотеки sklearn
bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_

# получим количество сегментов
segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])

# получим средний цвет сегмента
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)

# Для каждого пискеля проставим средний цвет его сегмента
mean_shift_image = avg[labeled].reshape((image.shape))
# Маской скроем один из сегментов
mask1 = mean_shift_image[:,:,0]
mask1[mask1==89] = 0
mean_shift_with_mask_image = cv.bitwise_and(image, image, mask=mask1)
# Построим изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.imshow(mean_shift_image, cmap='Set3')
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(mean_shift_with_mask_image, cv.COLOR_BGR2RGB))
plt.show()


### ЗАДАНИЕ

# -*- coding: utf-8 -*-

image_path = (
    "./1657737386_27-mykaleidoscope-ru-p-konfeti-sakura-"
    "rot-front-vkusnyashki-krasi-40.jpg"
)

image = cv.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Не удалось загрузить изображение: {image_path}"
    )

# Оригинал сохраняем, потому что дальше будем рисовать контуры
original = image.copy()

height, width = image.shape[:2]

image_hsv = cv.cvtColor(original, cv.COLOR_BGR2HSV)
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

print("Размер изображения:", width, "x", height)

plt.figure(figsize=(10, 7))
plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.axis("off")
plt.tight_layout()
plt.show()


# 1. ОСНОВАННЫЕ НА РЕГИОНАХ (REGION-BASED)


# Координаты записаны в формате (y, x)
seeds = [
    (100, 450),
    (100, 650),
    (300, 550),
    (700, 500),
    (875, 475),
    (800, 375),
    (900, 300),
    (1000, 450)
]

# Проверяем, что seed находятся внутри изображения
valid_seeds = [
    (y, x) for y, x in seeds
    if 0 <= y < height and 0 <= x < width
]

invalid_seeds = len(seeds) - len(valid_seeds)

if invalid_seeds:
    print("Пропущено seed вне изображения:", invalid_seeds)

if not valid_seeds:
    raise ValueError(
        "Все seed вне изображения. Проверь координаты."
    )

# ВАЖНО:
# Используем обновлённую функцию region_growingHSV
# из предыдущего сообщения, которая принимает отдельные пороги.
# Если модуль ещё содержит старую функцию, замени её в
# segmentation_utils.py перед запуском этого блока.

region_mask = segmentation_utils.region_growingHSV(
    image_hsv,
    valid_seeds,
    h_thresh=12,
    s_thresh=60,
    v_thresh=60
)

region_result = cv.bitwise_and(
    original, original, mask=region_mask
)

res1 = region_result.copy()

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
plt.scatter(
    [x for y, x in valid_seeds],
    [y for y, x in valid_seeds],
    marker="x", color="red", s=100
)
plt.title("Исходное изображение и seed")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(region_mask, cmap="gray")
plt.title("Маска Region Growing")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(region_result, cv.COLOR_BGR2RGB))
plt.title("Результат Region Growing")
plt.axis("off")

plt.tight_layout()
plt.show()



# 2. РАЗДЕЛЕНИЕ ОБЛАСТЕЙ


qt = segmentation_utils.QTree(
    stdThreshold=0.25,
    minPixelSize=4,
    img=original.copy()
)

qt.subdivide()

tree_image = qt.render_img(
    thickness=1,
    color=(0, 0, 0)
)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(tree_image, cv.COLOR_BGR2RGB))
plt.title("Разделение областей QTree")
plt.axis("off")

plt.tight_layout()
plt.show()


# 3. АЛГОРИТМ ВОДОРАЗДЕЛА (WATERSHED)

# 3.1. Сглаживаем изображение и получаем бинарную маску
blur_gray = cv.GaussianBlur(gray, (5, 5), 0)

binary_image = cv.threshold(
    blur_gray,
    0,
    255,
    cv.THRESH_BINARY + cv.THRESH_OTSU
)[1]

# Выбираем полярность, при которой белая область меньше.
# Это эвристика, а не универсальное правило:
# для светлых объектов на тёмном фоне её нужно проверить.
if np.mean(binary_image == 255) > 0.5:
    binary_image = cv.bitwise_not(binary_image)

# Удаляем небольшие шумовые компоненты
kernel = np.ones((3, 3), np.uint8)

binary_image = cv.morphologyEx(
    binary_image,
    cv.MORPH_OPEN,
    kernel,
    iterations=1
)

foreground = binary_image > 0

# 3.2. Карта расстояний
distance_map = ndimage.distance_transform_edt(foreground)

# 3.3. Локальные максимумы
local_max = peak_local_max(
    distance_map,
    min_distance=12,
    labels=foreground,
    exclude_border=False
)

local_max_mask = np.zeros(
    distance_map.shape,
    dtype=bool
)

if len(local_max) > 0:
    local_max_mask[tuple(local_max.T)] = True

markers, _ = ndimage.label(
    local_max_mask,
    structure=np.ones((3, 3), dtype=np.uint8)
)

# Если максимумы не найдены, используем связные компоненты
if markers.max() == 0:
    markers, _ = ndimage.label(foreground)

# 3.4. Watershed
labels = watershed(
    -distance_map,
    markers,
    mask=foreground
)

# 3.5. Контуры и фильтрация маленьких сегментов
min_area = 30

watershed_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)

contour_image = original.copy()
total_area = 0
segment_count = 0

for label_id in np.unique(labels):
    if label_id == 0:
        continue

    segment = (labels == label_id).astype(np.uint8) * 255

    contours, _ = cv.findContours(
        segment,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv.contourArea(contour)

        if area < min_area:
            continue

        total_area += area
        segment_count += 1

        # Сохраняем только принятые сегменты
        cv.drawContours(
            watershed_mask,
            [contour],
            -1,
            255,
            thickness=cv.FILLED
        )

        cv.drawContours(
            contour_image,
            [contour],
            -1,
            (0, 255, 0),
            1
        )

watershed_result = cv.bitwise_and(
    original,
    original,
    mask=watershed_mask
)

res2 = watershed_result.copy()

print("Количество сегментов Watershed:", segment_count)
print("Суммарная площадь контуров:", total_area, "пикселей")

plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.imshow(binary_image, cmap="gray")
plt.title("Бинарная маска")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(distance_map, cmap="gray")
plt.title("Карта расстояний")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(labels, cmap="nipy_spectral")
plt.title("Метки Watershed")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(watershed_mask, cmap="gray")
plt.title("Маска после фильтрации")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(cv.cvtColor(contour_image, cv.COLOR_BGR2RGB))
plt.title("Контуры сегментов")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(cv.cvtColor(watershed_result, cv.COLOR_BGR2RGB))
plt.title("Результат Watershed")
plt.axis("off")

plt.tight_layout()
plt.show()


# 4. K-MEANS
# Используем цветные пиксели, а не только оттенки серого.
# Lab часто удобнее для цветовой кластеризации.
lab = cv.cvtColor(original, cv.COLOR_BGR2LAB)

pixels = lab.reshape(-1, 3).astype(np.float32)

K = 5

kmeans = KMeans(
    n_clusters=K,
    random_state=0,
    n_init=10
)

labels_kmeans = kmeans.fit_predict(pixels)

# Восстанавливаем цвет кластерного центра для каждого пикселя
clustered_lab = kmeans.cluster_centers_[
    labels_kmeans
].reshape(height, width, 3)

clustered_lab = np.clip(
    clustered_lab, 0, 255
).astype(np.uint8)

kmeans_image = cv.cvtColor(
    clustered_lab,
    cv.COLOR_LAB2RGB
)



# Визуализируем номера кластеров отдельно
label_map = labels_kmeans.reshape(height, width)

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(label_map, cmap="tab10", vmin=0, vmax=K - 1)
plt.title(f"K-Means: {K} кластеров")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(kmeans_image)
plt.title("Цвета кластерных центров")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 5. СРАВНЕНИЕ МЕТОДОВ СЕГМЕНТАЦИИ
# ============================================================

print("\n========== СРАВНЕНИЕ МЕТОДОВ ==========")

# Получаем бинарные маски каждого метода.
# Белые пиксели означают область, выделенную методом.

# 1. Region Growing
mask_region = region_mask > 0

# 2. QTree — это разбиение на области, а не готовая маска.
# Для честного сравнения сегментации объектов этот метод
# нужно отдельно преобразовать в маску целевого объекта.

# 3. Watershed
mask_watershed = watershed_mask > 0

# 4. K-Means
# Для сравнения необходимо указать, какой кластер является объектом.
# Пока оцениваем долю каждого кластера.
labels_2d = labels_kmeans.reshape(height, width)

print("\n1. Region Growing")
print("Выделено пикселей:", int(mask_region.sum()))
print("Доля изображения:", round(mask_region.mean() * 100, 2), "%")

print("\n2. Watershed")
print("Выделено пикселей:", int(mask_watershed.sum()))
print("Доля изображения:", round(mask_watershed.mean() * 100, 2), "%")

print("\n3. K-Means")
for cluster_id in range(K):
    cluster_mask = labels_2d == cluster_id
    print(
        f"Кластер {cluster_id}: "
        f"{int(cluster_mask.sum())} пикселей "
        f"({cluster_mask.mean() * 100:.2f}%)"
    )

print("\n4. QTree")
print("Метод построил иерархическое разбиение изображения.")
print("Для количественного сравнения нужна маска объекта.")

# Визуальное сравнение результатов
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(mask_region, cmap="gray")
plt.title("Region Growing")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(mask_watershed, cmap="gray")
plt.title("Watershed")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(labels_2d, cmap="tab10")
plt.title("K-Means: метки кластеров")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv.cvtColor(tree_image, cv.COLOR_BGR2RGB))
plt.title("QTree")
plt.axis("off")

plt.tight_layout()
plt.show()