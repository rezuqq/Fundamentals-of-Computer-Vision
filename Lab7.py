# -*- coding: utf-8 -*-
"""
Поиск шаблона на изображении:
1. MatchTemplate
2. Сопоставление границ Собеля
3. SIFT + BFMatcher + RANSAC

Файлы picture.png и mask.png должны находиться
в рабочей папке программы.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt



# Загрузка изображений

rgb_img = cv2.imread("picture.jpg")
template = cv2.imread("mask.jpg")

if rgb_img is None:
    raise FileNotFoundError("Не удалось загрузить picture.jpg")

if template is None:
    raise FileNotFoundError("Не удалось загрузить mask.jpg")

gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
gray_templ = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# OpenCV: shape = (высота, ширина)
h, w = gray_templ.shape

if h > gray_img.shape[0] or w > gray_img.shape[1]:
    raise ValueError(
        "Шаблон должен быть не больше исходного изображения"
    )

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
plt.title("Шаблон")
plt.axis("off")



# 1. Поиск шаблона с помощью matchTemplate


res = cv2.matchTemplate(
    gray_img,
    gray_templ,
    cv2.TM_CCOEFF_NORMED
)

plt.figure(figsize=(8, 5))
plt.imshow(res, cmap="jet")
plt.colorbar()
plt.title("Карта сходства шаблона")

threshold = 0.8

# Позиции, в которых сходство выше порога
ys, xs = np.where(res >= threshold)

# Отрисовываем прямоугольники
result1 = rgb_img.copy()

for x, y in zip(xs, ys):
    cv2.rectangle(
        result1,
        (int(x), int(y)),
        (int(x + w), int(y + h)),
        (0, 255, 255),
        2
    )

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
plt.title(f"MatchTemplate, порог = {threshold}")
plt.axis("off")



# 2. Поиск по границам Собеля


# Выделяем границы исходного изображения
sx = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=3)
sy = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1, ksize=3)

abs_x = cv2.convertScaleAbs(sx)
abs_y = cv2.convertScaleAbs(sy)

edge_img = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

# Выделяем границы шаблона
# convertScaleAbs корректно обрабатывает изменение яркости
bright_templ = cv2.convertScaleAbs(gray_templ, alpha=2)

tx = cv2.Sobel(bright_templ, cv2.CV_16S, 1, 0, ksize=3)
ty = cv2.Sobel(bright_templ, cv2.CV_16S, 0, 1, ksize=3)

abs_tx = cv2.convertScaleAbs(tx)
abs_ty = cv2.convertScaleAbs(ty)

edge_templ = cv2.addWeighted(abs_tx, 0.5, abs_ty, 0.5, 0)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(edge_img, cmap="gray")
plt.title("Границы изображения")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edge_templ, cmap="gray")
plt.title("Границы шаблона")
plt.axis("off")

# Поиск по границам
res_edges = cv2.matchTemplate(
    edge_img,
    edge_templ,
    cv2.TM_CCOEFF_NORMED
)

plt.figure(figsize=(8, 5))
plt.imshow(res_edges, cmap="jet")
plt.colorbar()
plt.title("Сходство границ")

threshold_edges = 0.8
ys, xs = np.where(res_edges >= threshold_edges)

result2 = rgb_img.copy()

for x, y in zip(xs, ys):
    cv2.rectangle(
        result2,
        (int(x), int(y)),
        (int(x + w), int(y + h)),
        (0, 255, 255),
        2
    )

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
plt.title("Поиск по границам Собеля")
plt.axis("off")



# 3. Поиск с помощью SIFT и гомографии RANSAC


# Масштабирование шаблона для примера
scale = 1.1

new_w = max(1, int(w * scale))
new_h = max(1, int(h * scale))

scaled_template = cv2.resize(
    gray_templ,
    (new_w, new_h),
    interpolation=cv2.INTER_LINEAR
)

# Изменение яркости шаблона необязательно.
# При необходимости можно включить:
# scaled_template = cv2.convertScaleAbs(
#     scaled_template, alpha=0.9
# )

# Создаём детектор SIFT
if not hasattr(cv2, "SIFT_create"):
    raise RuntimeError(
        "SIFT недоступен в этой сборке OpenCV. "
        "Установите актуальную версию opencv-contrib-python."
    )

sift = cv2.SIFT_create()

# Особые точки и дескрипторы
kp_img, des_img = sift.detectAndCompute(gray_img, None)
kp_templ, des_templ = sift.detectAndCompute(
    scaled_template, None
)

print("Точек на изображении:", len(kp_img))
print("Точек в шаблоне:", len(kp_templ))

# Отображаем особые точки
keypoints_img = cv2.drawKeypoints(
    rgb_img, kp_img, None,
    color=(0, 255, 255)
)

keypoints_templ = cv2.drawKeypoints(
    cv2.cvtColor(scaled_template, cv2.COLOR_GRAY2BGR),
    kp_templ,
    None,
    color=(0, 255, 255)
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(keypoints_img, cv2.COLOR_BGR2RGB))
plt.title("Особые точки изображения")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(keypoints_templ, cv2.COLOR_BGR2RGB))
plt.title("Особые точки шаблона")
plt.axis("off")

result3 = rgb_img.copy()

# Проверяем наличие дескрипторов
if (
    des_img is None
    or des_templ is None
    or len(kp_img) < 2
    or len(kp_templ) < 2
):
    print("Недостаточно особых точек для поиска.")

else:
    # SIFT использует евклидово расстояние L2
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Для каждой точки шаблона ищем две ближайшие
    пары = bf.knnMatch(des_templ, des_img, k=2)

    # Тест отношения Лоу
    good = []

    for pair in пары:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.7 * n.distance:
            good.append(m)

    print("Хороших совпадений:", len(good))

    # Для вычисления гомографии нужны как минимум 4 пары
    if len(good) >= 4:
        src_pts = np.float32([
            kp_templ[m.queryIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp_img[m.trainIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        # Определяем преобразование шаблона
        # в координаты исходного изображения
        H, inlier_mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

        if H is not None:
            corners = np.float32([
                [0, 0],
                [new_w - 1, 0],
                [new_w - 1, new_h - 1],
                [0, new_h - 1]
            ]).reshape(-1, 1, 2)

            projected = cv2.perspectiveTransform(
                corners, H
            )

            # Проверяем, что контур расположен внутри изображения
            if np.isfinite(projected).all():
                cv2.polylines(
                    result3,
                    [np.int32(projected)],
                    True,
                    (0, 255, 255),
                    3
                )

                inliers = (
                    int(inlier_mask.sum())
                    if inlier_mask is not None
                    else 0
                )

                print("Совпадений-инлаеров:", inliers)

                # Отображаем совпавшие особые точки
                matches_img = cv2.drawMatches(
                    scaled_template,
                    kp_templ,
                    rgb_img,
                    kp_img,
                    good,
                    None,
                    matchColor=(0, 255, 0),
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                plt.figure(figsize=(14, 7))
                plt.imshow(
                    cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB)
                )
                plt.title("Совпадения SIFT")
                plt.axis("off")

            else:
                print("Получены некорректные координаты контура.")

        else:
            print("Не удалось вычислить гомографию.")

    else:
        print(
            "Недостаточно совпадений для локализации шаблона. "
            "Попробуйте снизить порог теста Лоу или использовать "
            "изображения с большим количеством деталей."
        )

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(result3, cv2.COLOR_BGR2RGB))
plt.title("Результат поиска SIFT + RANSAC")
plt.axis("off")

plt.show()


# 4. Отображение совпадения со стрелкой

arrow_img = rgb_img.copy()

# Для примера используем результат первого метода — matchTemplate
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Центр найденной области
center_x = max_loc[0] + w // 2
center_y = max_loc[1] + h // 2

# Обводим область совпадения
cv2.rectangle(
    arrow_img,
    max_loc,
    (max_loc[0] + w, max_loc[1] + h),
    (0, 255, 255),
    3
)

# Начало стрелки — над найденной областью
start_point = (
    max(0, center_x - 100),
    max(0, center_y - 100)
)

# Стрелка указывает в центр совпадения
cv2.arrowedLine(
    arrow_img,
    start_point,
    (center_x, center_y),
    (0, 0, 255),
    3,
    tipLength=0.2
)

# Подписываем степень совпадения
cv2.putText(
    arrow_img,
    f"Match: {max_val:.2f}",
    (max_loc[0], max(25, max_loc[1] - 10)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    2
)

# Отдельный вывод
plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(arrow_img, cv2.COLOR_BGR2RGB))
plt.title("Область совпадения со стрелкой")
plt.axis("off")
plt.show()


