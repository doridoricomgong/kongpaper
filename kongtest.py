import cv2
import numpy as np
import matplotlib.pyplot as plt
## Functions ##
def bean_check(img, overlab_rate, dish_x, dish_y, dish_r):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, max_bean_R * overlab_rate, param1=80, param2=9, minRadius=min_bean_R, maxRadius=max_bean_R)
    ans = 0
    for i in circles[0]:
        a = int(i[0])
        b = int(i[1])
        if (((a - dish_x) ** 2 + (b - dish_y) ** 2) < (dish_r * 0.97) ** 2):
            ans += 1
    return ans

ans_org_list = []
hidden_array = [10,1070,59,37,57,140,982,6,870,89,125,1541,873,69,52,116,1600,4,143,1480,860,1594,24,923,14,154,449,1178,150,39]

for index in range(5, 7):
    ## Image Read ##
    zfill_index = format(index, '02')
    #img = cv2.imread(f"/content/drive/MyDrive/CountingBean/Hidden/t{zfill_index}/5.jpg")
    img = cv2.imread(f"./Open/t{zfill_index}/5.jpg")

    ## Image Resizing ##
    img = cv2.resize(img, dsize=(800, 600), interpolation=cv2.INTER_AREA)

    ## Dish Mask ##
    # dish color range
    lower_white = np.array([0, 185, 185])
    upper_white = np.array([255, 255, 255])

    # masking
    mask = cv2.inRange(img, lower_white, upper_white)
    mask_img = cv2.bitwise_and(img, img, mask=mask)
    mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    # dish size range
    min_dish_R = int(img.shape[1] * 0.32)
    max_dish_R = int(img.shape[1] * 0.42)

    ## Dish Detection ##
    circles = cv2.HoughCircles(mask_img_gray, cv2.HOUGH_GRADIENT, 1, 100, param1 = 220, param2 = 35, minRadius = min_dish_R, maxRadius = max_dish_R)

    dish_x = 0
    dish_y = 0
    dish_r = 0
    diff_from_center = float("inf")
    last_diff_from_center = float("inf")

    # find best dish
    for i in circles[0]:
        diff_from_center = (i[0] - img.shape[1] * 0.5) ** 2 + (i[1] - img.shape[0] * 0.5) ** 2
        if diff_from_center < last_diff_from_center:
            cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
            best_circle = i
            last_diff_from_center = diff_from_center

    # Save Dish Coordinates
    dish_x = best_circle[0]
    dish_y = best_circle[1]
    dish_r = best_circle[2]

    ## Bean Mask ##
    img_been = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # bean color range
    lower = np.array([8, 45, 0])
    upper = np.array([25, 255, 255])

    # masking
    mask = cv2.inRange(img_been, lower, upper)
    img_been = cv2.bitwise_and(img_been, img_been, mask=mask)
    img_been_gray = cv2.cvtColor(img_been, cv2.COLOR_BGR2GRAY)

    # bean size range
    min_bean_R = int(img.shape[1] * 0.01)
    max_bean_R = int(img.shape[1] * 0.017)

    ## Bean Detection ##
    circles = cv2.HoughCircles(img_been_gray, cv2.HOUGH_GRADIENT, 1, max_bean_R * 0.8, param1=80, param2=9, minRadius=min_bean_R, maxRadius=max_bean_R)
    ans = 0
    for i in circles[0]:
        bean_size_diff = max_bean_R - min_bean_R
        bean_size_color = (int(i[2]) - min_bean_R) / (bean_size_diff / 255)
        if (((int(i[0]) - dish_x) ** 2 + (int(i[1]) - dish_y) ** 2) < (dish_r * 0.97) ** 2):
            ans += 1
    ans_org = ans
    ans_org_list.append(ans_org)

    cv2.imshow('test', img_been_gray)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

