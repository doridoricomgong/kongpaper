import cv2
import numpy as np

anslist = [44, 60, 49, 129, 39, 22, 513, 196, 263, 170, 98, 379, 1600, 2, 31, 5, 1190, 151, 108, 122, 75, 84, 10, 24, 1375, 7, 1032, 1429, 1323, 691]
score = 0


def bean_check(img, overlab_rate, dish_x, dish_y, dish_r):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, max_bean_R * overlab_rate, param1=80, param2=9, minRadius=min_bean_R, maxRadius=max_bean_R)
    ans = 0
    for i in circles[0]:
        a = int(i[0])
        b = int(i[1])
        if (((a - dish_x) ** 2 + (b - dish_y) ** 2) < (dish_r * 0.97) ** 2):
            ans += 1
    return ans

for index in range(1, 31):
    zfill_index = format(index, '02')
    img = cv2.imread(f"dis/Open/t{zfill_index}/5.jpg")
    img = cv2.resize(img, dsize=(800, 600), interpolation=cv2.INTER_AREA)

    #dish check
    min_dish_R = int(img.shape[1] * 0.32)
    max_dish_R = int(img.shape[1] * 0.42)

    lower_white = np.array([0, 185, 185])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower_white, upper_white)

    mask_img = cv2.bitwise_and(img, img, mask=mask)
    mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('test', mask_img_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    circles = cv2.HoughCircles(mask_img_gray, cv2.HOUGH_GRADIENT, 1, 100, param1 = 220, param2 = 35, minRadius = min_dish_R, maxRadius = max_dish_R)

    dish_x = 0
    dish_y = 0
    dish_r = 0

    diff_from_center = float("inf")
    last_diff_from_center = float("inf")
    for i in circles[0]:
        diff_from_center = (i[0] - img.shape[1] * 0.5) ** 2 + (i[1] - img.shape[0] * 0.5) ** 2

        if diff_from_center < last_diff_from_center:
            cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
            best_circle = i
            last_diff_from_center = diff_from_center

    cv2.circle(img, (int(best_circle[0]), int(best_circle[1])), int(best_circle[2]), (0, 255, 0), 2)
    dish_x = best_circle[0]
    dish_y = best_circle[1]
    dish_r = best_circle[2]


    #bean check
    img_been = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define blue color range
    lower = np.array([8, 45, 0])
    upper = np.array([25, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(img_been, lower, upper)

    # Bitwise-AND mask and original image
    img_been = cv2.bitwise_and(img_been, img_been, mask=mask)

    # cv2.imshow('test', img_been)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img_been_gray = cv2.cvtColor(img_been, cv2.COLOR_BGR2GRAY)

    min_bean_R = int(img.shape[1] * 0.01)
    max_bean_R = int(img.shape[1] * 0.017)

    # for overlab_rate in np.arange(0.4, 1.25, 0.05):
    #     ans = bean_check(img_been_gray, overlab_rate, dish_x, dish_y, dish_r)
    #     print("Q:", index, "\tOV_RATE:", round(overlab_rate, 2), "\tGUESS:", ans, "\tTRUE:", anslist[index - 1], "\tDIFF:", ans - anslist[index - 1], "\tERR RATE:", round(abs(100 * (ans - anslist[index - 1]) / anslist[index - 1])), "%")

    circles = cv2.HoughCircles(img_been_gray, cv2.HOUGH_GRADIENT, 1, max_bean_R * 0.8, param1=80, param2=9, minRadius=min_bean_R, maxRadius=max_bean_R)
    ans = 0
    for i in circles[0]:
        a = int(i[0])
        b = int(i[1])

        bean_size_diff = max_bean_R - min_bean_R
        bean_size_color = (int(i[2]) - min_bean_R) / (bean_size_diff / 255)
        if (((a - dish_x) ** 2 + (b - dish_y) ** 2) < (dish_r * 0.97) ** 2):
            cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (0, bean_size_color, 0), 1)
            ans += 1

    # print("Q:", index, "\tGUESS:", ans, "\tTRUE:", anslist[index - 1], "\tDIFF:", ans - anslist[index - 1], "\tERR RATE:", round(abs(100 * (ans - anslist[index - 1]) / anslist[index - 1])), "%")
    ans_org = ans
    diff_by_overlab = (bean_check(img_been_gray, 0.65, dish_x, dish_y, dish_r) - bean_check(img_been_gray, 0.95, dish_x, dish_y, dish_r)) / ans
    #print(round(diff_by_overlab, 3))
    if diff_by_overlab > 0.05 and ans_org < 225:
        ans = int(ans * 1.2)
    if diff_by_overlab > 0.05 and ans_org >= 225:
        ans = int(19 * (ans - 225) / 3)


    print("Q:", index, "\tGUESS_0:", ans_org, "\tGUESS_1:", ans,"\tTRUE:", anslist[index - 1], "\tDIFF:", ans - anslist[index - 1], "\tERR RATE:", round(abs(100 * (ans - anslist[index - 1]) / anslist[index - 1])), "%")
    #print('-----')



    #cv2.imwrite('test_result.jpg', img)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()