import cv2
import numpy as np
import datetime
from scipy.spatial import distance
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
## Functions ##
def bean_checkside(img, overlab_rate, min_bean_R, max_bean_R):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, max_bean_R * overlab_rate, param1=80, param2=9, minRadius=min_bean_R, maxRadius=max_bean_R)
    ans = 0
    for i in circles[0]:
        a = int(i[0])
        b = int(i[1])
        ans += 1
    return ans

def bean_check(img, overlab_rate, dish_x, dish_y, dish_r):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, max_bean_R * overlab_rate, param1=80, param2=9, minRadius=min_bean_R, maxRadius=max_bean_R)
    ans = 0
    for i in circles[0]:
        a = int(i[0])
        b = int(i[1])
        if (((a - dish_x) ** 2 + (b - dish_y) ** 2) < (dish_r * 0.97) ** 2):
            ans += 1
    return ans
## Initial Settings ##
#f = open("../out/Kong_03.txt", 'w')
#d = datetime.datetime.now()
#number_of_cases = 0
#result = ""

open_array = [44,60,49,129,39,22,513,196,263,170,98,379,1600,2,31,5,1190,151,108,122,75,84,10,24,1375,7,1032,1429,1323,691]
open_data_array = np.empty((0,5), int)

hidden_array = [10,1070,59,37,57,140,982,6,870,89,125,1541,873,69,52,116,1600,4,143,1480,860,1594,24,923,14,154,449,1178,150,39]
hidden_data_array = np.empty((0,5), int)
error_array = []
percent_array = []

test_array = [44,60,49,129,39,22,513,196,263,170,98,379,1600,2,31,5,1190,151,108,122,75,84,10,24,1375,7,1032,1429,1323,691,10,1070,59,37,57,140,982,6,870,89,125,1541,873,69,52,116,1600,4,143,1480,860,1594,24,923,14,154,449,1178,150,39]
for index in range(1, 31):
    ## Image Read ##
    zfill_index = format(index, '02')
    img1 = cv2.imread(f"./Open/t{zfill_index}/1.jpg")
    img2 = cv2.imread(f"./Open/t{zfill_index}/2.jpg")
    img3 = cv2.imread(f"./Open/t{zfill_index}/3.jpg")
    img4 = cv2.imread(f"./Open/t{zfill_index}/4.jpg")
    img = cv2.imread(f"./Open/t{zfill_index}/5.jpg")
    #img = cv2.imread(f"../../Open/t{zfill_index}/5.jpg")



    ## Image Resizing ##
    img1 = cv2.resize(img1, dsize=(800, 600), interpolation=cv2.INTER_AREA)
    img1 = img1[160:450, 160:720]
    img2 = cv2.resize(img2, dsize=(800, 600), interpolation=cv2.INTER_AREA)
    img2 = img2[160:450, 160:720]
    img3 = cv2.resize(img3, dsize=(800, 600), interpolation=cv2.INTER_AREA)
    img3 = img3[160:450, 160:720]
    img4 = cv2.resize(img4, dsize=(800, 600), interpolation=cv2.INTER_AREA)
    img4 = img4[160:450, 160:720]
    img = cv2.resize(img, dsize=(800, 600), interpolation=cv2.INTER_AREA)

    ## Dish Mask ##
    # dish color range
    lower_white = np.array([0, 185, 185])
    upper_white = np.array([255, 255, 255])

    # masking
    mask1 = cv2.inRange(img1, lower_white, upper_white)
    mask_img1 = cv2.bitwise_and(img1, img1, mask=mask1)
    mask_img_gray1 = cv2.cvtColor(mask_img1, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.inRange(img2, lower_white, upper_white)
    mask_img2 = cv2.bitwise_and(img2, img2, mask=mask2)
    mask_img_gray2 = cv2.cvtColor(mask_img2, cv2.COLOR_BGR2GRAY)
    mask3 = cv2.inRange(img3, lower_white, upper_white)
    mask_img3 = cv2.bitwise_and(img3, img3, mask=mask3)
    mask_img_gray3 = cv2.cvtColor(mask_img3, cv2.COLOR_BGR2GRAY)
    mask4 = cv2.inRange(img4, lower_white, upper_white)
    mask_img4 = cv2.bitwise_and(img4, img4, mask=mask4)
    mask_img_gray4 = cv2.cvtColor(mask_img4, cv2.COLOR_BGR2GRAY)

    mask = cv2.inRange(img, lower_white, upper_white)

    mask_img = cv2.bitwise_and(img, img, mask=mask)
    mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('test', mask_img_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # dish check
    min_dish_R = int(img.shape[1] * 0.32)
    max_dish_R = int(img.shape[1] * 0.42)

    circles = cv2.HoughCircles(mask_img_gray, cv2.HOUGH_GRADIENT, 1, 100, param1=220, param2=35, minRadius=min_dish_R,
                               maxRadius=max_dish_R)

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

    # bean check
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

    circles = cv2.HoughCircles(img_been_gray, cv2.HOUGH_GRADIENT, 1, max_bean_R * 0.8, param1=80, param2=9,
                               minRadius=min_bean_R, maxRadius=max_bean_R)
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
    diff_by_overlab = (bean_check(img_been_gray, 0.65, dish_x, dish_y, dish_r) - bean_check(img_been_gray, 0.95, dish_x,
                                                                                            dish_y, dish_r)) / ans
    # print(round(diff_by_overlab, 3))
    if diff_by_overlab > 0.05 and ans_org < 225:
        ans = int(ans * 1.2)
    if diff_by_overlab > 0.05 and ans_org >= 225:
        ans = int(19 * (ans - 225) / 3)

    ## Bean Mask ##
    img_been1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img_been2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    img_been3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    img_been4 = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)

    # bean color range
    lower = np.array([8, 45, 0])
    upper = np.array([25, 255, 255])

    # bean size range
    min_bean_R1 = int(img1.shape[1] * 0.01)
    max_bean_R1 = int(img1.shape[1] * 0.017)
    min_bean_R2 = int(img2.shape[1] * 0.01)
    max_bean_R2 = int(img2.shape[1] * 0.017)
    min_bean_R3 = int(img3.shape[1] * 0.01)
    max_bean_R3 = int(img3.shape[1] * 0.017)
    min_bean_R4 = int(img4.shape[1] * 0.01)
    max_bean_R4 = int(img4.shape[1] * 0.017)


    img_been_gray1 = cv2.cvtColor(img_been1, cv2.COLOR_BGR2GRAY)
    img_been_gray2 = cv2.cvtColor(img_been2, cv2.COLOR_BGR2GRAY)
    img_been_gray3 = cv2.cvtColor(img_been3, cv2.COLOR_BGR2GRAY)
    img_been_gray4 = cv2.cvtColor(img_been4, cv2.COLOR_BGR2GRAY)

    ## Bean Detection ##
    circles1 = cv2.HoughCircles(img_been_gray1, cv2.HOUGH_GRADIENT, 1, max_bean_R1 * 0.8, param1=80, param2=9, minRadius=min_bean_R1, maxRadius=max_bean_R1)
    circles2 = cv2.HoughCircles(img_been_gray2, cv2.HOUGH_GRADIENT, 1, max_bean_R2 * 0.8, param1=80, param2=9, minRadius=min_bean_R2, maxRadius=max_bean_R2)
    circles3 = cv2.HoughCircles(img_been_gray3, cv2.HOUGH_GRADIENT, 1, max_bean_R3 * 0.8, param1=80, param2=9, minRadius=min_bean_R3, maxRadius=max_bean_R3)
    circles4 = cv2.HoughCircles(img_been_gray4, cv2.HOUGH_GRADIENT, 1, max_bean_R4 * 0.8, param1=80, param2=9, minRadius=min_bean_R4, maxRadius=max_bean_R4)

    ans1 = 0
    ans2 = 0
    ans3 = 0
    ans4 = 0

    for i in circles1[0]:
        bean_size_diff1 = max_bean_R1 - min_bean_R1
        bean_size_color1 = (int(i[2]) - min_bean_R1) / (bean_size_diff1 / 255)
        cv2.circle(mask_img1, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
        ans1 += 1
    ans_org1 = ans1
#    cv2.imshow('test', mask_img1)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    for i in circles2[0]:
        bean_size_diff2 = max_bean_R2 - min_bean_R2
        bean_size_color2 = (int(i[2]) - min_bean_R2) / (bean_size_diff2 / 255)
        cv2.circle(mask_img2, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
        ans2 += 1
    ans_org2 = ans2
#    cv2.imshow('test', mask_img2)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    for i in circles3[0]:
        bean_size_diff3 = max_bean_R3 - min_bean_R3
        bean_size_color3 = (int(i[2]) - min_bean_R3) / (bean_size_diff3 / 255)
        cv2.circle(mask_img3, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
        ans3 += 1
    ans_org3 = ans3
#    cv2.imshow('test', mask_img3)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    for i in circles4[0]:
        bean_size_diff4 = max_bean_R4 - min_bean_R4
        bean_size_color4 = (int(i[2]) - min_bean_R4) / (bean_size_diff4 / 255)
        cv2.circle(mask_img4, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
        ans4 += 1
    ans_org4 = ans4
#    cv2.imshow('test', mask_img4)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


    ## Placement Complexity Calculation ##
    diff_by_overlab1 = (bean_checkside(img_been_gray1, 0.65, min_bean_R1, max_bean_R1) - bean_checkside(img_been_gray1, 0.95, min_bean_R1, max_bean_R1)) / ans_org1
    diff_by_overlab2 = (bean_checkside(img_been_gray2, 0.65, min_bean_R2, max_bean_R2) - bean_checkside(img_been_gray2, 0.95, min_bean_R2, max_bean_R2)) / ans_org2
    diff_by_overlab3 = (bean_checkside(img_been_gray3, 0.65, min_bean_R3, max_bean_R3) - bean_checkside(img_been_gray3, 0.95, min_bean_R3, max_bean_R3)) / ans_org3
    diff_by_overlab4 = (bean_checkside(img_been_gray4, 0.65, min_bean_R4, max_bean_R4) - bean_checkside(img_been_gray4, 0.95, min_bean_R4, max_bean_R4)) / ans_org4

    ## Linear Trend Estimation ##
    if diff_by_overlab1 > 0.05 and ans_org1 < 225:
        ans1 = int(ans1 * 1.2)
    if diff_by_overlab1 > 0.05 and ans_org1 >= 225:
        ans1 = int(19 * (ans1 - 225) / 3)
    if diff_by_overlab2 > 0.05 and ans_org2 < 225:
        ans2 = int(ans2 * 1.2)
    if diff_by_overlab2 > 0.05 and ans_org2 >= 225:
        ans2 = int(19 * (ans2 - 225) / 3)
    if diff_by_overlab3 > 0.05 and ans_org3 < 225:
        ans3 = int(ans3 * 1.2)
    if diff_by_overlab3 > 0.05 and ans_org3 >= 225:
        ans3 = int(19 * (ans3 - 225) / 3)
    if diff_by_overlab4 > 0.05 and ans_org4 < 225:
        ans4 = int(ans4 * 1.2)
    if diff_by_overlab4 > 0.05 and ans_org4 >= 225:
        ans4 = int(19 * (ans4 - 225) / 3)


    ## Exception Handling ##
    if ans1 < 2:
        ans1 = 2
    if ans1 > 2000:
        ans1 = 1550
    if ans1 > 1800:
        ans1 = 1500
    if ans1 > 1600:
        ans1 = 1450
    ans1 = int(ans1)
    if ans2 < 2:
        ans2 = 2
    if ans2 > 2000:
        ans2 = 1550
    if ans2 > 1800:
        ans2 = 1500
    if ans2 > 1600:
        ans2 = 1450
    ans2 = int(ans2)
    if ans3 < 2:
        ans3 = 2
    if ans3 > 2000:
        ans3 = 1550
    if ans3 > 1800:
        ans3 = 1500
    if ans3 > 1600:
        ans3 = 1450
    ans3 = int(ans3)
    if ans4 < 2:
        ans4 = 2
    if ans4 > 2000:
        ans4 = 1550
    if ans4 > 1800:
        ans4 = 1500
    if ans4 > 1600:
        ans4 = 1450
    ans4 = int(ans4)



    ## Save Result ##
    #print("Q:", zfill_index, "\tGUESS_0:", ans_org1, "\tGUESS_1:", ans1)
    #print("Q:", zfill_index, "\tGUESS_0:", ans_org2, "\tGUESS_1:", ans2)
    #print("Q:", zfill_index, "\tGUESS_0:", ans_org3, "\tGUESS_1:", ans3)
    #print("Q:", zfill_index, "\tGUESS_0:", ans_org4, "\tGUESS_1:", ans4)
    #print("Q:", zfill_index, "\tGUESS_0:", ans_org, "\tGUESS_1:", ans)
    ans5 = ans
    ans_org5 = ans_org

    answer_vector = [ans_org1, ans_org2, ans_org3, ans_org4, ans_org5]
    open_data_array = np.append(open_data_array, np.array([answer_vector]), axis=0)
    #print(answer_vector)

for index in range(1, 31):
    ## Image Read ##
    zfill_index = format(index, '02')
    img1 = cv2.imread(f"./Hidden/t{zfill_index}/1.jpg")
    img2 = cv2.imread(f"./Hidden/t{zfill_index}/2.jpg")
    img3 = cv2.imread(f"./Hidden/t{zfill_index}/3.jpg")
    img4 = cv2.imread(f"./Hidden/t{zfill_index}/4.jpg")
    img = cv2.imread(f"./Hidden/t{zfill_index}/5.jpg")
    #img = cv2.imread(f"../../Open/t{zfill_index}/5.jpg")



    ## Image Resizing ##
    img1 = cv2.resize(img1, dsize=(800, 600), interpolation=cv2.INTER_AREA)
    img1 = img1[160:450, 160:720]
    img2 = cv2.resize(img2, dsize=(800, 600), interpolation=cv2.INTER_AREA)
    img2 = img2[160:450, 160:720]
    img3 = cv2.resize(img3, dsize=(800, 600), interpolation=cv2.INTER_AREA)
    img3 = img3[160:450, 160:720]
    img4 = cv2.resize(img4, dsize=(800, 600), interpolation=cv2.INTER_AREA)
    img4 = img4[160:450, 160:720]
    img = cv2.resize(img, dsize=(800, 600), interpolation=cv2.INTER_AREA)

    ## Dish Mask ##
    # dish color range
    lower_white = np.array([0, 185, 185])
    upper_white = np.array([255, 255, 255])

    # masking
    mask1 = cv2.inRange(img1, lower_white, upper_white)
    mask_img1 = cv2.bitwise_and(img1, img1, mask=mask1)
    mask_img_gray1 = cv2.cvtColor(mask_img1, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.inRange(img2, lower_white, upper_white)
    mask_img2 = cv2.bitwise_and(img2, img2, mask=mask2)
    mask_img_gray2 = cv2.cvtColor(mask_img2, cv2.COLOR_BGR2GRAY)
    mask3 = cv2.inRange(img3, lower_white, upper_white)
    mask_img3 = cv2.bitwise_and(img3, img3, mask=mask3)
    mask_img_gray3 = cv2.cvtColor(mask_img3, cv2.COLOR_BGR2GRAY)
    mask4 = cv2.inRange(img4, lower_white, upper_white)
    mask_img4 = cv2.bitwise_and(img4, img4, mask=mask4)
    mask_img_gray4 = cv2.cvtColor(mask_img4, cv2.COLOR_BGR2GRAY)

    mask = cv2.inRange(img, lower_white, upper_white)

    mask_img = cv2.bitwise_and(img, img, mask=mask)
    mask_img_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('test', mask_img_gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # dish check
    min_dish_R = int(img.shape[1] * 0.32)
    max_dish_R = int(img.shape[1] * 0.42)

    circles = cv2.HoughCircles(mask_img_gray, cv2.HOUGH_GRADIENT, 1, 100, param1=220, param2=35, minRadius=min_dish_R,
                               maxRadius=max_dish_R)

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

    # bean check
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

    circles = cv2.HoughCircles(img_been_gray, cv2.HOUGH_GRADIENT, 1, max_bean_R * 0.8, param1=80, param2=9,
                               minRadius=min_bean_R, maxRadius=max_bean_R)
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
    diff_by_overlab = (bean_check(img_been_gray, 0.65, dish_x, dish_y, dish_r) - bean_check(img_been_gray, 0.95, dish_x,
                                                                                            dish_y, dish_r)) / ans
    # print(round(diff_by_overlab, 3))
    if diff_by_overlab > 0.05 and ans_org < 225:
        ans = int(ans * 1.2)
    if diff_by_overlab > 0.05 and ans_org >= 225:
        ans = int(19 * (ans - 225) / 3)

    ## Bean Mask ##
    img_been1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img_been2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    img_been3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
    img_been4 = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)

    # bean color range
    lower = np.array([8, 45, 0])
    upper = np.array([25, 255, 255])

    # bean size range
    min_bean_R1 = int(img1.shape[1] * 0.01)
    max_bean_R1 = int(img1.shape[1] * 0.017)
    min_bean_R2 = int(img2.shape[1] * 0.01)
    max_bean_R2 = int(img2.shape[1] * 0.017)
    min_bean_R3 = int(img3.shape[1] * 0.01)
    max_bean_R3 = int(img3.shape[1] * 0.017)
    min_bean_R4 = int(img4.shape[1] * 0.01)
    max_bean_R4 = int(img4.shape[1] * 0.017)


    img_been_gray1 = cv2.cvtColor(img_been1, cv2.COLOR_BGR2GRAY)
    img_been_gray2 = cv2.cvtColor(img_been2, cv2.COLOR_BGR2GRAY)
    img_been_gray3 = cv2.cvtColor(img_been3, cv2.COLOR_BGR2GRAY)
    img_been_gray4 = cv2.cvtColor(img_been4, cv2.COLOR_BGR2GRAY)

    ## Bean Detection ##
    circles1 = cv2.HoughCircles(img_been_gray1, cv2.HOUGH_GRADIENT, 1, max_bean_R1 * 0.8, param1=80, param2=9, minRadius=min_bean_R1, maxRadius=max_bean_R1)
    circles2 = cv2.HoughCircles(img_been_gray2, cv2.HOUGH_GRADIENT, 1, max_bean_R2 * 0.8, param1=80, param2=9, minRadius=min_bean_R2, maxRadius=max_bean_R2)
    circles3 = cv2.HoughCircles(img_been_gray3, cv2.HOUGH_GRADIENT, 1, max_bean_R3 * 0.8, param1=80, param2=9, minRadius=min_bean_R3, maxRadius=max_bean_R3)
    circles4 = cv2.HoughCircles(img_been_gray4, cv2.HOUGH_GRADIENT, 1, max_bean_R4 * 0.8, param1=80, param2=9, minRadius=min_bean_R4, maxRadius=max_bean_R4)

    ans1 = 0
    ans2 = 0
    ans3 = 0
    ans4 = 0

    for i in circles1[0]:
        bean_size_diff1 = max_bean_R1 - min_bean_R1
        bean_size_color1 = (int(i[2]) - min_bean_R1) / (bean_size_diff1 / 255)
        cv2.circle(mask_img1, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
        ans1 += 1
    ans_org1 = ans1
#    cv2.imshow('test', mask_img1)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    for i in circles2[0]:
        bean_size_diff2 = max_bean_R2 - min_bean_R2
        bean_size_color2 = (int(i[2]) - min_bean_R2) / (bean_size_diff2 / 255)
        cv2.circle(mask_img2, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
        ans2 += 1
    ans_org2 = ans2
#    cv2.imshow('test', mask_img2)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    for i in circles3[0]:
        bean_size_diff3 = max_bean_R3 - min_bean_R3
        bean_size_color3 = (int(i[2]) - min_bean_R3) / (bean_size_diff3 / 255)
        cv2.circle(mask_img3, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
        ans3 += 1
    ans_org3 = ans3
#    cv2.imshow('test', mask_img3)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    for i in circles4[0]:
        bean_size_diff4 = max_bean_R4 - min_bean_R4
        bean_size_color4 = (int(i[2]) - min_bean_R4) / (bean_size_diff4 / 255)
        cv2.circle(mask_img4, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
        ans4 += 1
    ans_org4 = ans4
#    cv2.imshow('test', mask_img4)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


    ## Placement Complexity Calculation ##
    diff_by_overlab1 = (bean_checkside(img_been_gray1, 0.65, min_bean_R1, max_bean_R1) - bean_checkside(img_been_gray1, 0.95, min_bean_R1, max_bean_R1)) / ans_org1
    diff_by_overlab2 = (bean_checkside(img_been_gray2, 0.65, min_bean_R2, max_bean_R2) - bean_checkside(img_been_gray2, 0.95, min_bean_R2, max_bean_R2)) / ans_org2
    diff_by_overlab3 = (bean_checkside(img_been_gray3, 0.65, min_bean_R3, max_bean_R3) - bean_checkside(img_been_gray3, 0.95, min_bean_R3, max_bean_R3)) / ans_org3
    diff_by_overlab4 = (bean_checkside(img_been_gray4, 0.65, min_bean_R4, max_bean_R4) - bean_checkside(img_been_gray4, 0.95, min_bean_R4, max_bean_R4)) / ans_org4

    ## Linear Trend Estimation ##
    if diff_by_overlab1 > 0.05 and ans_org1 < 225:
        ans1 = int(ans1 * 1.2)
    if diff_by_overlab1 > 0.05 and ans_org1 >= 225:
        ans1 = int(19 * (ans1 - 225) / 3)
    if diff_by_overlab2 > 0.05 and ans_org2 < 225:
        ans2 = int(ans2 * 1.2)
    if diff_by_overlab2 > 0.05 and ans_org2 >= 225:
        ans2 = int(19 * (ans2 - 225) / 3)
    if diff_by_overlab3 > 0.05 and ans_org3 < 225:
        ans3 = int(ans3 * 1.2)
    if diff_by_overlab3 > 0.05 and ans_org3 >= 225:
        ans3 = int(19 * (ans3 - 225) / 3)
    if diff_by_overlab4 > 0.05 and ans_org4 < 225:
        ans4 = int(ans4 * 1.2)
    if diff_by_overlab4 > 0.05 and ans_org4 >= 225:
        ans4 = int(19 * (ans4 - 225) / 3)


    ## Exception Handling ##
    if ans1 < 2:
        ans1 = 2
    if ans1 > 2000:
        ans1 = 1550
    if ans1 > 1800:
        ans1 = 1500
    if ans1 > 1600:
        ans1 = 1450
    ans1 = int(ans1)
    if ans2 < 2:
        ans2 = 2
    if ans2 > 2000:
        ans2 = 1550
    if ans2 > 1800:
        ans2 = 1500
    if ans2 > 1600:
        ans2 = 1450
    ans2 = int(ans2)
    if ans3 < 2:
        ans3 = 2
    if ans3 > 2000:
        ans3 = 1550
    if ans3 > 1800:
        ans3 = 1500
    if ans3 > 1600:
        ans3 = 1450
    ans3 = int(ans3)
    if ans4 < 2:
        ans4 = 2
    if ans4 > 2000:
        ans4 = 1550
    if ans4 > 1800:
        ans4 = 1500
    if ans4 > 1600:
        ans4 = 1450
    ans4 = int(ans4)



    ## Save Result ##
    #print("Q:", zfill_index, "\tGUESS_0:", ans_org1, "\tGUESS_1:", ans1)
    #print("Q:", zfill_index, "\tGUESS_0:", ans_org2, "\tGUESS_1:", ans2)
    #print("Q:", zfill_index, "\tGUESS_0:", ans_org3, "\tGUESS_1:", ans3)
    #print("Q:", zfill_index, "\tGUESS_0:", ans_org4, "\tGUESS_1:", ans4)
    #print("Q:", zfill_index, "\tGUESS_0:", ans_org, "\tGUESS_1:", ans)
    ans5 = ans
    ans_org5 = ans_org

    answer_vector = [ans_org1, ans_org2, ans_org3, ans_org4, ans_org5]
    hidden_data_array = np.append(hidden_data_array, np.array([answer_vector]), axis=0)
'''
    eucl_dis_array = []
    for j in range(0, len(open_data_array)):
        eud = distance.euclidean(open_data_array[j], answer_vector)
        eucl_dis_array.append(eud)
        # print(eud)

    cindex = eucl_dis_array.index(min(eucl_dis_array))

    error = (hidden_array[index - 1] - open_array[cindex]) * 100 / hidden_array[index - 1]
    print( error )
    percent_array.append(abs(error))
    error = error * error
    print( error )
    error_array.append(error)
'''


train_features = np.array(open_data_array)
test_features = np.array(hidden_data_array)
train_labels = np.array(open_array)
test_labels = np.array(hidden_array)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

model = LinearRegression()

model.fit(train_features, train_labels)

predictions = model.predict(test_features)

for i in range(30):
    error = abs(predictions[i] - test_labels[i]) * 100 / test_labels[i]
    print(error)
    percent_array.append(error)
    error = error * error
    print(error)
    error_array.append(error)

print(model.score(train_features, train_labels))

print(model.score(test_features, test_labels))

print(model.coef_)
#print(percent_array)
#print(error_array)
#print(sum(percent_array))
#print(sum(error_array))