from utils.utils_vision import *
from utils.get_window import *
import time  # 导入time模块
import cv2

if __name__ == "__main__":
    partial_window_title = "ELDEN RING"
    select_window_by_partial_title(partial_window_title)

    while True:
        start_time = time.time()  # 捕获开始时间

        screen = grab_screen(region=(200,90,850,500))
        hsv_screen = grab_screen2hsv((245,500,800,505))  # 假设这是boss血条的位置
        boss_blood, _ = grab_blood(hsv_screen)

        end_time = time.time()  # 捕获结束时间
        duration = end_time - start_time  # 计算处理时间

        print(f"Boss Blood: {boss_blood}, Processing Time: {duration} seconds")

        cv2.imshow('window', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
