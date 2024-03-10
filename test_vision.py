from utils.utils_vision import *
from utils.get_window import *


if __name__ == "__main__":
    partial_window_title = "ELDEN RING"
    select_window_by_partial_title(partial_window_title)

    while True:
        screen = grab_screen(region=(200,90,850,500))
        # player_blood = grab_screen2hsv((80,60,500,71))
        # boss_blood = grab_screen2hsv((245,500,800,505))
        # player_blood, dilate = grab_blood(player_blood)
        # boss_blood, dilate = grab_blood(boss_blood)
        # print(player_blood)
        cv2.imshow('window', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break