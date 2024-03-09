from utils.utils_vision import *
from utils.get_window import *


if __name__ == "__main__":
    partial_window_title = "ELDEN RING"
    select_window_by_partial_title(partial_window_title)

    while True:
        # screen = grab_screen(region=(left,top+40,right+200,bottom+120))
        player_blood = grab_screen2hsv((80,60,500,71))
        # boss_blood = grab_screen2hsv((245,500,800,505))
        player_blood, dilate = grab_blood(player_blood)
        # boss_blood, dilate = grab_blood(boss_blood)
        print(player_blood)
        cv2.imshow('window', dilate)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break