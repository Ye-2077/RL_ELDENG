import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api



#################################################
################## grab_screen ##################
#################################################
def grab_screen(region:tuple = None):
    """
    Grab the screen. of the specified region.

    Parameters:
    - region: 
        - e.g. (100,200,300,400). 
        - The region to grab. If None, grab the whole screen.
    """
    hwin = win32gui.GetDesktopWindow()

    if region:
            left, top, right, down = region
            width = right - left + 1
            height = down - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


def enum_windows_proc(hwnd, resultList):
    """
    Callback function for win32gui.EnumWindows.

    Parameters:
    - hwnd: The handle of the window.
    - resultList: The list to store the result.

    Returns:
    - None
    """
    window_text = win32gui.GetWindowText(hwnd)
    resultList.append((hwnd, window_text))


def find_windows_containing_text(text):
    """
    Find all windows containing the specified text.
    
    Parameters:
    - text: The text to search for.
    
    Returns:
    - windows: 
        - A list of tuples. 
        - Each tuple contains the handle of the window, the window text, and the window rect.
        - (hwnd, window_text, (left, top, right, bottom))
    """
    windows = []
    temp_list = []
    win32gui.EnumWindows(enum_windows_proc, temp_list)
    for hwnd, window_text in temp_list:
        if text.lower() in window_text.lower():
            rect = win32gui.GetWindowRect(hwnd)
            windows.append((hwnd, window_text, rect))
    return windows


def grab_screen2gray(region:tuple = None):
    """
    Grab the window and convert it to gray scale.

    Parameters:
    - region: 
        - e.g. (100,200,300,400). 
        - The region to grab. If None, grab the whole screen.
    """
    img = grab_screen(region)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def grab_screen2hsv(region:tuple = None):
    """
    Grab the window and convert it to HSV.

    Parameters:
    - region: 
        - e.g. (100,200,300,400). 
        - The region to grab. If None, grab the whole screen.
    """
    img = grab_screen(region)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img

#################################################
################## grab_blood ###################
#################################################
def grab_blood(hsv):

    lower_red1 = np.array([0, 125, 70])
    upper_red1 = np.array([15, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # fliter out the noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    blood_amount = cv2.countNonZero(dilated)

    return blood_amount, dilated