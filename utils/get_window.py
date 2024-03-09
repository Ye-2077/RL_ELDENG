import win32gui

def enum_windows_proc(hwnd, lParam):
    """
    Callback function for EnumWindows to check each window title
    """

    if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
        lParam.append(hwnd)
    return True


def find_window_by_partial_title(partial_title):
    """
    Find window by partial title.
    """

    hwnds = []
    win32gui.EnumWindows(enum_windows_proc, hwnds)
    for hwnd in hwnds:
        if partial_title.lower() in win32gui.GetWindowText(hwnd).lower():
            return hwnd
    return None


def set_foreground_window(hwnd):
    """
    Set the window with the given hwnd to the foreground
    """

    win32gui.SetForegroundWindow(hwnd)


def select_window_by_partial_title(partial_title):
    """
    Select the window with the given partial title by bringing it to the foreground
    """

    hwnd = find_window_by_partial_title(partial_title)
    if hwnd:
        set_foreground_window(hwnd)
        print(f"Window with partial title '{partial_title}' found and brought to foreground.")
    else:
        print(f"Window with partial title '{partial_title}' not found.")




## Example usage ##
# if __name__ == "__main__":
#     partial_window_title = "ELDEN RING"
#     select_window_by_partial_title(partial_window_title)
