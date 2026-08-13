"""Send a real X11 BackSpace key event to the MuJoCo window.

This is used when a physical keyboard "Back" key press works in MuJoCo, but
publishing a logical key through the ZMQ keyboard channel does not, because the
MuJoCo reset is tied to the GUI window's real keyboard event path.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import POINTER, byref, c_char_p, c_int, c_long, c_ubyte, c_uint, c_ulong
import time


XK_BACKSPACE = 0xFF08


class XWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("width", c_int),
        ("height", c_int),
        ("border_width", c_int),
        ("depth", c_int),
        ("visual", ctypes.c_void_p),
        ("root", c_ulong),
        ("class", c_int),
        ("bit_gravity", c_int),
        ("win_gravity", c_int),
        ("backing_store", c_int),
        ("backing_planes", c_ulong),
        ("backing_pixel", c_ulong),
        ("save_under", c_int),
        ("colormap", c_ulong),
        ("map_installed", c_int),
        ("map_state", c_int),
        ("all_event_masks", c_long),
        ("your_event_mask", c_long),
        ("do_not_propagate_mask", c_long),
        ("override_redirect", c_int),
        ("screen", ctypes.c_void_p),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a real BackSpace key to a MuJoCo X11 window.")
    parser.add_argument(
        "--display",
        default=None,
        help="X11 display to use. Defaults to DISPLAY from the environment.",
    )
    parser.add_argument(
        "--title-contains",
        default="mujoco",
        help="Case-insensitive substring used to find the MuJoCo window title.",
    )
    parser.add_argument(
        "--keysym",
        default="BackSpace",
        help="X11 keysym to send, for example BackSpace or 9. Defaults to BackSpace.",
    )
    parser.add_argument(
        "--focus-delay-sec",
        type=float,
        default=0.15,
        help="Delay after focusing the window before sending the key.",
    )
    parser.add_argument(
        "--restore-delay-sec",
        type=float,
        default=0.05,
        help="Delay before restoring the previous focused window.",
    )
    return parser


def load_x11():
    libx11 = ctypes.cdll.LoadLibrary("libX11.so.6")
    libxtst = ctypes.cdll.LoadLibrary("libXtst.so.6")

    libx11.XOpenDisplay.argtypes = [c_char_p]
    libx11.XOpenDisplay.restype = ctypes.c_void_p

    libx11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    libx11.XDefaultRootWindow.restype = c_ulong

    libx11.XQueryTree.argtypes = [
        ctypes.c_void_p,
        c_ulong,
        POINTER(c_ulong),
        POINTER(c_ulong),
        POINTER(POINTER(c_ulong)),
        POINTER(c_uint),
    ]
    libx11.XQueryTree.restype = c_int

    libx11.XFetchName.argtypes = [ctypes.c_void_p, c_ulong, POINTER(c_char_p)]
    libx11.XFetchName.restype = c_int

    libx11.XGetWindowAttributes.argtypes = [ctypes.c_void_p, c_ulong, POINTER(XWindowAttributes)]
    libx11.XGetWindowAttributes.restype = c_int

    libx11.XGetInputFocus.argtypes = [ctypes.c_void_p, POINTER(c_ulong), POINTER(c_int)]
    libx11.XGetInputFocus.restype = c_int

    libx11.XSetInputFocus.argtypes = [ctypes.c_void_p, c_ulong, c_int, c_ulong]
    libx11.XSetInputFocus.restype = c_int

    libx11.XRaiseWindow.argtypes = [ctypes.c_void_p, c_ulong]
    libx11.XRaiseWindow.restype = c_int

    libx11.XFlush.argtypes = [ctypes.c_void_p]
    libx11.XFlush.restype = c_int

    libx11.XSync.argtypes = [ctypes.c_void_p, c_int]
    libx11.XSync.restype = c_int

    libx11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, c_ulong]
    libx11.XKeysymToKeycode.restype = c_ubyte

    libx11.XStringToKeysym.argtypes = [c_char_p]
    libx11.XStringToKeysym.restype = c_ulong

    libx11.XFree.argtypes = [ctypes.c_void_p]
    libx11.XFree.restype = c_int

    libx11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    libx11.XCloseDisplay.restype = c_int

    libxtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, c_uint, c_int, c_ulong]
    libxtst.XTestFakeKeyEvent.restype = c_int

    return libx11, libxtst


def get_window_name(libx11, display, window: int) -> str | None:
    name_ptr = c_char_p()
    ok = libx11.XFetchName(display, window, byref(name_ptr))
    if not ok or not name_ptr.value:
        return None
    try:
        return name_ptr.value.decode("utf-8", errors="ignore")
    finally:
        libx11.XFree(name_ptr)


def is_viewable(libx11, display, window: int) -> bool:
    attrs = XWindowAttributes()
    ok = libx11.XGetWindowAttributes(display, window, byref(attrs))
    return bool(ok and attrs.map_state == 2)


def iter_windows(libx11, display, window: int):
    root = c_ulong()
    parent = c_ulong()
    children = POINTER(c_ulong)()
    nchildren = c_uint()

    status = libx11.XQueryTree(display, window, byref(root), byref(parent), byref(children), byref(nchildren))
    if not status:
        return

    try:
        for i in range(nchildren.value):
            child = children[i]
            yield child
            yield from iter_windows(libx11, display, child)
    finally:
        if children:
            libx11.XFree(children)


def find_target_window(libx11, display, title_contains: str) -> int | None:
    root = libx11.XDefaultRootWindow(display)
    needle = title_contains.lower()
    for window in iter_windows(libx11, display, root):
        if not is_viewable(libx11, display, window):
            continue
        title = get_window_name(libx11, display, window)
        if title and needle in title.lower():
            return window
    return None


def main() -> int:
    args = build_parser().parse_args()
    libx11, libxtst = load_x11()

    display_name = args.display.encode() if args.display else None
    display = libx11.XOpenDisplay(display_name)
    if not display:
        print("[send_mujoco_back_key] Failed to open X11 display.")
        return 1

    try:
        target = find_target_window(libx11, display, args.title_contains)
        if target is None:
            print(
                "[send_mujoco_back_key] Could not find a visible window with title containing:",
                args.title_contains,
            )
            return 2

        old_focus = c_ulong()
        revert_to = c_int()
        libx11.XGetInputFocus(display, byref(old_focus), byref(revert_to))

        libx11.XRaiseWindow(display, target)
        libx11.XSetInputFocus(display, target, 1, 0)
        libx11.XFlush(display)
        libx11.XSync(display, 0)
        time.sleep(max(args.focus_delay_sec, 0.0))

        keysym_name = str(args.keysym)
        if keysym_name.lower() in {"back", "backspace"}:
            keysym = XK_BACKSPACE
            keysym_name = "BackSpace"
        else:
            keysym = libx11.XStringToKeysym(keysym_name.encode("utf-8"))
        if keysym == 0:
            print(f"[send_mujoco_back_key] Failed to resolve keysym: {args.keysym}")
            return 3

        keycode = libx11.XKeysymToKeycode(display, keysym)
        if keycode == 0:
            print(f"[send_mujoco_back_key] Failed to resolve keycode for keysym: {keysym_name}")
            return 3

        libxtst.XTestFakeKeyEvent(display, keycode, 1, 0)
        libxtst.XTestFakeKeyEvent(display, keycode, 0, 0)
        libx11.XFlush(display)
        libx11.XSync(display, 0)
        print(f"[send_mujoco_back_key] Sent {keysym_name} to window id {target}.")

        time.sleep(max(args.restore_delay_sec, 0.0))
        if old_focus.value:
            libx11.XSetInputFocus(display, old_focus.value, revert_to.value, 0)
            libx11.XFlush(display)
            libx11.XSync(display, 0)

        return 0
    finally:
        libx11.XCloseDisplay(display)


if __name__ == "__main__":
    raise SystemExit(main())
