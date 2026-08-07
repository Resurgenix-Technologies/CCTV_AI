import sys
import cv2


def get_side(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Returns cross product value to determine which side of line (x1,y1)-(x2,y2) point (px,py) falls on."""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def calibrate_camera_lines(quadrant_names: list[str], get_camera_frame_fn) -> dict[str, tuple | None]:
    """Interactive GUI loop for setting boundary lines per camera feed using mouse clicks."""
    camera_lines = {}
    current_points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 2:
            current_points.append((x, y))
            print(f"Point {len(current_points)} set at: ({x}, {y})")

    cv2.namedWindow("Set Line")
    cv2.setMouseCallback("Set Line", mouse_callback)

    for cam_name in quadrant_names:
        current_points = []
        print(f"\n=== {cam_name} ke liye line set karo ===")
        print("2 points click karo, 'c' se confirm, 'r' se reset, 's' se skip")

        while True:
            cam_frame = get_camera_frame_fn(cam_name)
            if cam_frame is None:
                continue

            quad_frame = cv2.resize(cam_frame, (960, 540))
            display_frame = quad_frame.copy()

            for pt in current_points:
                cv2.circle(display_frame, pt, 6, (0, 0, 255), -1)

            if len(current_points) == 2:
                cv2.line(display_frame, current_points[0], current_points[1], (255, 0, 0), 2)
                cv2.putText(
                    display_frame,
                    "Press 'c' to confirm, 'r' to reset",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                )
            else:
                cv2.putText(
                    display_frame,
                    f"{cam_name}: Click point {len(current_points)+1}/2",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                )

            cv2.putText(
                display_frame,
                "Press 's' to skip this camera",
                (10, 510),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Set Line", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                current_points = []
            elif key == ord('c') and len(current_points) == 2:
                camera_lines[cam_name] = (current_points[0], current_points[1])
                print(f"{cam_name} line confirmed: {camera_lines[cam_name]}")
                break
            elif key == ord('s'):
                camera_lines[cam_name] = None
                print(f"{cam_name} skipped - full frame will be processed.")
                break
            elif key == ord('q'):
                cv2.destroyWindow("Set Line")
                sys.exit(0)

    cv2.destroyWindow("Set Line")
    print("\nSetup complete. Starting processing...\n")
    return camera_lines
