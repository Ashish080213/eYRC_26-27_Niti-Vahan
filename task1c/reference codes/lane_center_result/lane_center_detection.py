import cv2
import numpy as np
import pandas as pd


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_VIDEO = r"C:\Users\varma\Documents\Competitions\eYRC 26-27\Niti Vahan\Task1\Code Files\eYRC_26-27_Niti-Vahan\task1c\public\clip_19.mp4"

Y_REF = 380

OUTPUT_VIDEO = "annotated_lane_center.mp4"
OUTPUT_CSV = "center_points.csv"


# ============================================================
# DETECT LANE
# ============================================================

def detect_lane(frame, y_ref=Y_REF):

    h, w = frame.shape[:2]

    # ========================================================
    # HSV CONVERSION
    # ========================================================

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # ========================================================
    # YELLOW MASK
    # ========================================================

    lower_yellow = np.array([
        15,
        70,
        70
    ])

    upper_yellow = np.array([
        40,
        255,
        255
    ])

    yellow_mask = cv2.inRange(
        hsv,
        lower_yellow,
        upper_yellow
    )

    # ========================================================
    # WHITE MASK
    # ========================================================

    lower_white = np.array([
        0,
        0,
        150
    ])

    upper_white = np.array([
        180,
        80,
        255
    ])

    white_mask = cv2.inRange(
        hsv,
        lower_white,
        upper_white
    )

    # ========================================================
    # REMOVE SMALL NOISE
    # ========================================================

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # ========================================================
    # ROAD ROI
    # ========================================================

    roi_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    polygon = np.array([
        [
            (0, h - 5),
            (150, int(h * 0.20)),
            (490, int(h * 0.20)),
            (w, h - 5)
        ]
    ], dtype=np.int32)

    cv2.fillPoly(
        roi_mask,
        polygon,
        255
    )

    yellow_roi = cv2.bitwise_and(
        yellow_mask,
        roi_mask
    )

    white_roi = cv2.bitwise_and(
        white_mask,
        roi_mask
    )

    # ========================================================
    # EDGE DETECTION
    # ========================================================

    yellow_edges = cv2.Canny(
        yellow_roi,
        50,
        150
    )

    white_edges = cv2.Canny(
        white_roi,
        50,
        150
    )

    # ========================================================
    # HOUGH LINES
    # ========================================================

    yellow_lines = cv2.HoughLinesP(
        yellow_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=40
    )

    white_lines = cv2.HoughLinesP(
        white_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=40
    )

    # ========================================================
    # GET X AT Y_REF
    # ========================================================

    def get_x_at_y(
        x1,
        y1,
        x2,
        y2,
        target_y
    ):

        if y2 == y1:
            return None

        return (
            x1
            + (target_y - y1)
            * (x2 - x1)
            / (y2 - y1)
        )

    # ========================================================
    # EXTRACT LINE CANDIDATES
    # ========================================================

    yellow_candidates = []

    if yellow_lines is not None:

        for line in yellow_lines:

            x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1

            if abs(dy) < 10:
                continue

            slope = dx / dy

            x_ref = get_x_at_y(
                x1,
                y1,
                x2,
                y2,
                y_ref
            )

            if x_ref is None:
                continue

            length = np.hypot(
                dx,
                dy
            )

            y_low = min(
                y1,
                y2
            )

            y_high = max(
                y1,
                y2
            )

            # Yellow boundary should be reasonably slanted
            if (
                abs(slope) > 0.20
                and 0 <= x_ref < w
                and y_low < y_ref + 40
                and y_high > 100
            ):

                yellow_candidates.append(
                    (x_ref, length, slope)
                )

    # ========================================================
    # WHITE LINE CANDIDATES
    # ========================================================

    white_candidates = []

    if white_lines is not None:

        for line in white_lines:

            x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1

            if abs(dy) < 10:
                continue

            slope = dx / dy

            x_ref = get_x_at_y(
                x1,
                y1,
                x2,
                y2,
                y_ref
            )

            if x_ref is None:
                continue

            length = np.hypot(
                dx,
                dy
            )

            y_low = min(
                y1,
                y2
            )

            y_high = max(
                y1,
                y2
            )

            if (
                abs(slope) > 0.15
                and 0 <= x_ref < w
                and y_low < y_ref + 40
                and y_high > 100
            ):

                white_candidates.append(
                    (x_ref, length, slope)
                )

    # ========================================================
    # SELECT BEST WHITE CENTERLINE
    # ========================================================

    white_x = None

    if len(white_candidates) > 0:

        # Prefer lines near the image center.
        # This prevents random white objects near the edges
        # from becoming the centerline.

        scored = []

        for x, length, slope in white_candidates:

            distance_from_center = abs(
                x - w / 2
            )

            score = (
                length
                - 0.5 * distance_from_center
            )

            scored.append(
                (score, x, length, slope)
            )

        scored.sort(
            reverse=True
        )

        # Take several strongest candidates
        selected = scored[
            :min(5, len(scored))
        ]

        white_x = np.median([
            item[1]
            for item in selected
        ])

    # ========================================================
    # SELECT YELLOW BOUNDARY
    # ========================================================

    yellow_x = None

    if len(yellow_candidates) > 0:

        # At the bottom of the image the yellow boundary
        # should generally be toward one of the sides.

        left_yellow = [
            item
            for item in yellow_candidates
            if item[0] < w / 2
        ]

        right_yellow = [
            item
            for item in yellow_candidates
            if item[0] >= w / 2
        ]

        # Select the yellow boundary farther from
        # the detected white centerline.

        if white_x is not None:

            if len(left_yellow) > 0:

                best_left = max(
                    left_yellow,
                    key=lambda item: item[1]
                )

            else:

                best_left = None

            if len(right_yellow) > 0:

                best_right = max(
                    right_yellow,
                    key=lambda item: item[1]
                )

            else:

                best_right = None

            if best_left is not None and best_right is not None:

                if abs(
                    best_left[0] - white_x
                ) > abs(
                    best_right[0] - white_x
                ):

                    yellow_x = best_left[0]

                else:

                    yellow_x = best_right[0]

            elif best_left is not None:

                yellow_x = best_left[0]

            elif best_right is not None:

                yellow_x = best_right[0]

        else:

            # If white line is not found,
            # use strongest yellow candidate.

            best_yellow = max(
                yellow_candidates,
                key=lambda item: item[1]
            )

            yellow_x = best_yellow[0]

    # ========================================================
    # DETERMINE LANE
    # ========================================================

    lane = "unknown"
    center_x = None

    if white_x is not None:

        # ----------------------------------------------------
        # WHITE CENTERLINE RIGHT OF VEHICLE
        # ----------------------------------------------------

        if white_x > w / 2:

            lane = "left"

            if yellow_x is not None:

                # Yellow is left boundary
                if yellow_x < white_x:

                    center_x = (
                        yellow_x
                        + white_x
                    ) / 2

        # ----------------------------------------------------
        # WHITE CENTERLINE LEFT OF VEHICLE
        # ----------------------------------------------------

        else:

            lane = "right"

            if yellow_x is not None:

                # Yellow is right boundary
                if yellow_x > white_x:

                    center_x = (
                        white_x
                        + yellow_x
                    ) / 2

    # ========================================================
    # FALLBACK
    # ========================================================

    # If both boundaries are not available,
    # still keep the lane classification from the
    # white centerline.

    if center_x is None:

        if white_x is not None:

            if lane == "left":

                # Estimate lane center using image center
                center_x = (
                    white_x
                    + w / 2
                ) / 2

            elif lane == "right":

                center_x = (
                    white_x
                    + w / 2
                ) / 2

    return (
        yellow_x,
        white_x,
        center_x,
        lane,
        yellow_mask,
        white_mask,
        yellow_edges,
        white_edges
    )


# ============================================================
# MAIN
# ============================================================

def main():

    cap = cv2.VideoCapture(
        INPUT_VIDEO
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print("================================")
    print("VIDEO INFORMATION")
    print("================================")
    print(f"FPS       : {fps:.2f}")
    print(f"Resolution: {width} x {height}")
    print(f"Frames    : {frame_count}")
    print("================================")
    print()

    # ========================================================
    # VIDEO WRITER
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    results = []

    frame_number = 0

    # ========================================================
    # PROCESS FRAMES
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        (
            yellow_x,
            white_x,
            center_x,
            lane,
            yellow_mask,
            white_mask,
            yellow_edges,
            white_edges
        ) = detect_lane(
            frame,
            Y_REF
        )

        output = frame.copy()

        # ====================================================
        # Y_REF
        # ====================================================

        cv2.line(
            output,
            (0, Y_REF),
            (width, Y_REF),
            (255, 0, 255),
            1
        )

        # ====================================================
        # YELLOW BOUNDARY
        # ====================================================

        if yellow_x is not None:

            yellow_x_int = int(
                yellow_x
            )

            cv2.circle(
                output,
                (
                    yellow_x_int,
                    Y_REF
                ),
                8,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                output,
                f"Yellow: {yellow_x:.0f}",
                (
                    yellow_x_int - 60,
                    Y_REF - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

        # ====================================================
        # WHITE CENTERLINE
        # ====================================================

        if white_x is not None:

            white_x_int = int(
                white_x
            )

            cv2.circle(
                output,
                (
                    white_x_int,
                    Y_REF
                ),
                8,
                (255, 255, 255),
                -1
            )

            cv2.line(
                output,
                (
                    white_x_int,
                    100
                ),
                (
                    white_x_int,
                    height
                ),
                (255, 255, 255),
                2
            )

            cv2.putText(
                output,
                f"White: {white_x:.0f}",
                (
                    white_x_int - 60,
                    Y_REF + 30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        # ====================================================
        # LANE CENTER
        # ====================================================

        if center_x is not None:

            cx = int(
                center_x
            )

            cv2.circle(
                output,
                (
                    cx,
                    Y_REF
                ),
                10,
                (0, 0, 255),
                -1
            )

            cv2.line(
                output,
                (
                    cx,
                    Y_REF - 80
                ),
                (
                    cx,
                    Y_REF + 80
                ),
                (0, 0, 255),
                3
            )

            cv2.putText(
                output,
                f"Center: {center_x:.0f}",
                (
                    20,
                    40
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                output,
                "Center: UNKNOWN",
                (
                    20,
                    40
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ====================================================
        # LANE
        # ====================================================

        cv2.putText(
            output,
            f"LANE: {lane.upper()}",
            (
                20,
                80
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # ====================================================
        # FRAME NUMBER
        # ====================================================

        cv2.putText(
            output,
            f"Frame: {frame_number}",
            (
                20,
                120
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ====================================================
        # SAVE VIDEO
        # ====================================================

        writer.write(
            output
        )

        # ====================================================
        # SAVE CSV
        # ====================================================

        results.append({
            "frame": frame_number,
            "yellow_x": yellow_x,
            "white_x": white_x,
            "center_x": center_x,
            "lane": lane,
            "y_ref": Y_REF
        })

        # ====================================================
        # PRINT
        # ====================================================

        if frame_number % 10 == 0:

            print(
                f"Frame {frame_number:5d} | "
                f"Yellow = {yellow_x!s:>8} | "
                f"White = {white_x!s:>8} | "
                f"Center = {center_x!s:>8} | "
                f"Lane = {lane}"
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Niti Vahan Lane Detection",
            output
        )

        # 100 ms = approximately 10 FPS playback
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break

        frame_number += 1

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    writer.release()

    cv2.destroyAllWindows()

    # ========================================================
    # SAVE CSV
    # ========================================================

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("================================")
    print("PROCESSING COMPLETE")
    print("================================")
    print(
        f"Output video : {OUTPUT_VIDEO}"
    )
    print(
        f"Output CSV   : {OUTPUT_CSV}"
    )
    print(
        f"Frames       : {len(df)}"
    )
    print("================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()