'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1C of Niti Vahan (NV) Theme.
*
*  Filename:         lane_detection.py
*  Created:          2026
*  Last Modified:    01-October-2026
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          < 1125 >
# Author List:      < Penumetsa Sri Sai Ashish Varma >
# Filename:         lane_detection.py
# Functions:        detect_lane
# Global variables: < List any global variables you add, "None" if you add none >


####################### IMPORT MODULES #######################
import argparse
import json
import os

import cv2
import numpy as np
##############################################################

# The only three values "lane" is allowed to take.
LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

Y_AXIS = 300

CANNY_LOW = 50
CANNY_HIGH = 150

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 15
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 50

MIN_LANE_WIDTH = 120
MAX_LANE_WIDTH = 600

INITIAL_CENTER_FRACTION = 0.50
INITIAL_WIDTH = 250.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def x_at_y(x1, y1, x2, y2, y):

    dy = y2 - y1

    if abs(dy) < 1e-6:
        return None

    return x1 + (y - y1) * (x2 - x1) / dy


def weighted_average(values, weights):

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(values) == 0:
        return None

    if np.sum(weights) <= 0:
        return float(np.mean(values))

    return float(np.average(values, weights=weights))


# ============================================================
# COLOR MASKS
# ============================================================

def create_color_masks(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --------------------------------------------------------
    # Yellow road marking
    # --------------------------------------------------------

    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 80, 80], dtype=np.uint8),
        np.array([45, 255, 255], dtype=np.uint8)
    )

    # --------------------------------------------------------
    # White road marking
    # --------------------------------------------------------

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 170], dtype=np.uint8),
        np.array([180, 90, 255], dtype=np.uint8)
    )

    # --------------------------------------------------------
    # Fast 7x7 neighborhood score
    # --------------------------------------------------------

    yellow_score_map = cv2.boxFilter(
        yellow_mask,
        ddepth=cv2.CV_32F,
        ksize=(7, 7),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE
    ) / 255.0

    white_score_map = cv2.boxFilter(
        white_mask,
        ddepth=cv2.CV_32F,
        ksize=(7, 7),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE
    ) / 255.0

    return yellow_score_map, white_score_map


# ============================================================
# LINE COLOR SCORE
# ============================================================

def line_mask_score(score_map, line):

    x1, y1, x2, y2 = line

    length = int(
        max(
            abs(x2 - x1),
            abs(y2 - y1)
        )
    ) + 1

    if length < 2:
        return 0.0

    xs = np.rint(
        np.linspace(
            x1,
            x2,
            length
        )
    ).astype(np.int32)

    ys = np.rint(
        np.linspace(
            y1,
            y2,
            length
        )
    ).astype(np.int32)

    xs = np.clip(
        xs,
        0,
        score_map.shape[1] - 1
    )

    ys = np.clip(
        ys,
        0,
        score_map.shape[0] - 1
    )

    return float(
        np.mean(
            score_map[ys, xs]
        )
    )


# ============================================================
# HOUGH LINE CANDIDATES
# ============================================================

def get_hough_candidates(frame):

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Canny
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH,
        apertureSize=3
    )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    roi_mask = np.zeros_like(edges)

    roi_polygon = np.array(
        [[
            (0, height - 5),
            (100, 80),
            (540, 80),
            (width - 1, height - 5)
        ]],
        dtype=np.int32
    )

    cv2.fillPoly(
        roi_mask,
        roi_polygon,
        255
    )

    roi_edges = cv2.bitwise_and(
        edges,
        roi_mask
    )

    # --------------------------------------------------------
    # Hough
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    if lines is None:
        return []

    # Handles both:
    # (N,1,4)
    # (N,4)

    lines = np.asarray(lines).reshape(-1, 4)

    # --------------------------------------------------------
    # Color score maps
    # --------------------------------------------------------

    yellow_score_map, white_score_map = create_color_masks(
        frame
    )

    candidates = []

    color_threshold = 0.12

    # --------------------------------------------------------
    # Process every Hough line
    # --------------------------------------------------------

    for raw_line in lines:

        x1, y1, x2, y2 = map(
            float,
            raw_line
        )

        dy = y2 - y1

        if abs(dy) < 5:
            continue

        length = float(
            np.hypot(
                x2 - x1,
                y2 - y1
            )
        )

        if length < HOUGH_MIN_LINE_LENGTH:
            continue

        # Project line to reference Y
        x_ref = x_at_y(
            x1,
            y1,
            x2,
            y2,
            Y_AXIS
        )

        if x_ref is None:
            continue

        # Allow slight extrapolation outside image
        if x_ref < -150 or x_ref > width + 150:
            continue

        y_low = min(
            y1,
            y2
        )

        y_high = max(
            y1,
            y2
        )

        if y_high < 90:
            continue

        line = (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2))
        )

        # ----------------------------------------------------
        # Color scoring
        # ----------------------------------------------------

        yellow_score = line_mask_score(
            yellow_score_map,
            line
        )

        white_score = line_mask_score(
            white_score_map,
            line
        )

        # ----------------------------------------------------
        # Determine line color
        # ----------------------------------------------------

        if (
            yellow_score > white_score
            and yellow_score >= color_threshold
        ):

            color = "yellow"
            color_score = yellow_score

        elif (
            white_score > yellow_score
            and white_score >= color_threshold
        ):

            color = "white"
            color_score = white_score

        else:
            continue

        candidates.append(
            {
                "x": float(x_ref),

                "length": length,

                "line": line,

                "color": color,

                "color_score": float(
                    color_score
                ),

                "yellow_score": float(
                    yellow_score
                ),

                "white_score": float(
                    white_score
                ),

                "y_low": float(
                    y_low
                ),

                "y_high": float(
                    y_high
                )
            }
        )

    return candidates


# ============================================================
# CLUSTER CANDIDATES
# ============================================================

def cluster_candidates(candidates):

    clusters = []

    for color in (
        "yellow",
        "white"
    ):

        color_candidates = sorted(
            [
                c
                for c in candidates
                if c["color"] == color
            ],
            key=lambda c: c["x"]
        )

        current = []

        for candidate in color_candidates:

            if not current:

                current = [
                    candidate
                ]

                continue

            if (
                candidate["x"]
                - current[-1]["x"]
                <= 8
            ):

                current.append(
                    candidate
                )

            else:

                clusters.append(
                    current
                )

                current = [
                    candidate
                ]

        if current:

            clusters.append(
                current
            )

    representatives = []

    for cluster in clusters:

        weights = np.array(
            [
                c["length"]
                * c["color_score"]
                for c in cluster
            ],
            dtype=float
        )

        xs = np.array(
            [
                c["x"]
                for c in cluster
            ],
            dtype=float
        )

        representative_x = weighted_average(
            xs,
            weights
        )

        best = max(
            cluster,
            key=lambda c:
                c["length"]
                * c["color_score"]
        )

        representatives.append(
            {
                "x": representative_x,

                "length": float(
                    np.sum(
                        [
                            c["length"]
                            for c in cluster
                        ]
                    )
                ),

                "line": best["line"],

                "color": best["color"],

                "color_score": float(
                    max(
                        c["color_score"]
                        for c in cluster
                    )
                ),

                "y_low": best["y_low"],

                "y_high": best["y_high"]
            }
        )

    return representatives


# ============================================================
# BUILD POSSIBLE LANE PAIRS
# ============================================================

def build_lane_pairs(representatives):

    yellow = [
        c
        for c in representatives
        if c["color"] == "yellow"
    ]

    white = [
        c
        for c in representatives
        if c["color"] == "white"
    ]

    pairs = []

    for y in yellow:

        for w in white:

            # ------------------------------------------------
            # Determine left/right boundary
            # ------------------------------------------------

            if w["x"] < y["x"]:

                left = w
                right = y

            else:

                left = y
                right = w

            lane_width = (
                right["x"]
                - left["x"]
            )

            # ------------------------------------------------
            # Lane width constraint
            # ------------------------------------------------

            if not (
                MIN_LANE_WIDTH
                <= lane_width
                <= MAX_LANE_WIDTH
            ):
                continue

            center = (
                left["x"]
                + right["x"]
            ) / 2.0

            pairs.append(
                {
                    "left": left,
                    "right": right,

                    "width": float(
                        lane_width
                    ),

                    "center": float(
                        center
                    )
                }
            )

    return pairs


# ============================================================
# SELECT BEST LANE PAIR
# ============================================================

def select_lane_pair(
    pairs,
    previous_state,
    image_width
):

    if not pairs:
        return None

    initial_center = (
        image_width
        * INITIAL_CENTER_FRACTION
    )

    scored = []

    for pair in pairs:

        left = pair["left"]
        right = pair["right"]

        center = pair["center"]
        width = pair["width"]

        color_strength = (
            left["color_score"]
            + right["color_score"]
        )

        score = 0.0

        # ====================================================
        # FIRST FRAME
        # ====================================================

        if previous_state is None:

            score += (
                abs(
                    center
                    - initial_center
                )
                / 120.0
            )

            score += (
                abs(
                    width
                    - INITIAL_WIDTH
                )
                / 300.0
            )

        # ====================================================
        # FOLLOWING FRAMES
        # ====================================================

        else:

            previous_left = (
                previous_state["left_x"]
            )

            previous_right = (
                previous_state["right_x"]
            )

            previous_width = (
                previous_state["width"]
            )

            score += (
                abs(
                    left["x"]
                    - previous_left
                )
                / 120.0
            )

            score += (
                abs(
                    right["x"]
                    - previous_right
                )
                / 120.0
            )

            score += (
                abs(
                    width
                    - previous_width
                )
                / 180.0
            )

        # ----------------------------------------------------
        # Prefer stronger road-marking evidence
        # ----------------------------------------------------

        score -= (
            0.5
            * color_strength
        )

        # ----------------------------------------------------
        # Reject obvious vehicle-hood detections
        #
        # Important for clip_13.
        # ----------------------------------------------------

        for boundary in (
            left,
            right
        ):

            x1, y1, x2, y2 = (
                boundary["line"]
            )

            if (
                min(y1, y2) > 420
                and 150 < boundary["x"] < 450
            ):

                score += 1.0

        pair["score"] = score

        scored.append(
            pair
        )

    return min(
        scored,
        key=lambda p: p["score"]
    )


# ============================================================
# GLOBAL TRACKING STATE
# ============================================================

_previous_state = None


# ============================================================
# REQUIRED e-YANTRA FUNCTION
# ============================================================

def detect_lane(frame):

    global _previous_state

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # 1. Detect Hough candidates
    # --------------------------------------------------------

    candidates = get_hough_candidates(
        frame
    )

    # --------------------------------------------------------
    # 2. Cluster duplicate detections
    # --------------------------------------------------------

    representatives = cluster_candidates(
        candidates
    )

    # --------------------------------------------------------
    # 3. Build yellow-white lane pairs
    # --------------------------------------------------------

    lane_pairs = build_lane_pairs(
        representatives
    )

    # --------------------------------------------------------
    # 4. Select lane pair
    # --------------------------------------------------------

    lane_pair = select_lane_pair(
        lane_pairs,
        _previous_state,
        width
    )

    center_x = -1
    lane = LANE_UNKNOWN

    # --------------------------------------------------------
    # 5. Calculate center
    # --------------------------------------------------------

    if lane_pair is not None:

        left_x = lane_pair["left"]["x"]
        right_x = lane_pair["right"]["x"]

        center_x = (
            left_x
            + right_x
        ) / 2.0

        center_x = int(
            round(center_x)
        )

        # ----------------------------------------------------
        # Determine lane
        #
        # Yellow LEFT + White RIGHT
        #       => vehicle is in LEFT lane
        #
        # White LEFT + Yellow RIGHT
        #       => vehicle is in RIGHT lane
        # ----------------------------------------------------

        left_color = (
            lane_pair["left"]["color"]
        )

        right_color = (
            lane_pair["right"]["color"]
        )

        if (
            left_color == "yellow"
            and right_color == "white"
        ):

            lane = LANE_LEFT

        elif (
            left_color == "white"
            and right_color == "yellow"
        ):

            lane = LANE_RIGHT

        else:

            lane = LANE_UNKNOWN

        # ----------------------------------------------------
        # Save tracking state
        # ----------------------------------------------------

        _previous_state = {
            "left_x": left_x,

            "right_x": right_x,

            "center_x": center_x,

            "width": (
                right_x
                - left_x
            )
        }

    # --------------------------------------------------------
    # EXACT evaluator output
    # --------------------------------------------------------

    return {
        "center_x": int(center_x),
        "lane": lane
    }

# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_lane() - the evaluation script only
# ever calls that one function. List them in the file header too.
# ------------------------------------------------------------------


##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_result(result, frame_index):
    '''
    Purpose:
    ---
    Check that detect_lane() returned the expected structure and normalise it,
    so that a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    `result` :          [ object ]      whatever detect_lane() returned
    `frame_index` :     [ int ]         index of the frame, used in error messages

    Returns:
    ---
    `clean` :           [ dict ]        {"center_x": int, "lane": str}
    '''
    where = "detect_lane() on frame {}".format(frame_index)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"center_x", "lane"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    center_x = result["center_x"]
    if isinstance(center_x, bool) or not isinstance(center_x, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned center_x of type {}, expected a number".format(
            where, type(center_x).__name__))
    center_x = int(round(float(center_x)))

    lane = result["lane"]
    if not isinstance(lane, str):
        raise TypeError("{} returned lane of type {}, expected a string".format(
            where, type(lane).__name__))
    lane = lane.strip().lower()
    if lane not in VALID_LANES:
        raise ValueError("{} returned lane = '{}', expected one of {}".format(
            where, result["lane"], ", ".join(VALID_LANES)))

    return {"center_x": center_x, "lane": lane}


def draw_overlay(frame, result):
    '''
    Purpose:
    ---
    Draw the detected lane centre and lane label on a copy of the frame.
    This is where display code belongs - never inside detect_lane().

    Input Arguments:
    ---
    `frame` :   [ numpy.ndarray ]   the frame that was passed to detect_lane()
    `result` :  [ dict ]            the validated result for that frame

    Returns:
    ---
    `canvas` :  [ numpy.ndarray ]   a copy of the frame with the overlay drawn
    '''
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # frame centre, for reference - roughly where the vehicle is pointing
    cv2.line(canvas, (width // 2, height), (width // 2, height - 40), (128, 128, 128), 1)

    center_x = result["center_x"]
    if 0 <= center_x < width:
        cv2.line(canvas, (center_x, height), (center_x, height // 2), (0, 0, 255), 2)
        cv2.circle(canvas, (center_x, height - 10), 5, (0, 0, 255), -1)

    label = "lane: {}   center_x: {}".format(result["lane"], center_x)
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


def process_video(video_path, show=False):
    '''
    Purpose:
    ---
    Read a video frame by frame, hand each frame to detect_lane() and collect
    the results.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video file
    `show` :        [ bool ]    if True, display the overlay while processing

    Returns:
    ---
    `results` :     [ list ]    one dict per frame:
                                {"frame": int, "center_x": int, "lane": str}
    '''
    if not os.path.isfile(video_path):
        raise FileNotFoundError("no such video file: {}".format(video_path))

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError("OpenCV could not open the video: {}".format(video_path))

    window = "Task 1C - {}".format(os.path.basename(video_path))
    results = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # A copy is passed in, so anything drawn inside detect_lane() cannot
            # corrupt the frame used for display.
            result = validate_result(detect_lane(frame.copy()), frame_index)
            results.append({"frame": frame_index, **result})

            if show:
                cv2.imshow(window, draw_overlay(frame, result))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_index += 1
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if not results:
        raise IOError("no frames could be read from: {}".format(video_path))

    return results


def summarise(video_path, results):
    '''
    Purpose:
    ---
    Print a one-line-per-video summary, so you can see at a glance whether the
    detector is returning anything sensible.

    Input Arguments:
    ---
    `video_path` :  [ str ]     path to the video that was processed
    `results` :     [ list ]    output of process_video()

    Returns:
    ---
    None
    '''
    total = len(results)
    counts = {lane: 0 for lane in VALID_LANES}
    for entry in results:
        counts[entry["lane"]] += 1
    not_found = sum(1 for entry in results if entry["center_x"] < 0)

    print("{:<28} {:>5} frames | left {:>5} | right {:>5} | unknown {:>5} | no centre {:>5}".format(
        os.path.basename(video_path), total,
        counts[LANE_LEFT], counts[LANE_RIGHT], counts[LANE_UNKNOWN], not_found))


def expand_videos(paths):
    '''
    Purpose:
    ---
    Turn the command-line arguments into a list of video files, accepting a
    FOLDER as well as individual files.

    A folder is the portable way to say "all the clips": Windows shells do not
    expand `public/*.mp4` the way bash does - cmd and PowerShell hand the
    pattern through verbatim and the script would look for a file literally
    named "*.mp4". `python lane_detection.py public` behaves the same on every
    platform.

    Input Arguments:
    ---
    `paths` :   [ list ]    the raw command-line arguments

    Returns:
    ---
    `videos` :  [ list ]    paths to individual video files, folders expanded
    '''
    videos = []
    for raw in paths:
        if os.path.isdir(raw):
            found = sorted(f for f in os.listdir(raw) if f.lower().endswith(".mp4"))
            if not found:
                raise FileNotFoundError("no .mp4 files in the folder: {}".format(raw))
            videos.extend(os.path.join(raw, f) for f in found)
        else:
            videos.append(raw)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Task 1C - run your lane detector over one or more videos.")
    parser.add_argument("videos", nargs="+",
                        help="video file(s), or a folder holding them "
                             "(e.g. 'public')")
    parser.add_argument("--show", action="store_true",
                        help="display the detection overlay while processing (press q to stop)")
    parser.add_argument("--out", metavar="FILE",
                        help="write the per-frame results to this JSON file")
    args = parser.parse_args()

    all_results = {}
    for video_path in expand_videos(args.videos):
        results = process_video(video_path, show=args.show)
        summarise(video_path, results)
        all_results[os.path.basename(video_path)] = results

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
        print("\nresults written to {}".format(args.out))


if __name__ == "__main__":
    main()
