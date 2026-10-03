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
*  Last Modified:    02-October-2026
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
Y_START = 260
Y_END = 430

YELLOW_LOW = np.array([15, 70, 70], dtype=np.uint8)
YELLOW_HIGH = np.array([45, 255, 255], dtype=np.uint8)

WHITE_LOW = np.array([0, 0, 170], dtype=np.uint8)
WHITE_HIGH = np.array([180, 90, 255], dtype=np.uint8)

_previous_center = None
_previous_left = None
_previous_right = None

def get_marking_x(mask, x_min, x_max):

    region = mask[Y_START:Y_END, x_min:x_max]

    if region.size == 0:
        return None

    column_count = np.sum(region > 0, axis=0)

    if column_count.max() == 0:
        return None

    threshold = max(2, int(column_count.max() * 0.15))

    xs = np.where(column_count >= threshold)[0]

    if len(xs) == 0:
        return None

    weights = column_count[xs].astype(np.float32)

    x = np.average(
        xs + x_min,
        weights=weights
    )

    return float(x)

def get_markings(mask):
    """
    Find possible vertical/diagonal lane markings.

    Instead of Hough lines, divide the image into x regions
    and use the number of colored pixels.
    """

    region = mask[Y_START:Y_END, :]

    column_count = np.sum(region > 0, axis=0).astype(np.float32)

    if column_count.max() < 3:
        return []

    smooth = cv2.GaussianBlur(
        column_count.reshape(1, -1),
        (1, 15),
        0
    ).reshape(-1)

    threshold = max(
        3.0,
        smooth.max() * 0.15
    )

    active = smooth >= threshold

    markings = []

    start = None

    for i, value in enumerate(active):

        if value and start is None:
            start = i

        elif not value and start is not None:

            end = i

            if end - start >= 2:

                xs = np.arange(start, end)

                weights = smooth[start:end]

                if weights.sum() > 0:
                    center = np.average(
                        xs,
                        weights=weights
                    )
                else:
                    center = (start + end) / 2

                strength = float(
                    smooth[start:end].max()
                )

                markings.append(
                    (
                        float(center),
                        strength
                    )
                )

            start = None

    if start is not None:

        end = len(active)

        if end - start >= 2:

            xs = np.arange(start, end)
            weights = smooth[start:end]

            if weights.sum() > 0:
                center = np.average(
                    xs,
                    weights=weights
                )
            else:
                center = (start + end) / 2

            strength = float(
                smooth[start:end].max()
            )

            markings.append(
                (
                    float(center),
                    strength
                )
            )

    return markings

def find_lane_pair(yellow, white, width):

    if not yellow or not white:
        return None

    candidates = []

    MIN_WIDTH = 100
    MAX_WIDTH = 450

    for yx, ys in yellow:

        for wx, ws in white:

            distance = abs(wx - yx)

            if distance < MIN_WIDTH:
                continue

            if distance > MAX_WIDTH:
                continue

            left = min(yx, wx)
            right = max(yx, wx)

            center = (left + right) / 2.0

            center_error = abs(
                center - width * 0.5
            )

            strength = ys + ws

            score = (
                center_error / 120.0
                + distance / 500.0
                - strength / 100.0
            )

            candidates.append(
                (
                    score,
                    left,
                    right,
                    yx,
                    wx
                )
            )

    if not candidates:
        return None

    if _previous_center is not None:

        adjusted = []

        for item in candidates:

            score, left, right, yx, wx = item

            center = (left + right) / 2.0

            score += abs(
                center - _previous_center
            ) / 100.0

            adjusted.append(
                (
                    score,
                    left,
                    right,
                    yx,
                    wx
                )
            )

        candidates = adjusted

    return min(
        candidates,
        key=lambda x: x[0]
    )

def detect_lane(frame):

    global _previous_center
    global _previous_left
    global _previous_right

    if frame is None:
        return {
            "center_x": -1,
            "lane": LANE_UNKNOWN
        }

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    yellow_mask = cv2.inRange(
        hsv,
        YELLOW_LOW,
        YELLOW_HIGH
    )

    white_mask = cv2.inRange(
        hsv,
        WHITE_LOW,
        WHITE_HIGH
    )

    yellow_mask[:Y_START, :] = 0
    yellow_mask[Y_END:, :] = 0

    white_mask[:Y_START, :] = 0
    white_mask[Y_END:, :] = 0

    yellow = get_markings(
        yellow_mask
    )

    white = get_markings(
        white_mask
    )

    pair = find_lane_pair(
        yellow,
        white,
        frame.shape[1]
    )

    if pair is None:

        return {
            "center_x": -1,
            "lane": LANE_UNKNOWN
        }

    _, left, right, yellow_x, white_x = pair

    center = (
        left + right
    ) / 2.0

    center_x = int(
        round(center)
    )

    if yellow_x < white_x:

        lane = LANE_LEFT

    else:

        lane = LANE_RIGHT

    _previous_center = center
    _previous_left = left
    _previous_right = right

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
