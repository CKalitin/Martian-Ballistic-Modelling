import math
import re

# --- User-configurable variables ---

# Maximum pixel dimensions for normalization
max_x = 408  # e.g., width of the image in pixels
max_y = 448  # e.g., height of the image in pixels

# Real-world bounds for scaling
x_min = 0   # e.g., corresponds to pixel x = 0
x_max = 7500  # e.g., corresponds to pixel x = max_x

y_min = 0     # e.g., corresponds to pixel y = 0
y_max = 60000    # e.g., corresponds to pixel y = max_y

# Options
apply_log_scale = False          # True to transform outputs with natural log
sort_method = 'x'                # 'x' or 'distance'
ref_point = (0, 0)               # Reference for distance sorting


# Multi-line string of raw pixel points
points_str = '''
(0, 0)
(28, 18)
(21, 13)
(15, 9)
(4, 4)
(8, 6)
(29, 33)
(31, 49)
(33, 57)
(35, 66)
(37, 71)
(41, 74)
(46, 72)
(50, 67)
(56, 59)
(63, 50)
(70, 43)
(76, 38)
(82, 35)
(87, 34)
(93, 33)
(99, 34)
(108, 37)
(116, 41)
(126, 48)
(135, 55)
(143, 63)
(153, 74)
(163, 88)
(170, 99)
(177, 110)
(184, 120)
(192, 133)
(198, 142)
(204, 151)
(211, 161)
(216, 168)
(222, 177)
(229, 186)
(239, 199)
(248, 210)
(256, 219)
(264, 227)
(277, 239)
(286, 246)
(295, 252)
(306, 257)
(318, 261)
(330, 264)
(343, 267)
(353, 270)
(364, 275)
(374, 282)
(382, 290)
(390, 302)
(394, 312)
(397, 322)
(400, 334)
(402, 347)
(403, 354)
(404, 365)
(405, 377)
(406, 391)
(406, 409)
(407, 434)
(407, 447)
'''

# --- Core functions ---

def parse_points(s):
    """
    Parse a string of lines like '(x, y)' into a list of (float(x), float(y)).
    """
    pts = []
    for line in s.strip().splitlines():
        match = re.match(r"\s*\(\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\s*\)", line)
        if match:
            x, y = match.groups()
            pts.append((float(x), float(y)))
    return pts


def normalize_points(points, max_x, max_y):
    """
    Normalize raw pixel points to [0, 1] based on max_x and max_y.
    """
    return [(x / max_x, y / max_y) for x, y in points]


def scale_points(norm_points, x_min, x_max, y_min, y_max):
    """
    Scale normalized points into real-world units defined by bounds.
    """
    return [(
        x_min + x_n * (x_max - x_min),
        y_min + y_n * (y_max - y_min)
    ) for x_n, y_n in norm_points]


def sort_points(points, method, reference=(0, 0)):
    """
    Sort points by 'x' or by Euclidean distance from reference.
    """
    if method == 'x':
        return sorted(points, key=lambda p: p[0])
    elif method == 'distance':
        rx, ry = reference
        return sorted(points, key=lambda p: math.hypot(p[0] - rx, p[1] - ry))
    else:
        raise ValueError("sort_method must be 'x' or 'distance'.")


def apply_log(points):
    """
    Apply natural log to each coordinate; all values must be positive.
    """
    logged = []
    for x, y in points:
        if x <= 0 or y <= 0:
            raise ValueError(f"Cannot log non-positive value: ({x}, {y})")
        logged.append((math.log(x), math.log(y)))
    return logged

# --- Processing pipeline ---

def process_points():
    # 0. Parse string into raw_points
    raw_points = parse_points(points_str)

    # 1. Normalize
    normalized = normalize_points(raw_points, max_x, max_y)

    # 2. Scale to real-world
    real_points = scale_points(normalized, x_min, x_max, y_min, y_max)

    # 3. Optional log
    if apply_log_scale:
        real_points = apply_log(real_points)

    # 4. Sort
    sorted_points = sort_points(real_points, sort_method, reference=ref_point)

    return sorted_points

# --- Execute and print ---

if __name__ == '__main__':
    results = process_points()
    for x, y in results:
        #print(f"{x-545.39:.6f}, {y:.6f}") # Adjust for Perseverance entry time
        print(f"{x:.6f}, {y:.6f}")
