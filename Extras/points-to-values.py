import math
import re

# --- User-configurable variables ---

# Maximum pixel dimensions for normalization
max_x = 618  # e.g., width of the image in pixels
max_y = 425  # e.g., height of the image in pixels

# Real-world bounds for scaling
x_min = 0   # e.g., corresponds to pixel x = 0
x_max = 1000  # e.g., corresponds to pixel x = max_x

y_min = -90     # e.g., corresponds to pixel y = 0
y_max = 0    # e.g., corresponds to pixel y = max_y

# Options
apply_log_scale = False          # True to transform outputs with natural log
sort_method = 'x'                # 'x' or 'distance'
ref_point = (0, 0)               # Reference for distance sorting


# Multi-line string of raw pixel points
points_str = '''
(0, 235)
(37, 246)
(69, 255)
(87, 260)
(114, 268)
(143, 277)
(165, 284)
(195, 294)
(231, 307)
(256, 316)
(298, 333)
(337, 349)
(360, 359)
(382, 372)
(368, 363)
(387, 380)
(391, 389)
(395, 398)
(398, 405)
(401, 411)
(406, 416)
(411, 420)
(417, 423)
(425, 424)
(435, 423)
(445, 417)
(454, 407)
(460, 398)
(468, 385)
(476, 368)
(482, 346)
(485, 338)
(486, 327)
(489, 319)
(492, 294)
(491, 306)
(496, 275)
(498, 268)
(494, 283)
(500, 256)
(502, 236)
(504, 212)
(506, 191)
(508, 172)
(511, 140)
(514, 113)
(515, 105)
(516, 98)
(519, 90)
(522, 82)
(525, 78)
(529, 81)
(532, 77)
(536, 73)
(538, 63)
(540, 53)
(543, 42)
(545, 40)
(549, 43)
(555, 37)
(552, 41)
(557, 40)
(558, 57)
(559, 73)
(561, 119)
(563, 152)
(564, 171)
(566, 175)
(568, 166)
(569, 138)
(570, 117)
(571, 88)
(572, 49)
(573, 17)
(574, 0)
(580, 0)
(581, 32)
(585, 5)
(587, 44)
(591, 68)
(592, 126)
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
        print(f"{x-545.39:.6f}, {y:.6f}")
