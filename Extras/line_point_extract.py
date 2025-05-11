import cv2
import numpy as np
from collections import deque

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

# --- User-Configurable Variables ---
image_path = 'Perserverance Charts/perserverance-alt.jpg'   # Path to your input chart image
# Set your target HSV color (hue: 0-179, sat: 0-255, val: 0-255)
target_hsv = (int(240/2), int(100*2.55), int(0*2.55))
# HSV tolerance around target: (hue_range, sat_range, val_range)
hsv_range = (400, 400, 40)
# Number of evenly spaced points you want along the curve
num_points = 400

sort_by_corner  = True
# -----------------------------------

def extract_color_mask(image, target_hsv, hsv_range):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([max(0, target_hsv[0] - hsv_range[0]),
                      max(0, target_hsv[1] - hsv_range[1]),
                      max(0, target_hsv[2] - hsv_range[2])])
    upper = np.array([min(179, target_hsv[0] + hsv_range[0]),
                      min(255, target_hsv[1] + hsv_range[1]),
                      min(255, target_hsv[2] + hsv_range[2])])
    mask = cv2.inRange(hsv, lower, upper)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
    mask = cv2.dilate(mask, kernel3, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)

    return mask

def thin_mask(mask):
    if hasattr(cv2, 'ximgproc'):
        return cv2.ximgproc.thinning(mask)
    elif skeletonize is not None:
        bool_img = mask > 0
        thin = skeletonize(bool_img)
        return (thin.astype(np.uint8) * 255)
    else:
        raise RuntimeError("Install scikit-image or opencv-contrib for thinning")

def find_endpoints(skel):
    h, w = skel.shape
    ends = []
    for y in range(h):
        for x in range(w):
            if skel[y, x] == 0:
                continue
            nbr = 0
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dx == 0 and dy == 0:
                        continue
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < h and 0 <= xx < w and skel[yy,xx] > 0:
                        nbr += 1
            if nbr == 1:
                ends.append((x,y))
    return ends

def build_path(skel, start):
    visited = set([start])
    path = [start]
    current = start
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while True:
        found = None
        for dx, dy in neighbors:
            nx, ny = current[0]+dx, current[1]+dy
            if (nx,ny) in visited:
                continue
            if 0 <= nx < skel.shape[1] and 0 <= ny < skel.shape[0] and skel[ny, nx] > 0:
                found = (nx, ny)
                break
        if not found:
            break
        path.append(found)
        visited.add(found)
        current = found
    return path

def sort_along_skeleton(skel, points, start_pt):
    """
    skel : 2D binary (0/255) numpy array of your thinned line
    points: list of (x,y) sampled points
    start_pt: (x0,y0) coordinate to start BFS from (must lie on skel)
    Returns: points sorted by graph‐distance from start_pt
    """
    h, w = skel.shape
    # 1) BFS to build distance map
    D = -np.ones((h, w), dtype=int)
    dq = deque([start_pt])
    D[start_pt[1], start_pt[0]] = 0
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while dq:
        x,y = dq.popleft()
        for dx,dy in nbrs:
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h \
               and skel[ny, nx] > 0 \
               and D[ny, nx] < 0:
                D[ny, nx] = D[y, x] + 1
                dq.append((nx, ny))

    # 2) For each sample point, snap to nearest skel pixel within r
    #    (you can tune r to e.g. 2–5 pixels)
    sorted_pts = []
    for px, py in points:
        best_d = None
        best_pt = None
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                sx, sy = px+dx, py+dy
                if 0 <= sx < w and 0 <= sy < h and D[sy, sx] >= 0:
                    dval = D[sy, sx]
                    if best_d is None or dval < best_d:
                        best_d = dval
                        best_pt = (px, py)
        if best_d is None:
            # fallback: use Manhattan from start
            best_d = abs(px - start_pt[0]) + abs(py - start_pt[1])
        sorted_pts.append((best_d, (px, py)))

    # 3) sort by distance
    sorted_pts.sort(key=lambda x: x[0])
    return [pt for _, pt in sorted_pts]

def remove_duplicates(points):
    """
    Remove duplicate points from a list of (x,y) tuples.
    """
    seen = set()
    unique_points = []
    for pt in points:
        if pt not in seen:
            seen.add(pt)
            unique_points.append(pt)
    return unique_points

def sample_evenly(path, num_samples):
    dists = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        dists.append(dists[-1] + np.hypot(dx, dy))
    total = dists[-1]
    if total == 0 or num_samples < 2:
        return [path[0]] * num_samples

    pts = []
    for i in range(num_samples):
        target = total * i / (num_samples - 1)
        idx = np.searchsorted(dists, target)
        idx = max(1, min(idx, len(dists)-1))
        a, b = idx-1, idx
        seg_len = dists[b] - dists[a]
        t = 0.0 if seg_len == 0 else (target - dists[a]) / seg_len
        x = path[a][0] + t * (path[b][0] - path[a][0])
        y = path[a][1] + t * (path[b][1] - path[a][1])
        pts.append((int(round(x)), int(round(y))))
    return pts

def process(image_path, target_hsv, hsv_range, num_points):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    mask = extract_color_mask(img, target_hsv, hsv_range)
    skel = thin_mask(mask)

    ends = find_endpoints(skel)
    paths = [build_path(skel, e) for e in ends]

    def path_len(p):
        return sum(np.hypot(p[i][0]-p[i-1][0], p[i][1]-p[i-1][1]) for i in range(1, len(p)))
    lengths = [path_len(p) for p in paths]
    total_L = sum(lengths)

    all_samples = []
    for p, L in zip(paths, lengths):
        if L < 1.0:
            continue
        k = max(2, int(round(num_points * (L/total_L))))
        all_samples.extend(sample_evenly(p, k))

    if len(all_samples) > num_points:
        all_samples = all_samples[:num_points]
    else:
        all_samples.extend([all_samples[-1]] * (num_points - len(all_samples)))

    return mask, skel, all_samples

# Run processing with the above variables
mask, skel, even_points = process(image_path, target_hsv, hsv_range, num_points)
even_points = remove_duplicates(even_points)

cv2_visualize = False
if cv2_visualize:
    for (x, y) in even_points:
        cv2.circle(mask, (x, y), 2, (128, 128, 128), thickness=-1)
    cv2.imshow("Sampled Points", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Add this after you have `mask, even_points` and before your final print loop.
# --- Begin Interactive ROI Removal Block ---

# We’ll need to mutate these globals
selected_rois = []
current_roi   = []
drawing       = False

# Callback needs access to mask & even_points too
def mouse_cb(event, x, y, flags, param):
    global drawing, current_roi, selected_rois
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # initialize both corners to the click point
        current_roi = [(x, y), (x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # update the second corner
        current_roi[1] = (x, y)
        redraw()
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # finalize the second corner
        current_roi[1] = (x, y)
        # store the two‐point tuple
        selected_rois.append(tuple(current_roi))
        redraw()


def redraw():
    """Draw mask, points, and any ROIs on a window."""
    # mask is a gray image—convert to BGR so we can color things
    disp = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    # draw ALL current points
    for px, py in even_points:
        cv2.circle(disp, (px, py), 2, (0, 0, 255), -1)

    # draw completed ROIs in green
    for (x0, y0), (x1, y1) in selected_rois:
        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # draw the rectangle you’re currently dragging in blue
    if drawing and len(current_roi) == 2:
        (x0, y0), (x1, y1) = current_roi
        cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 0, 0), 1)

    cv2.imshow("Edit Points — draw box to remove", disp)

# Create window and set callback
cv2.namedWindow("Edit Points — draw box to remove")
cv2.setMouseCallback("Edit Points — draw box to remove", mouse_cb)

# Initial draw
redraw()

print("Draw boxes with the mouse to mark areas to remove. Press 'q' when done.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Filter out any points inside any ROI
def in_any_roi(pt):
    x, y = pt
    for (x0,y0),(x1,y1) in selected_rois:
        if min(x0,x1) <= x <= max(x0,x1) and min(y0,y1) <= y <= max(y0,y1):
            return True
    return False

even_points = [pt for pt in even_points if not in_any_roi(pt)]

# ——— Begin re‑ordering along the curve ———
ends     = find_endpoints(skel)
start_pt = max(ends, key=lambda p: p[1])
even_points = sort_along_skeleton(skel, even_points, start_pt)
# ——— End re‑ordering ———

# ——— Optional: sort by distance from bottom‑left ———
if sort_by_corner:
    h, w = skel.shape
    ref = (0, h-1)   # bottom-left corner
    even_points.sort(
        key=lambda pt: (pt[0] - ref[0])**2 + (pt[1] - ref[1])**2
    )
# ——— End corner sort ———

# Optional: visualize the newly ordered points
disp3 = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
for x, y in even_points:
    cv2.circle(disp3, (x, y), 2, (128, 128, 128), -1)
cv2.imshow("Ordered Points", disp3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finally, print your remaining coords
for x, y in even_points:
    print(f"{x}, {y}")