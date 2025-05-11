import numpy as np

def calculate_distances(times, hvels, keep_index=0, time_step=0.1):
    """
    Calculate distance traveled at each time step by interpolating velocities and return timestamps.
    
    Parameters:
    times (np.array): Array of time points
    hvels (np.array): Array of horizontal velocities
    drop_tenth (bool): If True, drop every 10th value in the output
    
    Returns:
    tuple: (np.array of timestamps, np.array of distances traveled at each time step)
    """
    
    time_steps = np.arange(times[0], times[-1], time_step)
    
    # Interpolate velocities at time points
    interpolated_vels = np.interp(time_steps, times, hvels)
    
    distances = np.zeros(len(time_steps))
    distances_sum = 0
    for i in range(1, len(time_steps)):
        # Calculate distance as velocity * time_step
        distances[i] = distances_sum + interpolated_vels[i] * time_step
        distances_sum += interpolated_vels[i] * time_step
    
    if keep_index != 0:
        mask = np.arange(len(distances)) % keep_index == 0
        distances = distances[mask]
        time_steps = time_steps[mask]
    
    return time_steps, distances

# Example usage with provided data
times = np.array([
    0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 
    99, 104, 109, 114, 119, 124, 129, 134, 139, 144, 149, 154, 159, 164, 169, 174, 
    179, 184, 189, 194, 199, 204, 209, 214, 219, 224, 229, 234, 239, 244, 249, 254, 
    259, 264, 269, 274, 279, 284, 289, 294, 299, 304, 309, 314, 319, 324, 329, 334, 
    339, 344, 349, 354, 359, 364, 369, 374, 379, 384, 389, 394, 399, 404, 409, 414
])

hvels = np.array([
    5127, 5138, 5149, 5161, 5173, 5182, 5190, 5198, 5199, 5190, 5182, 5137, 5071, 
    4926, 4692, 4301, 3827, 3414, 2939, 2561, 2178, 1932, 1732, 1534, 1381, 1267, 
    1164, 1077, 1002, 940, 884, 839, 794, 753, 715, 678, 649, 620, 591, 570, 551, 
    533, 514, 495, 475, 455, 436, 418, 314, 218, 174, 141, 121, 101, 82, 65, 49, 
    35, 30, 28, 26, 26, 27, 25, 23, 21, 13, 5, 0, 3, 5, 8, 13, 31, 36, 32, 13, 
    5, 7, 10, 12, 15, 17, 19
])

# Calculate distances and timestamps
timestamps, distances = calculate_distances(times, hvels, keep_index=500, time_step=0.01)

# Print results for verification
print(len(timestamps))
print(len(distances))

for i in range(len(timestamps)):
    print(f"{timestamps[i]:.6f}\t{distances[i]:.6f}")