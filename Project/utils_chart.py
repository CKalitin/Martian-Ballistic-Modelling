import matplotlib.pyplot as plt
import math

import utils_data

def sub_plot(position, title, x_label, y_label, x_data, y_data_list, series_labels, comparisons=[], comparison_x_key=None, comparison_y_keys=None, comparison_label_field='label', equal_aspect = False):
    """
    Creates a subplot and plots multiple data series, with optional comparison series.
        position (tuple): A tuple (nrows, ncols, index) specifying the subplot position.
        title (str): The title of the subplot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        x_data (array-like): The data for the x-axis, shared by all main series.
        y_data_list (list of array-like): List of y-axis data arrays, one for each main series.
        series_labels (list of str): List of labels for each main series.
        comparisons (list of dict, optional): List of comparison data dictionaries. Each dictionary should contain keys for x and y data and label.
        comparison_x_key (list of str, optional): List of keys to extract x-axis data from each comparison dictionary.
        comparison_y_keys (list of str, optional): List of keys to extract y-axis data from each comparison dictionary.
        comparison_label_field (str, optional): The key in the comparison dictionaries to use for the label. Defaults to 'label'.
        equal_aspect (bool, optional): Whether to set equal aspect ratio for the subplot. Defaults to False.
    Returns:
        None: The function creates a subplot and plots the provided data.
    Notes:
        - Adds a legend if more than one series is plotted.
        - Comparison series are plotted with dashed lines.
        - array-like means any structure that can be converted to a numpy array, such as lists or numpy arrays.
    """

    zorder = 999
    
    plt.subplot(position[0], position[1], position[2])

    for y_data, label in zip(y_data_list, series_labels):
        plt.plot(x_data, y_data, label=label, zorder=zorder)
        zorder -= 1
        
    for comparison in comparisons:
        for x_key, y_key in zip(comparison_x_key, comparison_y_keys):
            if x_key in comparison and y_key in comparison:
                label_field = comparison_label_field if comparison_label_field in comparison else 'label' # I FUCKING LOVE PYTHON
                line_style = (5, (10, 3)) if y_key == "body_points_y" else '--' # Hardcode Mars line style, see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
                plt.plot(comparison[x_key], comparison[y_key], linestyle=line_style, label=comparison[label_field], zorder=zorder)
                zorder -= 1
                
    if equal_aspect:
        x_range = max(x_data) - min(x_data) # x range of the data
        y_range = max(y_data_list[0]) - min(y_data_list[0])
        fig_aspect = plt.gcf().get_size_inches()[0] / position[0] / (plt.gcf().get_size_inches()[1] / position[1]) # On screen x to y pixel ratio
        if x_range / y_range > fig_aspect: # If the x range is larger than the y range, adjust y limits
            y_center = sum(y_data_list[0]) / len(y_data_list[0])
            plt.ylim(y_center - x_range / fig_aspect / 2, y_center + x_range / fig_aspect / 2)
        else: # If the y range is larger than the x range, adjust x limits
            x_center = sum(x_data) / len(x_data)
            plt.xlim(x_center - y_range * fig_aspect / 2, x_center + y_range * fig_aspect / 2)
        plt.gca().set_aspect('equal', adjustable='datalim')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if len(y_data_list) + len(comparisons) > 1: plt.legend()
    plt.grid(True)

def sub_plot_atmosphere(position, data):
    plt.subplot(position[0], position[1], position[2])
    ax5_1 = plt.gca()
    ax5_1.plot(data.t, data.drag_coeff, 'b-', label='Drag Coefficient')
    ax5_1.set_title('Drag Coefficient, Atmospheric Pressure, Temperature, Density vs Time')
    ax5_1.set_ylabel('Drag Coefficient', color='b')
    ax5_1.tick_params(axis='y', labelcolor='b')
    ax5_1.grid(True)

    ax5_2 = ax5_1.twinx()
    ax5_2.plot(data.t, data.atm_p, 'g-', label='Atmospheric Pressure')
    ax5_2.set_ylabel('Pressure (Pa)', color='g')
    ax5_2.tick_params(axis='y', labelcolor='g')

    # chart atmospheric temperature on the left y-axis
    ax5_3 = ax5_1.twinx()
    ax5_3.spines['left'].set_position(('outward', 55))  # Move the right spine outward
    ax5_3.yaxis.tick_left()
    ax5_3.yaxis.set_label_position('left')
    ax5_3.spines['left'].set_visible(True)
    ax5_3.plot(data.t, data.atm_t, 'r-', label='Atmospheric Temperature')
    ax5_3.set_ylabel('Temperature (K)', color='r')
    ax5_3.tick_params(axis='y', labelcolor='r')
    
    # Add legends for each y-axis
    lines, labels = ax5_1.get_legend_handles_labels()
    lines2, labels2 = ax5_2.get_legend_handles_labels()
    lines3, labels3 = ax5_3.get_legend_handles_labels()
    ax5_1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')
    ax5_1.set_xlabel('Time (s)')

    # Plot 3,3: Parameters
    plt.subplot(3, 3, 9)
    plt.axis('off')

def sub_plot_text(position, parameters, data):
    """
    Creates a subplot and plots multiple data series with text annotations.
    """
    
    plt.subplot(position[0], position[1], position[2])
    plt.axis('off')

    parameters_text = [
        (0, f"Parameters:", 'bold'),
        (0, f"Mass: {parameters['mass']} kg", None),
        (0, f"Area: {parameters['area']} m²", None),
        (0, f"Ballistic Coefficient: {round(parameters['ballistic_coefficient'])} kg/m²", None),
        (1, f"Entry Altitude: {parameters['entry_altitude']} m", None),
        (1, f"Entry Flight Path Angle: {parameters['entry_flight_path_angle']}°", None),
        (1, f"Entry Velocity: {parameters['entry_velocity']} m/s", None),
        (2, f"Time Step: {parameters['time_step']} s", None),
        (2, f"Max Time: {parameters['time_max']} s", None),
    ]
    
    if 'aoa' in parameters:
        parameters_text.append((3, f"Constant AoA: {parameters['aoa']}°", None))

    text_box(-0.10, 1.0, parameters_text)
    
    impact_values_text = [
        (0, f"Impact Values:", 'bold'),
        (0, f"Final Altitude: {data.alt[-1]:.2f} m", None),
        (0, f"Final Downrange Distance: {(data.ang_dist_rad[-1] * utils_data.MARS_RADIUS):.2f} m", None),
        (0, f"Final Angular Displacement: {math.degrees(data.ang_dist_rad[-1]):.2f}°", None),
        (1, f"Final Velocity: {data.v_net[-1]:.2f} m/s", None),
        (1, f"Final Horizontal Velocity: {data.v_ang[-1]:.2f} m/s", None),
        (1, f"Final Vertical Velocity: {data.v_rad[-1]:.2f} m/s", None),
        (2, f"Final Acceleration: {data.a_net[-1]:.2f} m/s²", None),
        (2, f"Final Horizontal Acceleration: {data.a_ang[-1]:.2f} m/s²", None),
        (2, f"Final Vertical Acceleration: {data.a_rad[-1]:.2f} m/s²", None),
        (2, f"Final Drag Acceleration: {data.a_drag[-1]:.2f} m/s²", None),
        (2, f"Final Gravity Acceleration: {data.a_grav[-1]:.2f} m/s²", None),
        (3, f"Final Flight Path Angle: {data.fpa[-1]:.2f}°", None),
        (3, f"Final Time: {data.t[-1]:.2f} s", None),
        (4, f"Execution Time: {data.execution_time[-1]:.2f} s", None)
    ]
    
    text_box(0.4, 1.0, impact_values_text)

def text_box(x, y, text_lines):
    # text_lines: array of tuples (group index, text, fontweight)
    for i, (group_index, text, fontweight) in enumerate(text_lines): # Enumerate adds a loop iteration number, Python! Man!
        plt.text(x, y - (i * 0.07) - (group_index * 0.03), text, fontsize=10, fontweight=fontweight) # 0.03 so it adds to 0.1

def remove_comparison_body_points_out_of_range(comparison, data):
    """
    Removes points from comparison['body_points_x'] and comparison['body_points_y'] that are out of the specified range.
    The lists are modified in place.
    
    Find the farthest points (in +/- x & y), if any body points are out of range, remove them
    If a point in the x list is out of range, remove the corresponding point in the y list
    """
    
    margin = 50 # km
    
    x_min = min(data.global_cartesian_pos_x) - margin
    x_max = max(data.global_cartesian_pos_x) + margin
    y_min = min(data.global_cartesian_pos_y) - margin
    y_max = max(data.global_cartesian_pos_y) + margin
    
    x_points = comparison['body_points_x']
    y_points = comparison['body_points_y']
    filtered_x = []
    filtered_y = []
    for x, y in zip(x_points, y_points):
        if x_min <= x <= x_max and y_min <= y <= y_max:
            filtered_x.append(x)
            filtered_y.append(y)
    comparison['body_points_x'] = filtered_x
    comparison['body_points_y'] = filtered_y

def add_and_trim_comparison_body_points(comparisons, data):
    """
    Adds body (Mars, currently) points to the comparison dictionary from the data object.
    Also trim the points in the body points and flight path points of comparisons (Currently only exist for Perseverance).
    """
    
    for comparison in comparisons:
        if 'body_points_x' in comparison and 'body_points_y' in comparison:
            remove_comparison_body_points_out_of_range(comparison, data)
    
    # Add Mars body for global cartesian position chart
    comparisons.insert(0, {'body_points_x': utils_data.mars_circumference_points_km_x, 'body_points_y': utils_data.mars_circumference_points_km_y, 'label': 'Mars Surface'})
    remove_comparison_body_points_out_of_range(comparisons[0], data)

    # remove last 3 from x and y, yea im just hardcoding this in, it makes it loop back and puts an ugly line across the graph
    comparisons[0]['body_points_x'] = comparisons[0]['body_points_x'][:-3]
    comparisons[0]['body_points_y'] = comparisons[0]['body_points_y'][:-3]

