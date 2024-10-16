import numpy as np
import random
import time
import threading
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class AnomalyDetection:
    """
    A class for detecting anomalies in a stream of data using a 
    combination of statistical methods.
    
    Parameters:
    window_size (int): The number of data points to keep in the 
        moving window for calculating the mean and standard 
        deviation.
    threshold (float): The threshold value for considering a data 
        point to be an anomaly.
    """
    def __init__(self, window_size=50, threshold=3):
        """
        Initialize the anomaly detector.
        
        Parameters:
        window_size (int): The number of data points to keep in the 
            moving window for calculating the mean and standard 
            deviation.
        threshold (float): The threshold value for considering a data 
            point to be an anomaly. The Modified Z-Score of the data 
            point must be greater than the threshold to be considered 
            an anomaly.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = deque(maxlen=window_size)  # Store the last 'window_size' data points
    
    def modified_z_score(self, data_point):
        """
        Calculate the Modified Z-Score for the given data point.

        The Modified Z-Score is a measure of how many standard deviations
        an element is from the median. It is more robust to outliers than
        the standard Z-Score.

        Parameters:
        data_point (float): The data point to calculate the Modified Z-Score for.

        Returns:
        float: The Modified Z-Score for the given data point.
        """
        if len(self.data_window) < 2:
            return 0  # Not enough data points to calculate the Modified Z-Score
        
        # Calculate the median of the data points in the window
        median = np.median(self.data_window)
        
        # Calculate the Median Absolute Deviation of the data points in the window
        mad = np.median([abs(x - median) for x in self.data_window])
        
        # Prevent division by zero
        if mad == 0:
            return 0
        
        # Calculate the Modified Z-Score
        return 0.6745 * (data_point - median) / mad
    
    def detect_anomaly(self, data_point):
        """
        Detect whether the given data point is an anomaly or not.

        This function uses the Modified Z-Score method to detect anomalies.

        Parameters:
        data_point (float): The data point to check for anomalies.

        Returns:
        tuple: A tuple containing a boolean indicating whether the data point
            is an anomaly and the Modified Z-Score of the data point.
        """
        # Calculate the Modified Z-Score for the given data point
        m_z = self.modified_z_score(data_point)
        
        # Add the data point to the window
        self.data_window.append(data_point)
        
        # Check if the Modified Z-Score is greater than the threshold
        if abs(m_z) > self.threshold:
            # If it is, return a tuple indicating that it is an anomaly
            # and the Modified Z-Score
            return True, m_z
        # If not, return a tuple indicating that it is not an anomaly
        # and the Modified Z-Score
        return False, m_z


def generate_data_stream(stop_event):
    """
    Generates a stream of data points that can be used for anomaly detection.

    This function generates a stream of data points that are based on the current
    second of the day. The data points are scaled between 0.01 and 1000 and have
    a moderate drift (0.02) and high volatility (0.3) to simulate financial data.
    The data points also have occasional "jumps" to simulate spikes in the data.

    The data points are generated in a loop until the stop_event is set.

    :param stop_event: An event that can be set to stop the generation of data points.
    :yield: A stream of data points.
    """
    current_second = time.localtime().tm_sec  # Get the current second
    start_value = (current_second / 59) * (1000 - 0.01) + 0.01  # Scale between 0.01 and 1000
    mu = 0.02  # A moderate drift for long-term growth
    sigma = 0.3  # Higher volatility to simulate financial data
    dt = 1  # Time step
    current_value = start_value
    
    while not stop_event.is_set():
        Z = np.random.normal(0, 1)
        # Introduce occasional "jumps" for spikes in the data
        jump = np.random.choice([0, np.random.normal(20, 5)], p=[0.9, 0.1])  # 10% chance of a jump
        
        # Update the value with a drift, noise, and possible jump
        current_value = current_value * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z) + jump
        
        yield current_value

def update_plot(frame, detector, data_stream, data, anomalies, line, scatter):
    """
    Update the plot with new data from the stream and detect anomalies.

    This function is used to update the plot with new data points from the stream,
    detect anomalies using the Modified Z-Score method and update the scatter plot
    with the anomaly points.

    Parameters:
    frame (int): The frame number, used to determine the x-axis limits.
    detector (AnomalyDetection): The anomaly detector object.
    data_stream (generator): A generator of data points from the stream.
    data (list): The list of data points.
    anomalies (list): The list of anomaly points (index and value).
    line (matplotlib.lines.Line2D): The line object representing the normal data.
    scatter (matplotlib.collections.PathCollection): The scatter plot object representing the anomaly points.

    Returns:
    tuple: A tuple of the updated line and scatter objects.
    """
    # Get new data point from the stream
    data_point = next(data_stream)
    
    # Check for anomalies
    is_anomaly, m_z_score = detector.detect_anomaly(data_point)
    
    # Update data buffer
    data.append(data_point)
    
    # Update the line plot
    line.set_data(range(len(data)), data)
    
    # If anomaly detected, store its index and value
    if is_anomaly:
        anomalies.append((len(data) - 1, data_point))
    
    # Update scatter plot with anomaly points
    if anomalies:
        anomaly_indices, anomaly_values = zip(*anomalies)  # Unzip the list of tuples
        scatter.set_offsets(np.array([anomaly_indices, anomaly_values]).T)  # Update scatter points

    # Set dynamic limits for y-axis based on the current data
    if data:
        # Dynamically adjust y-limits to encompass all data points
        ax.set_ylim(min(data) * 0.95, max(data) * 1.05)  # 5% padding

    # Adjust the x-axis limits dynamically based on data length
    ax.set_xlim(0, len(data))  # Set x limits from 0 to the current number of data points

    # Set x-ticks at regular intervals for better readability
    if len(data) > 0:
        # Set x-ticks to show indices of data points
        tick_spacing = max(1, len(data) // 10)  # Show up to 10 ticks
        ax.set_xticks(np.arange(0, len(data), tick_spacing))
        ax.set_xticklabels(np.arange(0, len(data), tick_spacing), rotation=45)

    # Set the x-axis label to indicate the time or sequence of the data
    ax.set_xlabel('Index of Data Points')
    
    return line, scatter
        
def wait_for_enter(stop_event):
    """
    Wait for the user to press Enter to stop the anomaly detection.

    This function is used to wait for the user to press Enter to stop the anomaly
    detection. It is used in the main function to stop the generation of the data
    stream.
    """
    input("Press Enter to stop...")
    stop_event.set()


def main():
    stop_event = threading.Event()
    
    detector = AnomalyDetection(window_size=50, threshold=3)
    data_stream = generate_data_stream(stop_event)

    thread = threading.Thread(target=wait_for_enter, args=(stop_event,))
    thread.start()

    # Set up the plot
    global ax
    fig, ax = plt.subplots()
    ax.set_title("Real-Time Data Stream with Anomalies")
    ax.set_xlabel("Time")
    ax.set_ylabel("Data Value")
    
    # Plot buffers
    data = deque(maxlen=100)  # Stores the last 100 data points
    anomalies = []  # Stores (index, value) of anomaly points
    
    # Line plot for data
    line, = ax.plot([], [], lw=2)
    
    # Scatter plot for anomalies
    scatter = ax.scatter([], [], color='red', label='Anomalies')
    
    # Animation function to update the plot
    ani = FuncAnimation(fig, update_plot, fargs=(detector, data_stream, data, anomalies, line, scatter), 
                        interval=100, blit=True)
    
    # Display the legend for anomalies
    ax.legend(loc='upper left')
    
    # Start the animation
    plt.show()

    thread.join()
    print("Exiting...")

if __name__ == "__main__":
    main()