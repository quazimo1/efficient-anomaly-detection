import numpy as np
import random
import time
import threading
from collections import deque

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
        window_size (int): The size of the moving window.
        threshold (float): The threshold value for considering a 
            data point to be an anomaly.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = deque(maxlen=window_size)
    
    def modified_z_score(self, data_point):
        """
        Calculate the Modified Z-Score for the given data point.

        The Modified Z-Score is a measure of how many standard deviations
        an element is from the median. It is more robust to outliers than
        the standard Z-Score.

        :param data_point: The data point to calculate the Modified Z-Score for.
        :return: The Modified Z-Score for the given data point.
        """
        if len(self.data_window) < 2:
            return 0  # Not enough data points to calculate the Modified Z-Score
        
        median = np.median(self.data_window)
        mad = np.median([abs(x - median) for x in self.data_window])  # Median Absolute Deviation
        
        if mad == 0:  # Prevent division by zero
            return 0
        
        return 0.6745 * (data_point - median) / mad
    
    def detect_anomaly(self, data_point):
        """
        Detect whether the given data point is an anomaly or not.

        This function uses the Modified Z-Score method to detect anomalies.

        Parameters:
        data_pāoint (float): The data point to check for anomalies.

        Returns:
        tuple: A tuple containing a boolean indicating whether the data point
            is an anomaly and the Modified Z-Score of the data point.
        """
        m_z = self.modified_z_score(data_point)
        self.data_window.append(data_point)
        if abs(m_z) > self.threshold:
            # If the Modified Z-Score is greater than the threshold, then it
            # is an anomaly.
            return True, m_z
        # If the Modified Z-Score is less than or equal to the threshold, then
        # it is not an anomaly.
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
        
def wait_for_enter(stop_event):
    """
    Wait for the user to press Enter to stop the anomaly detection.

    This function is used to wait for the user to press Enter to stop the anomaly detection.
    It is used in the main function to stop the generation of the data stream.

    Parameters:
    stop_event (threading.Event): An event that is set when the user presses Enter.
    """
    input("Press Enter to stop...")
    stop_event.set()


def main():
    """
    The main function of the anomaly detection program.

    This function sets up the anomaly detection and starts a thread to wait for
    the user to press Enter to stop the anomaly detection. It then prints out the
    data points and Modified Z-Score as they are generated, and detects anomalies
    using the Modified Z-Score method. The program exits when the user presses Enter.
    """
    stop_event = threading.Event()
    
    detector = AnomalyDetection(window_size=50, threshold=3)
    data_stream = generate_data_stream(stop_event)

    thread = threading.Thread(target=wait_for_enter, args=(stop_event,))
    thread.start()

    for data_point in data_stream:
        is_anomaly, modified_z_score = detector.detect_anomaly(data_point)
        print(f"Data: {data_point:.2f}, Modified-Z-Score: {modified_z_score:.2f}, Anomaly: {is_anomaly}")
        time.sleep(0.1)  # Simulating real-time delay

    thread.join()
    print("Exiting...")

if __name__ == "__main__":
    main()