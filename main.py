# Import necessary libraries
import numpy as np          # Used for mathematical operations such as mean, standard deviation, etc.
import time                 # Used for simulating real-time data with delays
from collections import deque # Used to store sliding window data for anomaly detection
import matplotlib.pyplot as plt # For real-time data visualization
import matplotlib.animation as animation # For animating the real-time plot

# Section 1: Data Stream Simulation
def data_stream_simulation(noise_level=0.5, amplitude=10, frequency=0.05, anomaly_probability=0.1):
    """
    Simulates a continuous data stream, generating values that follow a sinusoidal pattern with added noise.
    Random anomalies are introduced based on a specified probability.

    Parameters:
    - noise_level: The standard deviation of the normal noise added to the sinusoidal values.
    - amplitude: The maximum amplitude (height) of the sine wave.
    - frequency: Frequency of the sine wave (controls how fast the sine wave oscillates).
    - anomaly_probability: Probability of generating an anomalous value (between 0 and 1).

    Yields:
    - A floating-point number representing the next data point in the stream and its type.
    """
    t = 0  # Time counter (keeps track of the data points)
    
    while True:
        # Generate a sinusoidal pattern with random noise
        pattern = amplitude * np.sin(2 * np.pi * frequency * t)
        noise = np.random.normal(0, noise_level)  # Add random Gaussian noise to the pattern
        
        # Randomly decide whether to generate an anomaly
        if np.random.rand() < anomaly_probability:
            # Generate an anomalous value
            anomaly = pattern + noise + np.random.uniform(amplitude, amplitude * 2)  # Add a significant deviation
            yield anomaly, "anomaly"  # Return the anomalous value and type
        else:
            yield pattern + noise, "normal"  # Return the normal value and type
        
        t += 1  # Increment the time counter
        
        # Randomly decide the next delay time to simulate variable time intervals
        time.sleep(np.random.uniform(0.05, 0.3))  # Random delay between 0.05 to 0.3 seconds

# Section 2: Z-score Anomaly Detection
def z_score_anomaly_detection_optimized(stream, window_size=30, threshold=3):
    """
    Detects anomalies in the data stream using the Z-score method, optimized with error handling.

    Parameters:
    - stream: The data stream generator yielding data points.
    - window_size: Number of recent data points to use for calculating the mean and standard deviation (sliding window).
    - threshold: Z-score threshold above which a data point is considered anomalous.
    """
    window = deque(maxlen=window_size)  # Create a sliding window of fixed size to hold recent data points

    # Process each value from the stream
    for value, value_type in stream:
        try:
            # Wait until the window is filled with enough data points for analysis
            if len(window) < window_size:
                window.append(value)  # Append new value to the window
                continue  # Skip anomaly detection until the window is full

            # Calculate mean and standard deviation of the data in the window
            mean = np.mean(window)
            std_dev = np.std(window)

            # Check if the current value's Z-score exceeds the threshold
            if std_dev != 0 and abs((value - mean) / std_dev) > threshold:
                print(f"Anomaly detected! Value: {value}, Type: {value_type}")
            else:
                print(f"Normal value: {value}")

            # Update the sliding window by appending the new value
            window.append(value)

        except Exception as e:
            # Handle any potential errors (e.g., division by zero or issues with input data)
            print(f"Error processing stream: {e}")
            continue

# Section 3: Real-Time Visualization
def visualize_data_stream(stream, window_size=100, threshold=3):
    """
    Visualizes the data stream in real-time, highlighting anomalies on the graph.

    Parameters:
    - stream: The data stream generator that provides real-time data points.
    - window_size: Number of recent data points to display in the graph.
    - threshold: Z-score threshold used to detect anomalies.
    """
    data = []  # List to store recent data points for plotting
    anomalies = []  # List to store indices of detected anomalies

    # Set up the real-time plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2, label='Data Stream')  # Line plot for the data stream
    normal_dots, = ax.plot([], [], 'g.', label='Normal Values')  # Green dots for normal values
    anomaly_dots, = ax.plot([], [], 'ro', label='Anomalies')  # Red dots for anomalies

    # Initialize the plot
    def init():
        ax.set_ylim(-25, 25)  # Set the y-axis range (adjust based on your data)
        ax.set_xlim(0, window_size)  # Set the x-axis range (window size controls the number of data points shown)
        line.set_data([], [])  # Initialize the line plot
        normal_dots.set_data([], [])  # Initialize normal markers
        anomaly_dots.set_data([], [])  # Initialize anomaly markers
        return line, normal_dots, anomaly_dots

    # Update the plot with each new data point
    def update(frame):
        value, value_type = next(stream)  # Get the next value and its type from the data stream
        data.append(value)  # Add the value to the data list

        # Maintain a fixed window size for the plot
        if len(data) > window_size:
            data.pop(0)  # Remove the oldest value to maintain the window size

        # Anomaly detection using Z-score
        if len(data) >= 30:  # Ensure there are enough data points to calculate statistics
            mean = np.mean(data[-30:])  # Calculate the mean of the last 30 data points
            std_dev = np.std(data[-30:])  # Calculate the standard deviation
            # Detect anomaly if Z-score exceeds the threshold
            if std_dev != 0 and abs((value - mean) / std_dev) > threshold:
                anomalies.append(len(data) - 1)  # Mark the index of the anomaly

        # Update the line plot and markers
        line.set_data(range(len(data)), data)  # Update the line with the new data
        normal_indices = [i for i in range(len(data)) if i not in anomalies]  # Indices of normal values
        normal_dots.set_data(normal_indices, [data[i] for i in normal_indices])  # Update normal markers
        anomaly_dots.set_data(anomalies, [data[i] for i in anomalies])  # Update anomaly markers

        return line, normal_dots, anomaly_dots

    # Use matplotlib's animation functionality to update the plot continuously
    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=100)
    plt.legend()  # Show legend for the markers
    plt.show()  # Display the plot

# Example usage
if __name__ == "__main__":
    # Create a data stream generator
    stream = data_stream_simulation(anomaly_probability=0.1)
    
    # Visualize the data stream and detect anomalies
    visualize_data_stream(stream)
