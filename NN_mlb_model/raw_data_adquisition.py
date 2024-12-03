import socket
import struct
import os
import datetime
import time
import json

def load_config(config_path="config.json"):
    """
    Load configuration from a JSON file.

    Parameters:
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required keys
    required_keys = [
        "ip_address", "streaming_port", "device_label",
        "base_dir", "session_duration_seconds", "session_delay_seconds"
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration key: '{key}'")
    
    return config

def flush_socket(sock, buffer_size, flush_time):
    """
    Flush the socket buffer by reading all pending data.
    Parameters:
        sock: The socket to flush.
        buffer_size: Size of the buffer to read in each call.
        flush_time: Time in seconds to allow flushing.
    """
    sock.settimeout(0.1)  # Set a short timeout to avoid blocking indefinitely
    start_flush = time.perf_counter()
    try:
        while time.perf_counter() - start_flush < flush_time:
            data = sock.recv(buffer_size)
            if not data:
                break
    except socket.timeout:
        # Timeout means the buffer is cleared
        pass
    finally:
        sock.settimeout(None)  # Reset to default blocking mode

def streaming_to_binary(config):
    """
    Stream data from the Red Pitaya and save it to binary files.

    Parameters:
        config (dict): Configuration dictionary loaded from JSON.
    """
    # Extract configurations
    ip_address = config["ip_address"]
    streaming_port = config["streaming_port"]
    device_label = config["device_label"]
    base_dir = config["base_dir"]
    session_duration = config["session_duration_seconds"]
    session_delay = config["session_delay_seconds"]
    buffer_size = 16384  # Default buffer size
    bytes_per_sample = 2  # Each sample is 2 bytes (16-bit ADC)

    # Organize output directory
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    device_dir = os.path.join(base_dir, device_label, current_date)
    os.makedirs(device_dir, exist_ok=True)

    # Determine initial session index
    session_index = len([f for f in os.listdir(device_dir) if f.startswith("session_")]) + 1

    try:
        # Connect to Red Pitaya streaming server
        print(f"Connecting to Red Pitaya at {ip_address}:{streaming_port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip_address, streaming_port))
        print("Connection successful!")

        while True:
            # Adjust flush time to match the delay
            print("Flushing socket buffer...")
            flush_socket(sock, buffer_size, flush_time=session_delay - session_duration)

            # Create binary file for current session
            session_filepath = os.path.join(device_dir, f"session_{session_index:03d}.bin")
            print(f"Streaming data for {session_duration} seconds for session {session_index}...")

            chunk_size = buffer_size * bytes_per_sample * 2  # *2 for two channels
            start_time = time.perf_counter()  # Use high-resolution timer
            with open(session_filepath, 'wb') as f:
                while time.perf_counter() - start_time < session_duration:
                    f.write(sock.recv(chunk_size))  # Write raw binary data directly to the file

            print(f"Session {session_index} data saved to {session_filepath}")

            # Increment session index for next iteration
            session_index += 1

            # Sleep before next iteration
            print(f"Sleeping for {session_delay} seconds...")
            time.sleep(session_delay)

    except KeyboardInterrupt:
        print("\nData collection stopped by user.")
    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        sock.close()
        print("Socket closed.")