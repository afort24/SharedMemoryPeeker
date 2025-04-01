import numpy as np
import posix_ipc
import struct
import mmap
import time
import signal
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView, QLabel,
                             QPushButton, QHBoxLayout, QSpinBox, QLabel)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

# Constants (must match your JUCE setup)
SHM_NAME = "/my_shared_audio_buffer"
FLOAT_SIZE = 4
RING_BUFFER_SIZE = 8192
MAX_CHANNELS = 8
MAX_BUFFER_SIZE = 1024 * 1024  # 1 MB

# Global flag to handle clean exit
keep_running = True


def signal_handler(sig, frame):
    global keep_running
    print("\nTerminating gracefully...")
    keep_running = False
    QApplication.quit()


# Attach signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


class AudioDataReader(QThread):
    """Thread for reading audio data from shared memory"""
    dataReady = pyqtSignal(np.ndarray, int, int)  # Data, write index, channels
    memoryDump = pyqtSignal(list)  # For hex dump display

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.debug_offset = 0  # Starting memory offset for debugging

        # Shared memory setup
        try:
            self.shm = posix_ipc.SharedMemory(SHM_NAME, flags=0)
            self.map_file = mmap.mmap(self.shm.fd, MAX_BUFFER_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
            self.shm.close_fd()
            print("Successfully connected to shared memory")

            # Parse the shared memory header based on the hex dump
            self.parse_memory_header()

        except Exception as e:
            print(f"Failed to access shared memory: {e}")
            sys.exit(1)

    def parse_memory_header(self):
        """Parse header information from the memory dump"""
        # Read the first 64 bytes of memory
        self.map_file.seek(0)
        header = self.map_file.read(64)

        # Analyze and print what we find
        print("\nParsing shared memory header:")

        # First 8 bytes appear to be the write index (or related)
        self.write_index_offset = 0
        write_index = struct.unpack("Q", header[0:8])[0]
        print(f"Offset 0: Value {write_index} (possible write index)")

        # Bytes 16-20 seem to have value 1
        value_at_16 = struct.unpack("i", header[16:20])[0]
        print(f"Offset 16: Value {value_at_16}")

        # Bytes 20-24 seem to have value 2 - could be num_channels
        self.num_channels_offset = 20
        num_channels = struct.unpack("i", header[20:24])[0]
        print(f"Offset 20: Value {num_channels} (possible channel count)")

        # Try to determine where the audio data starts
        # From your hex dump, after the first 64 bytes, it's mostly zeros
        self.audio_data_offset = 64
        print(f"Assuming audio data starts at offset {self.audio_data_offset}")

        # Send a full hex dump for display
        self.update_memory_dump()

    def update_memory_dump(self):
        """Read a section of memory for hex dump display"""
        self.map_file.seek(self.debug_offset)
        memory_block = self.map_file.read(256)  # Read 256 bytes
        hex_values = [f"{b:02x}" for b in memory_block]

        # Find actual number of channels
        self.map_file.seek(self.num_channels_offset)
        self.num_channels = struct.unpack("i", self.map_file.read(4))[0]

        # Send the dump to the UI
        self.memoryDump.emit(hex_values)

    def run(self):
        print("AudioDataReader thread started")

        while self.running and keep_running:
            try:
                # Update memory dump occasionally
                if np.random.random() < 0.2:  # 20% chance each cycle
                    self.update_memory_dump()

                # Read the raw data
                data, write_index, num_channels = self.read_audio_data_raw()
                if data is not None:
                    self.dataReady.emit(data, write_index, num_channels)
            except Exception as e:
                print(f"Error in reader thread: {e}")
                import traceback
                traceback.print_exc()

            time.sleep(0.1)  # Update rate (10 Hz)

    def read_audio_data_raw(self):
        try:
            # Read write index
            self.map_file.seek(self.write_index_offset)
            write_index_bytes = self.map_file.read(8)
            if len(write_index_bytes) < 8:
                return None, 0, 0

            write_index = struct.unpack("Q", write_index_bytes)[0]

            # Read number of channels
            self.map_file.seek(self.num_channels_offset)
            channel_bytes = self.map_file.read(4)
            if len(channel_bytes) < 4:
                return None, write_index, 0

            num_channels = struct.unpack("i", channel_bytes)[0]
            num_channels = min(8, max(1, num_channels))  # Ensure reasonable range

            # Calculate where to read the most recent frames
            frames_to_read = 16  # Just read a few frames for display

            # Based on your hex dump, the buffer might be very small
            # or the write index approach might need adjustment
            current_pos = write_index % RING_BUFFER_SIZE
            start_frame = max(0, current_pos - frames_to_read)

            # Create result array
            result = np.zeros((frames_to_read, MAX_CHANNELS))

            # Try reading raw bytes first for diagnostic
            self.map_file.seek(self.audio_data_offset)
            raw_data = self.map_file.read(1024)  # Read some raw data
            if all(b == 0 for b in raw_data):
                # If all zeros, the audio data hasn't been written yet
                return result, write_index, num_channels

            # Read each frame individually
            for i in range(frames_to_read):
                frame_index = (start_frame + i) % RING_BUFFER_SIZE
                offset = self.audio_data_offset + frame_index * num_channels * FLOAT_SIZE

                if offset + (num_channels * FLOAT_SIZE) > MAX_BUFFER_SIZE:
                    continue  # Skip if would read beyond buffer

                # Read all channels in this frame
                self.map_file.seek(offset)
                frame_bytes = self.map_file.read(FLOAT_SIZE * num_channels)

                if len(frame_bytes) < FLOAT_SIZE * num_channels:
                    continue

                # Extract each channel value
                for ch in range(num_channels):
                    start_idx = ch * FLOAT_SIZE
                    end_idx = start_idx + FLOAT_SIZE
                    try:
                        if start_idx < len(frame_bytes) and end_idx <= len(frame_bytes):
                            value = struct.unpack("f", frame_bytes[start_idx:end_idx])[0]
                            result[i, ch] = value
                    except Exception as e:
                        print(f"Error unpacking channel {ch}: {e}")

            return result, write_index, num_channels

        except Exception as e:
            print(f"Error reading audio data: {e}")
            import traceback
            traceback.print_exc()
            return None, 0, 0

    def set_debug_offset(self, offset):
        """Set the memory offset for debugging"""
        self.debug_offset = offset
        self.update_memory_dump()

    def stop(self):
        self.running = False
        self.wait()
        if hasattr(self, 'map_file'):
            self.map_file.close()


class AudioBufferMonitor(QMainWindow):
    """Main window for displaying raw buffer data"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Buffer Raw Data Monitor")
        self.setMinimumSize(1000, 600)

        # Central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Info labels
        self.info_label = QLabel("Waiting for data...")
        layout.addWidget(self.info_label)

        # Create table for raw data display
        self.data_table = QTableWidget(16, MAX_CHANNELS + 1)  # +1 for frame index
        self.data_table.setHorizontalHeaderItem(0, QTableWidgetItem("Frame"))
        for ch in range(MAX_CHANNELS):
            self.data_table.setHorizontalHeaderItem(ch + 1, QTableWidgetItem(f"Ch {ch + 1}"))

        # Set table properties
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setAlternatingRowColors(True)
        font = QFont("Courier", 10)  # Use Courier instead of Monospace
        self.data_table.setFont(font)
        layout.addWidget(self.data_table)

        # Stats label
        self.stats_label = QLabel("Buffer Stats: None")
        layout.addWidget(self.stats_label)

        # Hex dump for debugging
        dump_layout = QVBoxLayout()
        dump_header = QHBoxLayout()

        dump_header.addWidget(QLabel("Memory Hex Dump:"))

        # Offset selector
        dump_header.addWidget(QLabel("Offset:"))
        self.offset_spinner = QSpinBox()
        self.offset_spinner.setRange(0, MAX_BUFFER_SIZE - 256)
        self.offset_spinner.setSingleStep(16)
        self.offset_spinner.valueChanged.connect(self.change_debug_offset)
        dump_header.addWidget(self.offset_spinner)

        dump_header.addStretch()
        dump_layout.addLayout(dump_header)

        # Hex dump display
        self.hex_dump = QTableWidget(16, 16)
        self.hex_dump.setFont(font)
        self.hex_dump.verticalHeader().setVisible(False)
        self.hex_dump.setMaximumHeight(200)

        # Set hex dump header (0-F)
        for i in range(16):
            self.hex_dump.setHorizontalHeaderItem(i, QTableWidgetItem(f"{i:X}"))

        dump_layout.addWidget(self.hex_dump)
        layout.addLayout(dump_layout)

        self.setCentralWidget(central_widget)

        # Set dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #212121;
                color: #FFFFFF;
            }
            QTableWidget {
                background-color: #333333;
                alternate-background-color: #3a3a3a;
                color: #EEEEEE;
                gridline-color: #444444;
                selection-background-color: #4444AA;
            }
            QHeaderView::section {
                background-color: #444444;
                color: #FFFFFF;
                padding: 4px;
                border: 1px solid #555555;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 12px;
            }
            QSpinBox {
                background-color: #333333;
                color: #FFFFFF;
                border: 1px solid #555555;
            }
        """)

        # Start the data reader thread
        self.data_reader = AudioDataReader()
        self.data_reader.dataReady.connect(self.update_table)
        self.data_reader.memoryDump.connect(self.update_hex_dump)
        self.data_reader.start()

        # Last update time for FPS calculation
        self.last_update_time = time.time()
        self.updates_count = 0

    def update_table(self, data, write_index, num_channels):
        """Update the table with new data"""
        self.updates_count += 1

        # Update info label
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        if elapsed >= 1.0:
            fps = self.updates_count / elapsed
            self.info_label.setText(f"Write Index: {write_index} | Channels: {num_channels} | Updates: {fps:.1f}/sec")
            self.last_update_time = current_time
            self.updates_count = 0

        # Calculate statistics
        data_min = np.min(data)
        data_max = np.max(data)
        data_mean = np.mean(data)
        data_std = np.std(data)
        self.stats_label.setText(
            f"Buffer Stats: Min={data_min:.6f}, Max={data_max:.6f}, Mean={data_mean:.6f}, StdDev={data_std:.6f}")

        # Populate table with data
        num_frames = min(16, data.shape[0])

        for row in range(num_frames):
            # Frame index
            frame_idx = write_index - num_frames + row
            frame_item = QTableWidgetItem(str(frame_idx))
            self.data_table.setItem(row, 0, frame_item)

            # Channel data
            for col in range(min(MAX_CHANNELS, data.shape[1])):
                value = data[row, col]
                item = QTableWidgetItem(f"{value:.6f}")

                # Color code based on value
                if abs(value) > 0.9:
                    # Near clipping
                    item.setBackground(QColor(230, 80, 80))
                elif abs(value) > 0.01:
                    # Normal signal
                    item.setBackground(QColor(80, 180, 80))

                self.data_table.setItem(row, col + 1, item)

    def update_hex_dump(self, hex_values):
        """Update the hex dump display"""
        # Clear the table
        for row in range(16):
            for col in range(16):
                index = row * 16 + col
                if index < len(hex_values):
                    item = QTableWidgetItem(hex_values[index])

                    # Color non-zero values differently
                    if hex_values[index] != "00":
                        item.setBackground(QColor(100, 150, 200))

                    self.hex_dump.setItem(row, col, item)
                else:
                    self.hex_dump.setItem(row, col, QTableWidgetItem(""))

        # Update the address labels
        for row in range(16):
            addr = self.offset_spinner.value() + row * 16
            self.hex_dump.setVerticalHeaderItem(row, QTableWidgetItem(f"{addr:04X}"))

    def change_debug_offset(self, value):
        """Change the memory offset for debugging"""
        if hasattr(self, 'data_reader'):
            self.data_reader.set_debug_offset(value)

    def closeEvent(self, event):
        """Handle window close event"""
        self.data_reader.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = AudioBufferMonitor()
    window.show()

    # Start the event loop
    exit_code = app.exec_()

    # Cleanup
    print("Application closed")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()