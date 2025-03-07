import sys
import os
import platform
import subprocess
import webbrowser
import speech_recognition as sr
import pyttsx3
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit, 
                             QPushButton, QVBoxLayout, QWidget, QMessageBox,
                             QProgressBar, QLabel, QHBoxLayout, QInputDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent
from PyQt6.QtGui import QFont, QIcon
from pynput.keyboard import Controller, Key
# Attempt to import GestureController from a module named Gesture_Controller
try:
    from Gesture_Controller import GestureController
except ImportError:
    GestureController = None

class SpeechThread(QThread):
    text_detected = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    listening_level = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.is_listening = True
        self.recognizer = sr.Recognizer()
        # Improve noise handling
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 4000

    def run(self):
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                while self.is_listening:
                    try:
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        if hasattr(audio, 'get_raw_data'):
                            level = max(abs(x) for x in audio.get_raw_data()) / 32768.0
                            self.listening_level.emit(level)
                        text = self.recognizer.recognize_google(audio, language='en-US').lower()
                        if text.strip():
                            self.text_detected.emit(text)
                    except sr.WaitTimeoutError:
                        continue
                    except sr.RequestError as e:
                        self.error_occurred.emit(f"Could not request results: {str(e)}")
                    except Exception as e:
                        self.error_occurred.emit(f"Error processing audio: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Microphone error: {str(e)}")

    def stop(self):
        self.is_listening = False

class VoiceAssistant(QMainWindow):
    def __init__(self):
        super().__init__()
        self.speech_engine = None
        self.speech_thread = None
        self.command_history = []
        self.command_index = -1
        
        # For file navigation: set default path based on OS
        if platform.system() == "Windows":
            self.current_path = "C:\\"
        else:
            self.current_path = "/"
        self.file_list = []
        
        # For gesture recognition
        self.gesture_active = False
        self.gesture_controller = None
        
        # For clipboard operations
        self.keyboard = Controller()
        
        # System active flag (for sleep/wake functionality)
        self.active = True
        
        self.setWindowTitle("Proton Voice Assistant")
        self.setGeometry(100, 100, 800, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QLabel {
                font-size: 14px;
                color: #2c3e50;
            }
            QTextEdit {
                background-color: white;
                color: black;
                border: 2px solid #e1e4e8;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Arial', sans-serif;
                font-size: 14px;
            }
            QLineEdit {
                background-color: white;
                color: black;
                padding: 8px;
                border: 2px solid #e1e4e8;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton {
                font-weight: bold;
                padding: 10px;
                border-radius: 6px;
            }
        """)
        
        self.init_speech_engine()
        self.init_ui()
        self.wish()  # greet the user upon startup
        self.is_listening = False

    def wish(self):
        """Greet the user based on the current time and introduce Proton."""
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good Morning!"
        elif hour < 18:
            greeting = "Good Afternoon!"
        else:
            greeting = "Good Evening!"
        self.speak(f"{greeting} I am Proton, how may I help you?")

    def toggle_voice_input(self):
        if not self.is_listening:
            self.is_listening = True
            self.voice_btn.setText("Stop Listening")
            self.voice_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
            """)
            self.status_label.setText("Listening...")
            self.speech_thread = SpeechThread()
            self.speech_thread.text_detected.connect(self.process_voice_command)
            self.speech_thread.error_occurred.connect(self.handle_speech_error)
            self.speech_thread.listening_level.connect(self.update_level_bar)
            self.speech_thread.start()
        else:
            self.is_listening = False
            self.voice_btn.setText("Voice Input")
            self.voice_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            self.status_label.setText("Ready")
            if self.speech_thread:
                self.speech_thread.stop()
                self.speech_thread.wait()
                self.speech_thread = None
            self.level_bar.setValue(0)

    def handle_speech_error(self, error_message):
        self.status_label.setText("Error")
        self.conversation_log.append(f"\n⚠️ Error: {error_message}")
        if self.is_listening:
            if self.speech_thread:
                self.speech_thread.stop()
                self.speech_thread.wait()
            self.speech_thread = SpeechThread()
            self.speech_thread.text_detected.connect(self.process_voice_command)
            self.speech_thread.error_occurred.connect(self.handle_speech_error)
            self.speech_thread.listening_level.connect(self.update_level_bar)
            self.speech_thread.start()

    def update_level_bar(self, level):
        self.level_bar.setValue(int(level * 100))

    def closeEvent(self, event):
        if self.speech_thread:
            self.speech_thread.stop()
            self.speech_thread.wait()
        event.accept()

    def init_speech_engine(self):
        try:
            self.speech_engine = pyttsx3.init()
            voices = self.speech_engine.getProperty('voices')
            us_voice = None
            # Prefer an American English female voice (e.g., "Samantha" on macOS)
            for voice in voices:
                if "samantha" in voice.name.lower():
                    us_voice = voice
                    break
            if not us_voice:
                for voice in voices:
                    if (("female" in voice.id.lower() or "woman" in voice.id.lower() or "zira" in voice.id.lower()) and
                        ("en" in voice.id.lower() or "english" in voice.name.lower())):
                        us_voice = voice
                        break
            if not us_voice and len(voices) > 1:
                us_voice = voices[1]
            elif not us_voice and voices:
                us_voice = voices[0]
            if us_voice:
                self.speech_engine.setProperty('voice', us_voice.id)
            self.speech_engine.setProperty('rate', 165)
            self.speech_engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"Warning: Could not initialize speech engine: {str(e)}")
            self.speech_engine = None

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        self.level_bar = QProgressBar()
        self.level_bar.setMaximum(100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e1e4e8;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        status_layout.addWidget(self.level_bar)
        main_layout.addLayout(status_layout)

        self.conversation_log = QTextEdit()
        self.conversation_log.setReadOnly(True)
        self.conversation_log.setMinimumHeight(400)
        main_layout.addWidget(self.conversation_log)

        input_layout = QHBoxLayout()
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Type a command or press Voice Input...")
        self.command_input.returnPressed.connect(self.process_text_command)
        input_layout.addWidget(self.command_input)

        self.voice_btn = QPushButton("Voice Input")
        self.voice_btn.clicked.connect(self.toggle_voice_input)
        self.voice_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        input_layout.addWidget(self.voice_btn)
        main_layout.addLayout(input_layout)

        help_btn = QPushButton("Show Commands")
        help_btn.clicked.connect(self.show_help)
        help_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        main_layout.addWidget(help_btn)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.command_input.installEventFilter(self)

    def show_help(self):
        help_text = """
Available Commands:
- "open [application]" - Opens specified application
- "time" - Shows current time
- "date" - Shows current date
- "clear" - Clears conversation log
- "help" - Shows this help message
- "search [query]" - Searches Google for the query
- "location" - Looks up a location on Google Maps
- "launch gesture recognition" - Starts gesture recognition
- "stop gesture recognition" - Stops gesture recognition
- "copy" - Simulates copy (Ctrl+C)
- "paste" - Simulates paste (Ctrl+V)
- "list" - Lists files/folders in the current directory
- "open [number]" - Opens a file/folder from the list
- "back" - Navigates to the parent directory
- "bye" - Puts Proton to sleep
- "wake up" - Wakes Proton up
- "exit/terminate" - Closes the application
- Basic conversation: hi, hello, how are you, thanks
        """
        QMessageBox.information(self, "Available Commands", help_text)

    def process_voice_command(self, command):
        self.conversation_log.append(f"\n👤 User (voice): {command}")
        self.execute_command(command)

    def process_text_command(self):
        command = self.command_input.text().lower()
        if command:
            self.command_history.append(command)
            self.command_index = -1
            self.conversation_log.append(f"\n⌨️ User (text): {command}")
            self.execute_command(command)
            self.command_input.clear()

    def speak(self, text):
        self.conversation_log.append(f"\n🤖 Assistant: {text}")
        self.status_label.setText("Speaking...")
        if self.speech_engine:
            try:
                sentences = text.split('.')
                for sentence in sentences:
                    if sentence.strip():
                        self.speech_engine.say(sentence.strip())
                        self.speech_engine.runAndWait()
            except Exception as e:
                print(f"Speech engine error: {str(e)}")
                self.conversation_log.append("(Speech output failed)")
        self.status_label.setText("Ready")

    def execute_command(self, command):
        try:
            # If Proton is asleep, ignore commands except "wake up"
            if not self.active and "wake up" not in command:
                self.speak("I am sleeping. Please say 'wake up' to reactivate me.")
                return

            # System Control: exit/terminate
            if any(word in command for word in ['exit', 'terminate']):
                self.speak("Goodbye! Have a great day!")
                QTimer.singleShot(2000, self.close)
            
            # Sleep and wake commands
            elif "bye" in command:
                self.active = False
                self.speak("Going to sleep. Say 'wake up' to reactivate me.")
            
            elif "wake up" in command:
                self.active = True
                self.wish()
            
            elif 'search' in command:
                query = command.split("search", 1)[1].strip()
                if query:
                    url = f"https://google.com/search?q={query}"
                    webbrowser.open(url)
                    self.speak(f"Searching for {query}")
                else:
                    self.speak("Please specify what you want to search for.")
            
            elif 'location' in command:
                # Prompt user for location using a dialog box
                location, ok = QInputDialog.getText(self, "Location Lookup", "Enter the location:")
                if ok and location:
                    url = f"https://google.nl/maps/place/{location}"
                    webbrowser.open(url)
                    self.speak(f"Looking up the location: {location}")
                else:
                    self.speak("No location provided.")
            
            elif "launch gesture recognition" in command:
                if not self.gesture_active:
                    if GestureController is not None:
                        try:
                            self.gesture_controller = GestureController()
                            self.gesture_controller.start()
                            self.gesture_active = True
                            self.speak("Gesture recognition launched.")
                        except Exception as e:
                            self.speak(f"Failed to launch gesture recognition: {str(e)}")
                    else:
                        self.speak("Gesture recognition module is not available.")
                else:
                    self.speak("Gesture recognition is already active.")
            
            elif "stop gesture recognition" in command:
                if self.gesture_active:
                    try:
                        if self.gesture_controller:
                            self.gesture_controller.stop()  # assuming a stop method or control flag
                        self.gesture_active = False
                        self.speak("Gesture recognition stopped.")
                    except Exception as e:
                        self.speak(f"Failed to stop gesture recognition: {str(e)}")
                else:
                    self.speak("Gesture recognition is not active.")
            
            elif "copy" in command:
                # Simulate Ctrl+C for copy
                self.keyboard.press(Key.ctrl)
                self.keyboard.press('c')
                self.keyboard.release('c')
                self.keyboard.release(Key.ctrl)
                self.speak("Copied to clipboard.")
            
            elif any(word in command for word in ["paste", "page", "pest"]):
                # Simulate Ctrl+V for paste
                self.keyboard.press(Key.ctrl)
                self.keyboard.press('v')
                self.keyboard.release('v')
                self.keyboard.release(Key.ctrl)
                self.speak("Pasted from clipboard.")
            
            elif command.strip() == "list":
                # List files/folders in the current directory
                try:
                    self.file_list = os.listdir(self.current_path)
                    if not self.file_list:
                        self.speak("The directory is empty.")
                    else:
                        response = "Listing files and folders:\n"
                        for idx, item in enumerate(self.file_list, start=1):
                            response += f"{idx}. {item}\n"
                        self.speak(response)
                except Exception as e:
                    self.speak(f"Failed to list directory: {str(e)}")
            
            elif command.startswith("open "):
                # Check if a number is provided to open an item from the list
                parts = command.split()
                if len(parts) == 2 and parts[1].isdigit():
                    index = int(parts[1]) - 1
                    if 0 <= index < len(self.file_list):
                        item = self.file_list[index]
                        item_path = os.path.join(self.current_path, item)
                        if os.path.isdir(item_path):
                            self.current_path = item_path
                            try:
                                self.file_list = os.listdir(self.current_path)
                                response = f"Opened folder {item}. Listing contents:\n"
                                for idx, sub_item in enumerate(self.file_list, start=1):
                                    response += f"{idx}. {sub_item}\n"
                                self.speak(response)
                            except Exception as e:
                                self.speak(f"Failed to open folder: {str(e)}")
                        else:
                            try:
                                if platform.system() == "Windows":
                                    os.startfile(item_path)
                                elif platform.system() == "Darwin":
                                    subprocess.Popen(["open", item_path])
                                else:
                                    subprocess.Popen(["xdg-open", item_path])
                                self.speak(f"Opened file {item}.")
                            except Exception as e:
                                self.speak(f"Failed to open file: {str(e)}")
                    else:
                        self.speak("Invalid file number.")
                else:
                    # Fallback: try opening the command as an application name
                    app = command.replace('open', '').strip()
                    self.open_application(app)
            
            elif command.strip() == "back":
                parent = os.path.dirname(self.current_path.rstrip(os.sep))
                if parent and parent != self.current_path:
                    self.current_path = parent
                    try:
                        self.file_list = os.listdir(self.current_path)
                        response = "Moved back. Listing contents:\n"
                        for idx, item in enumerate(self.file_list, start=1):
                            response += f"{idx}. {item}\n"
                        self.speak(response)
                    except Exception as e:
                        self.speak(f"Failed to list directory: {str(e)}")
                else:
                    self.speak("Already at the root directory.")
            
            elif 'time' in command:
                current_time = datetime.now().strftime('%I:%M %p')
                self.speak(f"The current time is {current_time}")
            
            elif 'date' in command:
                current_date = datetime.now().strftime('%B %d, %Y')
                self.speak(f"Today is {current_date}")
            
            elif 'clear' in command:
                self.conversation_log.clear()
                self.speak("I've cleared the conversation log")
            
            elif any(greet in command for greet in ['hi', 'hello', 'hey']):
                self.speak("Hello there! How can I assist you today?")
            
            elif "how are you" in command:
                self.speak("I'm doing great, thank you! How can I help you?")
            
            elif "thank" in command:
                self.speak("You're welcome!")
            
            else:
                response = self.generate_response(command)
                self.speak(response)
        
        except Exception as e:
            self.speak(f"I encountered an error: {str(e)}")
            print(f"Error executing command: {str(e)}")

    def open_application(self, app_name):
        try:
            if platform.system() == "Windows":
                os.startfile(app_name)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-a", app_name])
            else:
                subprocess.Popen([app_name])
            self.speak(f"Opening {app_name}")
        except Exception as e:
            self.speak(f"Failed to open {app_name}: {str(e)}")

    def generate_response(self, command):
        return f"Sorry, I don't know how to '{command}'."

    def eventFilter(self, source, event):
        if source == self.command_input and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Up:
                if self.command_history:
                    self.command_index = (self.command_index - 1) % len(self.command_history)
                    self.command_input.setText(self.command_history[self.command_index])
                return True
            elif event.key() == Qt.Key.Key_Down:
                if self.command_history:
                    self.command_index = (self.command_index + 1) % len(self.command_history)
                    self.command_input.setText(self.command_history[self.command_index])
                return True
        return super().eventFilter(source, event)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont('Arial', 10))
    assistant = VoiceAssistant()
    assistant.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
