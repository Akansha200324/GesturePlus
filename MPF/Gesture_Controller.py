import cv2
import mediapipe as mp
import pyautogui
import math
from enum import IntEnum
import numpy as np
import platform
import logging
import os  # required for brightness control on Linux

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# OS-specific audio control
OS_NAME = platform.system().lower()
if OS_NAME == 'darwin':  # macOS
    import osascript
elif OS_NAME == 'linux':
    import alsaaudio
else:  # Windows
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False

# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Gesture Encodings 
class Gest(IntEnum):
    """
    Enum for mapping all hand gestures to a binary number.
    """
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31
    # Extra Mappings
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36

# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

# Convert Mediapipe Landmarks to recognizable Gestures
class HandRecog:
    """
    Converts MediaPipe hand landmarks into recognized gestures.
    """
    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label
    
    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_signed_dist(self, point):
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        dist = math.sqrt(dist)
        return dist * sign
    
    def get_dist(self, point):
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        return math.sqrt(dist)
    
    def get_dz(self, point):
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)
    
    def set_finger_state(self):
        if self.hand_result is None:
            return
        points = [[8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
        self.finger = 0
        self.finger = self.finger | 0  # thumb (currently not processed)
        for point in points:
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            try:
                ratio = round(dist/dist2, 1)
            except:
                ratio = round(dist/0.01, 1)
            self.finger = self.finger << 1
            if ratio > 0.5:
                self.finger = self.finger | 1
    
    def get_gesture(self):
        if self.hand_result is None:
            return Gest.PALM

        current_gesture = Gest.PALM
        if self.finger in [Gest.LAST3, Gest.LAST4] and self.get_dist([8, 4]) < 0.05:
            if self.hand_label == HLabel.MINOR:
                current_gesture = Gest.PINCH_MINOR
            else:
                current_gesture = Gest.PINCH_MAJOR
        elif Gest.FIRST2 == self.finger:
            point = [[8, 12], [5, 9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1/dist2
            if ratio > 1.7:
                current_gesture = Gest.V_GEST
            else:
                if self.get_dz([8, 12]) < 0.1:
                    current_gesture = Gest.TWO_FINGER_CLOSED
                else:
                    current_gesture = Gest.MID
        else:
            current_gesture = self.finger
        
        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.prev_gesture = current_gesture
        if self.frame_count > 4:
            self.ori_gesture = current_gesture
        return self.ori_gesture

# Executes commands according to detected gestures
class Controller:
    tx_old = 0
    ty_old = 0
    trial = True
    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    pinchstartxcoord = None
    pinchstartycoord = None
    pinchdirectionflag = None
    prevpinchlv = 0
    pinchlv = 0
    framecount = 0
    prev_hand = None
    pinch_threshold = 0.3

    def getpinchylv(hand_result):
        dist = round((Controller.pinchstartycoord - hand_result.landmark[8].y) * 10, 1)
        return dist

    def getpinchxlv(hand_result):
        dist = round((hand_result.landmark[8].x - Controller.pinchstartxcoord) * 10, 1)
        return dist
    
    def scrollVertical():
        pyautogui.scroll(120 if Controller.pinchlv > 0.0 else -120)
        
    def scrollHorizontal():
        pyautogui.keyDown('shift')
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-120 if Controller.pinchlv > 0.0 else 120)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('shift')

    def get_position(hand_result):
        point = 9
        position = [hand_result.landmark[point].x, hand_result.landmark[point].y]
        sx, sy = pyautogui.size()
        x_old, y_old = pyautogui.position()
        x = int(position[0] * sx)
        y = int(position[1] * sy)
        if Controller.prev_hand is None:
            Controller.prev_hand = (x, y)
        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]
        distsq = delta_x**2 + delta_y**2
        ratio = 1
        Controller.prev_hand = (x, y)
        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** 0.5)
        else:
            ratio = 2.1
        x, y = x_old + delta_x * ratio, y_old + delta_y * ratio
        return (x, y)

    def pinch_control_init(hand_result):
        Controller.pinchstartxcoord = hand_result.landmark[8].x
        Controller.pinchstartycoord = hand_result.landmark[8].y
        Controller.pinchlv = 0
        Controller.prevpinchlv = 0
        Controller.framecount = 0

    def pinch_control(hand_result, controlHorizontal, controlVertical):
        if Controller.framecount == 5:
            Controller.framecount = 0
            Controller.pinchlv = Controller.prevpinchlv
            if Controller.pinchdirectionflag == True:
                controlHorizontal()  # x-axis control
            elif Controller.pinchdirectionflag == False:
                controlVertical()    # y-axis control

        lvx = Controller.getpinchxlv(hand_result)
        lvy = Controller.getpinchylv(hand_result)
            
        if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = False
            if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvy
                Controller.framecount = 0
        elif abs(lvx) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = True
            if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvx
                Controller.framecount = 0

    def handle_controls(gesture, hand_result):      
        x, y = None, None
        if gesture != Gest.PALM:
            x, y = Controller.get_position(hand_result)
        
        # Reset flags if gesture changes
        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button="left")

        if gesture != Gest.PINCH_MAJOR and Controller.pinchmajorflag:
            Controller.pinchmajorflag = False

        if gesture != Gest.PINCH_MINOR and Controller.pinchminorflag:
            Controller.pinchminorflag = False

        # Execute controls based on recognized gesture
        if gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration=0.1)
        elif gesture == Gest.FIST:
            if not Controller.grabflag:
                Controller.grabflag = True
                pyautogui.mouseDown(button="left")
            pyautogui.moveTo(x, y, duration=0.1)
        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click()
            Controller.flag = False
        elif gesture == Gest.INDEX and Controller.flag:
            pyautogui.click(button='right')
            Controller.flag = False
        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            pyautogui.doubleClick()
            Controller.flag = False
        elif gesture == Gest.PINCH_MINOR:
            if Controller.pinchminorflag == False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchminorflag = True
            Controller.pinch_control(hand_result, Controller.scrollHorizontal, Controller.scrollVertical)
        elif gesture == Gest.PINCH_MAJOR:
            if Controller.pinchmajorflag == False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchmajorflag = True
            Controller.pinch_control(hand_result, Controller.changesystembrightness, Controller.changesystemvolume)

    @staticmethod
    def changesystemvolume():
        if OS_NAME == 'darwin':  # macOS
            volume = Controller.pinchlv
            volume = int((volume * 100) if volume >= 0 else 0)
            osascript.osascript(f"set volume output volume {volume}")
        elif OS_NAME == 'linux':
            mixer = alsaaudio.Mixer()
            current_volume = mixer.getvolume()[0]
            volume = Controller.pinchlv
            volume = int((volume * 100) if volume >= 0 else 0)
            mixer.setvolume(volume)
        else:  # Windows
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(Controller.pinchlv, None)

    @staticmethod
    def changesystembrightness():
        if OS_NAME == 'darwin':  # macOS
            pass  # Requires additional permissions or a third-party tool
        elif OS_NAME == 'linux':
            brightness = Controller.pinchlv
            brightness = int((brightness * 100) if brightness >= 0 else 0)
            os.system(f"xrandr --output $(xrandr -q | grep ' connected' | head -n 1 | cut -d ' ' -f1) --brightness {brightness/100}")
        else:  # Windows
            import screen_brightness_control as sbcontrol
            brightness = Controller.pinchlv
            brightness = int((brightness * 100) if brightness >= 0 else 0)
            sbcontrol.set_brightness(brightness, display=0)

# Gesture Controller class
class GestureController:
    gc_mode = 1
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    hr_major = None
    hr_minor = None
    exit_counter = 0  # Counter for exit gesture
    
    def __init__(self):
        GestureController.gc_mode = 1
        self._setup_camera()
        self._setup_mediapipe()
        
    def _setup_camera(self):
        GestureController.cap = cv2.VideoCapture(0)
        if not GestureController.cap.isOpened():
            raise RuntimeError("Could not open video capture device")
        GestureController.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        GestureController.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureController.CAM_WIDTH = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        
    def _setup_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
    @staticmethod
    def classify_hands(results):
        GestureController.hr_major = None
        GestureController.hr_minor = None
        try:
            if results.multi_handedness:
                for idx, hand_handedness in enumerate(results.multi_handedness):
                    handedness_dict = hand_handedness.classification[0]
                    if handedness_dict.label == 'Right':
                        GestureController.hr_major = results.multi_hand_landmarks[idx]
                    else:
                        GestureController.hr_minor = results.multi_hand_landmarks[idx]
        except Exception as e:
            logger.error(f"Error classifying hands: {str(e)}")
    
    def start(self):
        """
        Entry point of the program.
        Captures video frames, obtains landmarks from MediaPipe, and independently processes
        gestures for both detected hands. Also supports an exit gesture (bringing both wrists close)
        if held for 30 consecutive frames.
        """
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)

        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while GestureController.cap.isOpened() and GestureController.gc_mode:
                success, image = GestureController.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    # Classify hands based on handedness
                    GestureController.classify_hands(results)
                    
                    # Process right hand (major)
                    if GestureController.hr_major:
                        handmajor.update_hand_result(GestureController.hr_major)
                        handmajor.set_finger_state()
                        gest_major = handmajor.get_gesture()
                        Controller.handle_controls(gest_major, handmajor.hand_result)
                    
                    # Process left hand (minor)
                    if GestureController.hr_minor:
                        handminor.update_hand_result(GestureController.hr_minor)
                        handminor.set_finger_state()
                        gest_minor = handminor.get_gesture()
                        Controller.handle_controls(gest_minor, handminor.hand_result)
                    
                    # Draw landmarks for all detected hands
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # ------------------ Exit Gesture Feature ------------------
                    # When two hands are present, check if their wrist landmarks are very close.
                    # If they remain close for 30 consecutive frames, exit the program.
                    if len(results.multi_hand_landmarks) >= 2:
                        wrist1 = results.multi_hand_landmarks[0].landmark[0]
                        wrist2 = results.multi_hand_landmarks[1].landmark[0]
                        dist = math.sqrt((wrist1.x - wrist2.x)**2 + (wrist1.y - wrist2.y)**2)
                        if dist < 0.1:
                            GestureController.exit_counter += 1
                            if GestureController.exit_counter >= 30:
                                print("Exit gesture detected. Exiting...")
                                break
                        else:
                            GestureController.exit_counter = 0
                    # ------------------------------------------------------------
                else:
                    Controller.prev_hand = None
                
                cv2.imshow('Gesture Controller (Press Enter or show exit gesture to exit)', image)
                if cv2.waitKey(5) & 0xFF == 13:
                    break
        GestureController.cap.release()
        cv2.destroyAllWindows()

# Uncomment to run directly
gc1 = GestureController()
gc1.start()
