#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import io
import speech_recognition as sr
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError
import google_auth_httplib2
import time

class ImageSaver:
    def __init__(self):
        self.node_name = "image_saver"
        rospy.init_node(self.node_name, anonymous=True)

        self.bridge = CvBridge()

        # Google Drive API setup
        self.setup_google_drive()

        # Initialize flags to ensure each image is saved only once per command
        self.depth_saved = False
        self.color_saved = False

        # No active subscriptions initially
        self.depth_subscription = None
        self.color_subscription = None

        # Start listening for voice commands
        self.listen_for_command()

    def setup_google_drive(self):
        # OAuth 2.0 scope
        SCOPES = ['https://www.googleapis.com/auth/drive.file']

        # Load credentials and authenticate
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('/home/kyhb/catkin_ws/src/grasp_demo/scripts/credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        # Create Drive API client
        self.drive_service = build('drive', 'v3', credentials=creds)

    def upload_to_drive(self, file_name, file_data, mime_type):
        file_id = self.get_file_id(file_name)
        while True:  # Continuously try until the upload is successful
            try:
                if file_id:
                    # Update the existing file
                    file = self.drive_service.files().update(
                        fileId=file_id,
                        media_body=MediaIoBaseUpload(io.BytesIO(file_data), mimetype=mime_type)
                    ).execute()
                    rospy.loginfo('File {} updated in Google Drive with ID: {}'.format(file_name, file.get("id")))
                else:
                    # Create a new file
                    file_metadata = {
                        'name': file_name,
                        'parents': ['17spYQEf3v3qTsQI0D3_YSJ5nZ0FjszHW']  # Replace with your Google Drive folder ID
                    }
                    file = self.drive_service.files().create(
                        body=file_metadata,
                        media_body=MediaIoBaseUpload(io.BytesIO(file_data), mimetype=mime_type),
                        fields='id'
                    ).execute()
                    rospy.loginfo('File {} uploaded to Google Drive with ID: {}'.format(file_name, file.get("id")))
                return  # If upload is successful, exit the function

            except HttpError as e:
                rospy.logerr('Failed to upload {} to Google Drive: {}'.format(file_name, e))
                time.sleep(1)  # Wait a short time before retrying
            except Exception as e:
                rospy.logerr('Unknown error occurred while uploading {}: {}'.format(file_name, e))
                time.sleep(1)  # Wait a short time before retrying

    def get_file_id(self, file_name):
        """Check if the file already exists in Google Drive and return its ID."""
        try:
            query = "name='{}' and parents in '17spYQEf3v3qTsQI0D3_YSJ5nZ0FjszHW' and trashed=false".format(file_name)
            results = self.drive_service.files().list(q=query, fields="files(id, name)").execute()
            files = results.get('files', [])
            if files:
                return files[0]['id']
            return None
        except HttpError as e:
            rospy.logerr('Failed to check for existing file: {}'.format(e))
            return None

    def depth_callback(self, msg):
        if not self.depth_saved:
            try:
                # Convert the ROS Image message to a CV2 Image for depth
                cv_image_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # Encode image to PNG format in memory
                _, encoded_image = cv2.imencode('.png', cv_image_depth)
                # Upload to Google Drive
                self.upload_to_drive('depth.png', encoded_image.tobytes(), 'image/png')
                # Set flag to True to prevent saving again
                self.depth_saved = True
            except Exception as e:
                rospy.logerr('Failed to process depth image: %s' % e)

    def color_callback(self, msg):
        if not self.color_saved:
            try:
                # Convert the ROS Image message to a CV2 Image for color
                cv_image_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                # Encode image to PNG format in memory
                _, encoded_image = cv2.imencode('.png', cv_image_color)
                # Upload to Google Drive
                self.upload_to_drive('color.png', encoded_image.tobytes(), 'image/png')
                # Set flag to True to prevent saving again
                self.color_saved = True
            except Exception as e:
                rospy.logerr('Failed to process color image: %s' % e)

    def listen_for_command(self):
        """Continuously listen for the voice command 'what can you see' and trigger image capture/upload."""
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        rospy.loginfo("Listening for the command 'what can you see'...")

        while not rospy.is_shutdown():
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = recognizer.listen(source, timeout=5.0)
                    command = recognizer.recognize_google(audio).lower()

                    if "what can you see" in command:
                        rospy.loginfo("Heard the command 'what can you see'. Capturing images...")

                        # Reset flags to allow new image uploads
                        self.depth_saved = False
                        self.color_saved = False

                        # Start subscribing to image topics
                        self.start_subscribers()

                        # Wait until both images have been uploaded
                        while not (self.depth_saved and self.color_saved):
                            rospy.sleep(0.1)

                        # Stop subscriptions after images are uploaded
                        self.stop_subscribers()

                        rospy.loginfo("Images have been captured and uploaded. Listening for the next command...")
                
                except sr.UnknownValueError:
                    rospy.loginfo("Could not understand the command. Please try again.")
                except sr.RequestError as e:
                    rospy.logerr("Could not request results from the speech recognition service; {0}".format(e))
                except Exception as e:
                    rospy.logerr('Error occurred while listening for command: {}'.format(e))

    def start_subscribers(self):
        """Activate the ROS subscribers when the voice command is detected."""
        self.depth_subscription = rospy.Subscriber(
            '/d435/depth/image_raw',
            Image,
            self.depth_callback,
            queue_size=10)
        
        self.color_subscription = rospy.Subscriber(
            '/d435/color/image_raw',
            Image,
            self.color_callback,
            queue_size=10)

    def stop_subscribers(self):
        """Deactivate the ROS subscribers after the images are captured and uploaded."""
        if self.depth_subscription:
            self.depth_subscription.unregister()
            self.depth_subscription = None
        
        if self.color_subscription:
            self.color_subscription.unregister()
            self.color_subscription = None

if __name__ == '__main__':
    try:
        image_saver = ImageSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
