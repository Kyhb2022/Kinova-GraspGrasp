#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import tf
import numpy as np
import speech_recognition as sr  # Import the speech recognition library
import os  # Used to call espeak for direct speech output
from geometry_msgs.msg import TransformStamped

def speak_text(text):
    """Use espeak to output text as speech"""
    try:
        # Use os.system to call espeak and play the text content directly
        os.system('espeak "{}"'.format(text))
    except Exception as e:
        print("Error with TTS engine: {}".format(e))

def get_user_choice(choices):
    """Select a grasp pose via voice input"""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # Output prompt via speaker (using espeak)
    speak_text("I can see the following items: " + ", ".join(choices))
    
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Please say the item you choose:")
            speak_text("Please say the item you choose.")
            audio = recognizer.listen(source)

        try:
            # Recognize voice input (in English)
            command = recognizer.recognize_google(audio, language="en-US").strip()

            print("You said: {}".format(command))

            # Compare ignoring case and remove extra spaces
            command_lower = command.lower().strip()
            choices_lower = [choice.lower().strip() for choice in choices]

            # Check if the user said a valid grasp pose
            if command_lower in choices_lower:
                selected_object = choices[choices_lower.index(command_lower)]
                print("Selected: {}".format(selected_object))
                speak_text("You selected {}".format(selected_object))
                return selected_object
            else:
                print("Did not recognize a valid grasp position, please try again.")
                speak_text("Did not recognize a valid grasp position, please try again.")
        except sr.UnknownValueError:
            print("Sorry, I could not understand your speech. Please try again.")
            speak_text("Sorry, I could not understand your speech. Please try again.")
        except sr.RequestError as e:
            print("Could not request results; {}".format(e))
            speak_text("Could not request results; {}".format(e))


def read_gg_values(filepath):
    """Read grasp pose information"""
    with open(filepath, 'r') as file:
        lines = file.readlines()

    poses = {}
    current_object = None
    translation = None
    rotation = []

    for i, line in enumerate(lines):
        if 'Object Name:' in line:
            current_object = line.split(':')[1].strip()
            poses[current_object] = {}
        elif 'translation:' in line:
            translation_str = line.split('[')[1].split(']')[0]
            translation = [float(num) for num in translation_str.split()]
            poses[current_object]['translation'] = translation
        elif 'rotation:' in line:
            rotation = []
            for j in range(3):
                rotation_line = lines[i + 1 + j]
                rotation_row = [float(num) for num in rotation_line.strip().strip('[]').split()]
                rotation.append(rotation_row)
            poses[current_object]['rotation'] = rotation

    return poses  # Return all grasp poses

class StaticTfBroadcaster:
    """Handle multiple grasp poses and broadcast them via TF"""
    def __init__(self):
        rospy.init_node('static_tf2_broadcaster')

        self.broadcaster = tf.TransformBroadcaster()

        # Read grasp poses from a file
        filepath = '/home/kyhb/catkin_ws/data/gg_values.txt'
        self.poses = read_gg_values(filepath)

        # Get grasp pose names and prompt the user to choose one
        object_choices = list(self.poses.keys())
        selected_object = get_user_choice(object_choices)

        # Process the grasp pose based on the user's choice
        self.translation = self.poses[selected_object]['translation']
        self.rotation_matrix = self.poses[selected_object]['rotation']

        rospy.Timer(rospy.Duration(0.1), self.broadcast_tf)

    def broadcast_tf(self, event):
        """Broadcast the TF transform"""
        # Use the selected translation values
        translation = tuple(self.translation)

        # Use the selected rotation matrix
        rotation_matrix = np.array(self.rotation_matrix).reshape((3, 3))

        # Convert the rotation matrix to a quaternion
        quaternion_original = tf.transformations.quaternion_from_matrix(
            np.vstack((np.hstack((rotation_matrix, [[0], [0], [0]])), [0, 0, 0, 1]))
        )

        # Broadcasting the transform
        self.broadcaster.sendTransform(translation, quaternion_original, rospy.Time.now(), 'grasp', 'my_gen3/d435_depth_optical_frame')

if __name__ == '__main__':
    broadcaster = StaticTfBroadcaster()
    rospy.spin()
