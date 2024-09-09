#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import tf
import numpy as np
import speech_recognition as sr  # 导入语音识别库
import os  # 用于调用 espeak 直接输出语音
from geometry_msgs.msg import TransformStamped

def speak_text(text):
    """使用 espeak 将文本直接输出为语音"""
    try:
        # 使用 os.system 调用 espeak 并直接播放文本内容
        os.system('espeak "{}"'.format(text))
    except Exception as e:
        print("Error with TTS engine: {}".format(e))

def get_user_choice(choices):
    """通过语音选择抓取位姿"""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # 通过扬声器输出提示信息（使用 espeak）
    speak_text("I can see the following items: " + ", ".join(choices)+"       please select one to be grasped")
    
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Please say the item you choose:")
            speak_text("Please say the item position you choose.")
            audio = recognizer.listen(source)

        try:
            # 识别语音输入（使用英语）
            command = recognizer.recognize_google(audio, language="en-US").strip()

            print("You said: {}".format(command))

            # 忽略大小写比较，并移除多余空格
            command_lower = command.lower().strip()
            choices_lower = [choice.lower().strip() for choice in choices]

            # 检查用户是否说出了一个有效的抓取位姿
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
    """读取抓取位姿信息"""
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

    return poses  # 返回所有抓取位姿

class StaticTfBroadcaster:
    """处理多个抓取位姿并通过TF广播"""
    def __init__(self):
        rospy.init_node('static_tf2_broadcaster')

        self.broadcaster = tf.TransformBroadcaster()

        # 从文件中读取抓取位姿
        filepath = '/home/kyhb/catkin_ws/data/gg_values.txt'
        self.poses = read_gg_values(filepath)

        # 获取抓取位姿名称并提示用户选择
        object_choices = list(self.poses.keys())
        selected_object = get_user_choice(object_choices)

        # 根据用户选择的抓取位姿进行处理
        self.translation = self.poses[selected_object]['translation']
        self.rotation_matrix = self.poses[selected_object]['rotation']

        rospy.Timer(rospy.Duration(0.1), self.broadcast_tf)

    def broadcast_tf(self, event):
        """广播TF变换"""
        # 使用选中的translation值
        translation = tuple(self.translation)

        # 使用选中的rotation矩阵
        rotation_matrix = np.array(self.rotation_matrix).reshape((3, 3))

        # 将旋转矩阵转换为四元数
        quaternion_original = tf.transformations.quaternion_from_matrix(
            np.vstack((np.hstack((rotation_matrix, [[0], [0], [0]])), [0, 0, 0, 1]))
        )

        # 广播变换
        self.broadcaster.sendTransform(translation, quaternion_original, rospy.Time.now(), 'grasp', 'my_gen3/d435_depth_optical_frame')

if __name__ == '__main__':
    broadcaster = StaticTfBroadcaster()
    rospy.spin()
