#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tf
import sys
import rospy
import moveit_commander
from control_msgs.msg import GripperCommandActionGoal
from geometry_msgs.msg import PoseStamped, Pose
# from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse
import os

class MoveRobot():
    def __init__(self):
        # Initialize planning group
        self.robot = moveit_commander.robot.RobotCommander()
        self.arm_group = moveit_commander.move_group.MoveGroupCommander("arm")
        self.gripper_group = moveit_commander.move_group.MoveGroupCommander("gripper")

        # Set the speed and acceleration of the robotic arm
        self.arm_group.set_max_acceleration_scaling_factor(1)
        self.arm_group.set_max_velocity_scaling_factor(1)

        # Object suction interface

        # self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
        # self.attach_srv.wait_for_service()
        # self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
        # self.detach_srv.wait_for_service()

        # Object position
        self.Obj_pose = PoseStamped()
        self.Obj_pose.pose.position.x = 0
        self.find_enable = False

        self.obj_pose_sub = rospy.Subscriber("/objection_position_pose", Pose, self.pose_callback)

    def read_gg_values(self, filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()

        width = None
        for line in lines:
            if 'width:' in line:
                width = float(line.split('width:')[1].split(',')[0].strip())
                break  # Exit the loop once found
        return width

    def pose_callback(self, msg):
        if self.find_enable:
            self.Obj_pose.pose = msg
            print(self.Obj_pose)

        if self.Obj_pose.pose.position.x != 0:
            self.find_enable = False

    def stop(self):
        moveit_commander.roscpp_initializer.roscpp_shutdown()

    def gripperMove(self, width):
        gripper_joints_state = self.gripper_group.get_current_joint_values()
        gripper_joints_state[2] = width
        self.gripper_group.set_joint_value_target(gripper_joints_state)
        self.gripper_group.go()

    def plan_cartesian_path(self, pose):
        """
        Cartesian path planning

        Parameters:
            pose - Target pose

        Returns:
            None
        """
        waypoints = []
        waypoints.append(pose)

        # Set the current state as the start state
        self.arm_group.set_start_state_to_current_state()

        # Compute the trajectory
        (plan, fraction) = self.arm_group.compute_cartesian_path(
            waypoints,   # Waypoint poses
            0.01,        # eef_step - End-effector step size
            0.0,         # jump_threshold - Jump threshold
            False)       # avoid_collisions - Collision avoidance

        self.arm_group.execute(plan, wait=True)

    # def gazeboAttach(self, i):
    #     rospy.loginfo("Attaching gripper and object")
    #     req = AttachRequest()
    #     req.model_name_1 = "my_gen3_lite"
    #     req.link_name_1 = "end_effector_link"
    #     if (i == 'coke'):
    #         req.model_name_2 = "coke_can"
    #         req.link_name_2 = "base_link"
    #     if (i == 'salt'):
    #         req.model_name_2 = "salt"
    #         req.link_name_2 = "salt_link"   
    #     if (i == 'banana'):
    #         req.model_name_2 = "banana"
    #         req.link_name_2 = "link_0"   
    #     self.attach_srv.call(req)

    # def gazeboDetach(self, i):
    #     rospy.loginfo("Detaching gripper and object")
    #     req = AttachRequest()
    #     req.model_name_1 = "my_gen3_lite"
    #     req.link_name_1 = "end_effector_link"
    #     if (i == 'coke'):
    #         req.model_name_2 = "coke_can"
    #         req.link_name_2 = "base_link"
    #     if (i == 'salt'):
    #         req.model_name_2 = "salt"
    #         req.link_name_2 = "salt_link"   
    #     if (i == 'banana'):
    #         req.model_name_2 = "banana"
    #         req.link_name_2 = "link_0"   
    #     self.detach_srv.call(req)

    def goSP(self):
        self.arm_group.set_joint_value_target([-0.08498747219394814, -0.2794001977631106, 0.7484180883797364, -1.570090066123494, -2.114137663337607, -1.6563429070772748])
        self.arm_group.go(wait=True)

    def grasp_obj(self):
        # Move to the position above the object
        print(self.Obj_pose)

        self.Obj_pose.pose.position.x -= 0.055
        self.Obj_pose.pose.position.z += 0.05

        self.arm_group.set_pose_target(self.Obj_pose.pose)
        self.arm_group.go()
        self.plan_cartesian_path(self.Obj_pose.pose)

        if self.obj_class == 'big':
            self.Obj_pose.pose.position.z -= 0.09
        elif self.obj_class == 'start':
            self.Obj_pose.pose.position.z -= 0.075
        elif self.obj_class == 'small':
            self.Obj_pose.pose.position.z -= 0.06

        self.plan_cartesian_path(self.Obj_pose.pose)
        self.arm_group.set_pose_target(self.Obj_pose.pose)
        self.arm_group.go()

    def place_obj(self, pose):
        pose.pose.position.z += 0.07
        self.plan_cartesian_path(pose.pose)

        # if self.obj_class == 'coke':
        self.arm_group.set_joint_value_target([-1.1923993012061151, 0.7290586635521652, -0.7288901499177471, 1.6194515338395425, -1.6699862200379725, 0.295133228129065])
        self.arm_group.go()

        # else:
        #     self.arm_group.set_joint_value_target([-1.843368103280336, -0.3940365853740246, 1.1440979734990149, 1.5241253776209207, 1.5848037621254936, -0.3618106449571939])
        #     self.arm_group.go()

        self.gripperMove(0.6)
        # self.gazeboDetach(self.obj_class)

    def main_loop(self):
        try:
            self.goSP()  # 1. Go to the pre-grasp position
            self.gripperMove(0.6)

            rospy.loginfo("Arrived at start point")
            self.obj_class = str(raw_input("Input obj class:="))

            self.find_enable = True
            rospy.sleep(3)  # 2. Identify the current grasp pose (3s)
            if not self.find_enable:
                rospy.loginfo('Find object, start grasp')

                # 4. Grasp the object
                self.grasp_obj()
                filepath = '/home/kyhb/catkin_ws/data/gg_values.txt'
                width = self.read_gg_values(filepath)
                # 5. Close the gripper
                # if self.obj_class == 'big':
                #     self.gripperMove(0.3)
                # elif self.obj_class == 'medium':
                #     self.gripperMove(0.2)
                # else:
                #     self.gripperMove(0.1)
                # self.gazeboAttach(self.obj_class)
                self.gripperMove(3.6 * width)
                # # 6. Place the object
                self.place_obj(self.Obj_pose)
                self.Obj_pose.pose.position.x = 0
            else:
                rospy.logwarn('Cannot find object')

        except Exception as e:
            rospy.logerr(str(e))

def main():
    rospy.init_node('grasp_demo', anonymous=True)
    rospy.loginfo('Start Grasp Demo')
    moverobot = MoveRobot()
    while not rospy.is_shutdown():
        moverobot.main_loop()

if __name__ == "__main__":
    main()
