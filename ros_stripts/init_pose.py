#! /usr/bin/env python
import sys
import rospy
import moveit_commander
import geometry_msgs
import tf

 
moveit_commander.roscpp_initializer.roscpp_initialize(sys.argv)
rospy.init_node('reset_pose', anonymous=True)
robot = moveit_commander.robot.RobotCommander()

arm_group = moveit_commander.move_group.MoveGroupCommander("arm")
joint_state_positions = arm_group.get_current_joint_values()
print (str(joint_state_positions))

arm_group.set_joint_value_target([-1.571440978454157, 0.3895226046760191, -1.1851060388291423, 1.620664367977247, -1.5722788537280565, -0.08912153872210177])
# arm_group.go(wait=True)

cp = arm_group.get_current_pose()
print(cp)

cp.pose.orientation.x = -0.707
cp.pose.orientation.y = 0.707
cp.pose.orientation.z = 0.0
cp.pose.orientation.w = 0.0

arm_group.set_pose_target(cp.pose)
#arm_group.go()

moveit_commander.roscpp_initializer.roscpp_shutdown()
