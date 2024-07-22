#!/usr/bin/env python3
import rospy
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from sensor_msgs.msg import JointState
import numpy as np
from copy import copy
from gripper_controller_implementacion import GripperController

cos = np.cos
sin = np.sin
pi = np.pi

# Funciones de Cinemática
def dh(d, theta, a, alpha):
    cth = cos(theta)
    sth = sin(theta)
    ca = cos(alpha)
    sa = sin(alpha)
    T = np.array([[cth, -ca * sth, sa * sth, a * cth],
                  [sth, ca * cth, -sa * cth, a * sth],
                  [0, sa, ca, d],
                  [0, 0, 0, 1]])
    return T

def fkine_open_manipulator(q):
    beta = 11 * pi / 180
    T1 = dh(0.077, q[0] + pi, 0, pi / 2)
    T2 = dh(0, q[1] + pi / 2 + beta, 0.13, 0)
    T3 = dh(0, q[2] + pi / 2 - beta, 0.124, 0)
    T4 = dh(0, q[3], 0.126, 0)
    T = T1.dot(T2).dot(T3).dot(T4)
    return T

def jacobian_open_manipulator(q, delta=0.0001):
    J = np.zeros((3, 4))
    T = fkine_open_manipulator(q)
    for i in range(4):
        dq = copy(q)
        dq[i] += delta
        dT = fkine_open_manipulator(dq)
        J[0, i] = (dT[0, 3] - T[0, 3]) / delta
        J[1, i] = (dT[1, 3] - T[1, 3]) / delta
        J[2, i] = (dT[2, 3] - T[2, 3]) / delta
    return J

def ik_gradient_open_manipulator(xdes, q0, fixed_joint_index=None, fixed_joint_value=None):
    epsilon = 0.001
    max_iter = 1000
    delta = 0.00001
    alfa = 0.5
    q = copy(q0)
    for i in range(max_iter):
        if fixed_joint_index is not None:
            q[fixed_joint_index] = fixed_joint_value
        T = fkine_open_manipulator(q)
        x = T[0:3, 3]
        if np.linalg.norm(x - xdes) < epsilon:
            break
        J = jacobian_open_manipulator(q, delta)
        e = xdes - x
        delta_q = np.linalg.pinv(J).dot(e)
        q += alfa * delta_q
    return q

class OpenManipulatorController:
    def __init__(self):
        rospy.wait_for_service('/goal_joint_space_path')
        self.set_joint_position = rospy.ServiceProxy('/goal_joint_space_path', SetJointPosition)
        rospy.Subscriber('/open_manipulator/joint_states', JointState, self.joint_state_callback)
        rospy.Subscriber('/object_position', Point, self.object_position_callback)
        self.joint_states = None
        self.object_position = None

    def joint_state_callback(self, msg):
        self.joint_states = msg

    def object_position_callback(self, msg):
        self.object_position = msg
        self.move_to_position(msg.x, msg.y, msg.z)

    def move_to_joint_positions(self, positions):
        req = SetJointPositionRequest()
        req.joint_position.joint_name = ["joint1", "joint2", "joint3", "joint4"]
        req.joint_position.position = positions
        req.path_time = 2.0
        try:
            resp = self.set_joint_position(req)
            return resp
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def get_current_joint_states(self):
        return self.joint_states.position if self.joint_states else None

    def move_to_position(self, x, y, z, fixed_joint_index=None, fixed_joint_value=None):
        xdes = np.array([x, y, z])
        if self.joint_states:
            q0 = np.array(self.joint_states.position)
        else:
            q0 = np.zeros(4)

        q_des = ik_gradient_open_manipulator(xdes, q0, fixed_joint_index, fixed_joint_value)

        rospy.loginfo(f"Posición deseada: {xdes}")
        rospy.loginfo(f"Ángulos calculados de las articulaciones: {q_des}")

        result = self.move_to_joint_positions(q_des)
        if result:
            rospy.loginfo("Move to position successful.")
        else:
            rospy.logwarn("Move to position failed.")

if __name__ == '__main__':
    rospy.init_node('object_manipulation_controller', anonymous=True)
    object_manipulation_controller = OpenManipulatorController()
    gripper_controller = GripperController()
    gripper_controller.open_gripper()
    rospy.sleep(1)

    rospy.spin()
