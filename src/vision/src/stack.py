import rospy
import intera_interface
from intera_interface import gripper as robot_gripper

def main():
    print("starting main")
    # Initialize ROS node
    rospy.init_node('sawyer_stack')

    # Initialize Sawyer limb interface
    print("initing limbs")
    limb = intera_interface.Limb('right')  # replace with 'left' if you are using the left arm
    gripper = robot_gripper.Gripper('right_gripper')

    # Define positions (replace with positions in S frame)
    print("defining pos")
    pickup_position = {'right_j0': 0.0, 'right_j1': 0.0, 'right_j2': 0.0, 'right_j3': 0.0, 'right_j4': 0.0, 'right_j5': 0.0, 'right_j6': 0.0}
    place_position = {'right_j0': 1.0, 'right_j1': 1.0, 'right_j2': 1.0, 'right_j3': 1.0, 'right_j4': 1.0, 'right_j5': 1.0, 'right_j6': 1.0}

    # Pick up object
    print("moving to pos")
    limb.move_to_joint_positions(pickup_position)

    # TODO: Add code for closing gripper to pick up the object
    print("closing gripper")
    gripper.close()
    rospy.sleep(1)

    # Move to place position
    print("moving to pos")
    limb.move_to_joint_positions(place_position)
    rospy.sleep(1)

    # TODO: Add code for opening gripper to place the object
    print("opening gripper")
    gripper.open()
    rospy.sleep(1)

if __name__ == '__main__':
    main()
