from frankx import Affine, LinearRelativeMotion, Robot, LinearMotion, InvalidOperationException


class FrankaRobot:
    def __init__(self, id="192.168.5.12"):
        self.robot = Robot(id)
        self.gripper = self.robot.get_gripper()
        self.robot.set_default_behavior()
        self.robot.velocity_rel = 1
        self.robot.acceleration_rel = 0.5
        self.robot.jerk_rel = 0.01
        self.robot.recover_from_errors()

    def move_to_init(self, initial_pos):
        self.robot.move(LinearMotion(Affine(*initial_pos, 0, 0, 0)))
        # self.robot.move(LinearMotion(Affine(0, 0, -0.03, 0, 0, 0)))

    def clamp(self):
        self.gripper.clamp()

    def release(self, disp=0.05):
        self.gripper.release(disp)

    def move(self, displacement):
        self.robot.move(LinearRelativeMotion(Affine(*displacement)))

    def current_pose(self):
        return self.robot.current_pose().vector()

    def current_joint_state(self):
        # Get the current state handling the read exception when the robot is in motion
        try:
            robot_state = self.robot.get_state(read_once=True)
        except InvalidOperationException:
            robot_state = self.robot.get_state(read_once=False)
        joint_pose = robot_state.q
        joint_vel = robot_state.dq
        return joint_pose, joint_vel


if __name__ == "__main__":
    robot = FrankaRobot()
    cur_pose = robot.current_pose()
    cur_joint_pose, cur_joint_vel = robot.current_joint_state()

    print(cur_pose)
    print(cur_joint_pose)
    print(cur_joint_vel)
