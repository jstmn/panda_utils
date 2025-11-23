import rospy
from deoxys.franka_interface import FrankaInterface


def cm_to_m(cm: float) -> float:
    return cm / 100.0


def to_public_dict(obj):
    """Recursively convert an object to a dict, skipping private attributes."""
    # Base cases: primitives
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_public_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_public_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: to_public_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if hasattr(obj, "__slots__"):
        return {
            slot: to_public_dict(getattr(obj, slot))
            for slot in obj.__slots__
            if hasattr(obj, slot) and not slot.startswith("_")
        }
    # Fallback to dir()
    result = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        value = getattr(obj, attr)
        if callable(value):
            continue
        result[attr] = to_public_dict(value)
    return result


def wait_for_deoxys_ready(robot_interface: FrankaInterface) -> bool:
    rate = rospy.Rate(50)

    failed_counter = 0
    while not rospy.is_shutdown():
        rate.sleep()
        arm_q = robot_interface.last_q
        last_gripper_q = robot_interface.last_gripper_q

        # Check if the arm q or gripper angle is available
        if arm_q is None or last_gripper_q is None:
            if failed_counter % 10 == 0:
                rospy.logwarn(
                    f"Arm q or gripper angle isn't available. Is deoxys running on the RT pc? last_gripper_q: {last_gripper_q}, arm_q={arm_q}"
                )
            failed_counter += 1
            if failed_counter > 100:
                rospy.logerr("Failed to get arm q or gripper angle for 100 consecutive times. Shutting down.")
                rospy.signal_shutdown("Failed to get arm q or gripper angle for 100 consecutive times")
                return False
            continue

        return True

    return False
