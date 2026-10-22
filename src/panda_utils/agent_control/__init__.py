"""Agent (LLM) control of the Panda: camera images in, structured end effector pose deltas out.

Modules are imported individually; only the script pulls in rclpy, and only EEPoseServo pulls in deoxys:
    actions   EEPoseDelta (the structured action the model returns), applying it in the world or end effector
              frame with clamping, and EEPoseServo (closed-loop pose tracking on OSC_POSE_PRECISE, our deoxys fork)
    agent     PoseDeltaAgent, the model call and observation history
    viewer    AgentViewer, the browser viewer (viser)
Cameras come from panda_utils.ros2_cameras.
"""
