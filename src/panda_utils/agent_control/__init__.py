"""Agent (LLM) control of the Panda: camera images in, structured end effector pose deltas out.

Modules are imported individually so the light ones (schema, geometry) do not pull in rclpy or deoxys:
    schema    EEPoseDelta, the structured action the model returns
    geometry  applying deltas in the world or end effector frame, clamping, drawing the EE frame on images
    agent     PoseDeltaAgent, the Claude call and observation history
    servo     EEPoseServo, closed-loop OSC_POSE tracking of a target pose via deoxys
Cameras come from panda_utils.ros2_cameras.
"""
