import depthai as dai

print("Available DepthAI devices:")
for d in dai.Device.getAllAvailableDevices():
    print("MXID:", d.getMxId(), "| State:", d.state)
