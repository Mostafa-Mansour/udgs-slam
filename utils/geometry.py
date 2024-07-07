

import torch


R_acc_cam = torch.zeros((3, 3), device='cuda')
R_acc_cam[0,1] = -1
R_acc_cam[1,2] = 1
R_acc_cam[2,0] = -1


def acc2cam(acc_data):
    return torch.matmul(R_acc_cam.double(), acc_data)


def quat2rot(quat_data):
    w, x, y, z = quat_data
    # Compute the rotation matrix using the quaternion values
    R = torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ], dtype=torch.float32, device='cuda')
    return R

def rot2quat(R):
    m00 = R[0, 0]
    m01 = R[0, 1]
    m02 = R[0, 2]
    m10 = R[1, 0]
    m11 = R[1, 1]
    m12 = R[1, 2]
    m20 = R[2, 0]
    m21 = R[2, 1]
    m22 = R[2, 2]

    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * torch.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * torch.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return torch.tensor([w, x, y, z], device='cuda')


def remove_gravity(acc_data, R):
    # Gravity vector in the world frame (assuming g = 9.81 m/s^2)
    g_world = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64, device='cuda')

    # Transform the gravity vector to the sensor frame
    g_sensor = torch.matmul(R, g_world)

    # Subtract the gravity vector from the accelerometer data
    linear_acc = acc_data - g_sensor
    return linear_acc
    
    