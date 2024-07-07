import torch
import cv2
import numpy as np
class EKF:
    
    def __init__(self, q=0.03, r=0.1, initial_time_stamp=0.0, acc_data_array=None):
        
        # Process and measurement noise covariances
        var_position = 0.8 # Variance for position components
        var_velocity = 0.01 # Variance for velocity components
        var_quaternion = 0.1 # Variance for quaternion components
        # Construct the process covariance matrix Q
        
        self.Q = torch.diag(torch.tensor([
        var_position, var_position, var_position, # Position components
        var_velocity, var_velocity, var_velocity, # Velocity components
        var_quaternion, var_quaternion, var_quaternion, var_quaternion # Quaternion components
        ], dtype=torch.float32, device='cuda'))
        
        
        """
        self.Q = torch.eye(10, dtype=torch.float32, device='cuda')
        self.Q[0,0] = q/3
        self.Q[1,1] = q/3
        self.Q[2,2] = q/3
        self.Q[3,3] = q
        self.Q[4,4] = q
        self.Q[5,5] = q
        self.Q[6,6] = var_quaternion
        self.Q[7,7] = var_quaternion
        self.Q[8,8] = var_quaternion
        self.Q[9,9] = var_quaternion
        
        self.Q[0,3] = q/2
        self.Q[1,4] = q/2
        self.Q[2,5] = q/2
        
        self.Q[3,0] = q/2
        self.Q[4,1] = q/2
        self.Q[5,2] = q/2
        """
        #print(f"Q matrix is {self.Q}")
        
        
        
        
        
        
        
        var_position_meas = 0.3
        var_orientation_meas = 0.5
        self.R = torch.diag(torch.tensor([
        var_position_meas, var_position_meas, var_position_meas, # Position components
        var_orientation_meas, var_orientation_meas, var_orientation_meas, var_orientation_meas # Quaternion components
        ], dtype=torch.float32, device='cuda'))
        
        self.previous_time = torch.tensor(initial_time_stamp, dtype=torch.float64, device='cuda')
        self.acc_data = torch.tensor(acc_data_array) if acc_data_array is not None else None
        self.x = None
        self.P = None
        
        self.previous_image = None
        
    
    def state_transition(self, x, acc, dt):
        p, v, q = x[:3], x[3:6], x[6:]
        a = acc
        
        # Update position and velocity using accelerometer data
        p_new = p + v * dt + 0.5 * a * dt**2
        v_new = v + a * dt

        # Orientation remains the same (no gyroscope data)
        q_new = q
        
        return torch.cat((p_new, v_new, q_new))
    
    # Define the measurement function
    def measurement_function(self, x):
        pos = x[:3]
        orientation = x[6:]
        return torch.cat((pos, orientation), dim=0)  # Extract position and orientation (no velocity in measurement)

    # Define the Jacobians
    def jacobian_F(self, x, accel, dt):
        F = torch.eye(10).to('cuda')
        F[0:3, 3:6] = torch.eye(3).to('cuda') * dt
        F[0:3, 6:9] = 0.5 * dt**2 * torch.eye(3).to('cuda')
        F[3:6, 6:9] = dt * torch.eye(3).to('cuda')
        return F

    def jacobian_H(self):
        H = torch.zeros((7, 10)).to('cuda')
        H[:3, :3] = torch.eye(3).to('cuda')  # Position part
        H[3:, 6:] = torch.eye(4).to('cuda')  # Orientation part (assuming quaternion here)
        return H
    
    def update_Q(self, dt):
        
        Q = self.Q.clone()
        
        Q[0,0] *= dt**3
        Q[1,1] *= dt**3
        Q[2,2] *= dt**3
        Q[3,3] *= dt
        Q[4,4] *= dt
        Q[5,5] *= dt
        
        Q[0,3] *= dt**2
        Q[1,4] *= dt**2
        Q[2,5] *= dt**2
        
        Q[3,0] *= dt**2
        Q[4,1] *= dt**2
        Q[5,2] *= dt**2
        
        Q[6,6] *= dt**2
        Q[7,7] *= dt**2
        Q[8,8] *= dt**2
        Q[9,9] *= dt**2
        
        return Q
        
        
        
    def predict(self, x, P, acc, dt):
        x_pred = self.state_transition(x, acc, dt)
        F = self.jacobian_F(x, acc, dt)
        Q = self.update_Q(dt)
        P_pred = F @ P @ F.T + self.Q
        return x_pred, P_pred
    
    def update(self, z, x_pred, P_pred):
        H = self.jacobian_H()
        residual = z - self.measurement_function(x_pred)
        
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ torch.linalg.inv(S)
        x = x_pred + K.double() @ residual
        P = (torch.eye(10).to('cuda') - K @ H) @ P_pred
        return x, P
    
    def get_acc_data(self, time_stamp):
        # Get the column of timestamps
        timestamps = self.acc_data[:, 0]
    
        # Find the indices of the rows where the timestamps fall between the start and end times
        start_index = torch.searchsorted(timestamps, torch.tensor(self.previous_time).clone(), side='left')
        end_index = torch.searchsorted(timestamps, torch.tensor(time_stamp).clone(), side='right')
        
        # Slice the tensor and return the result
        return self.acc_data[start_index:end_index].clone()
    
    def calculate_rotation(self, img):
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        
        img_1 = self.previous_image.astype(np.float32)
        img_1 = cv2.cvtColor(img_1.transpose((1,2,0)), cv2.COLOR_RGB2GRAY)
        img_1 = (img_1 * 255).astype(np.uint8)
        
        img_2 = img.cpu().numpy().astype(np.float32)
        img_2 = cv2.cvtColor(img_2.transpose((1,2,0)), cv2.COLOR_RGB2GRAY)
        img_2 = (img_2 * 255).astype(np.uint8)
        # Find keypoints and descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img_1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img_2, None)

        # Convert keypoints to NumPy arrays3
        keypoints1_np = np.float32([kp.pt for kp in keypoints1])
        keypoints2_np = np.float32([kp.pt for kp in keypoints2])
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Extract matched keypoints
        matched_points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        matched_points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
        # Estimate rotation using RANSAC
        rotation_matrix, _ = cv2.estimateAffinePartial2D(matched_points1, matched_points2, method=cv2.RANSAC)

        # Extract the 2x2 rotation and scaling part of the affine matrix
        rotation_part = rotation_matrix[:2, :2]
        
        # Ensure it is a proper rotation matrix by normalizing using SVD
        u, _, vh = np.linalg.svd(rotation_part, full_matrices=True)
        rotation_matrix_2x2 = np.dot(u, vh)
        
        # Convert to 3x3 rotation matrix
        rotation_matrix_3x3 = np.eye(3)
        rotation_matrix_3x3[:2, :2] = rotation_matrix_2x2
        
        # Convert to PyTorch tensor
        rotation_matrix_tensor = torch.from_numpy(rotation_matrix_3x3).to('cuda')
        # Convert rotation matrix to PyTorch tensor
        #rotation_matrix_tensor = torch.from_numpy(rotation_matrix)
        return rotation_matrix_tensor