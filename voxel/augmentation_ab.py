import numpy as np
import torch
from helpers import utils
from pytorch3d import transforms as torch3d_tf
import time
import os


def perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds):
    """Perturb point clouds with given transformation.
    :param pcd: list of point clouds [[bs, 3, N], ...] for N cameras
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds
    """
    # baatch bounds if necessary
    bs = pcd[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_pcd = []
    for p in pcd:
        p_shape = p.shape
        num_points = p_shape[-1] * p_shape[-2]

        action_trans_3x1 = (
            action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        )
        trans_shift_3x1 = (
            trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        )

        # flatten point cloud
        p_flat = p.reshape(bs, 3, -1)
        p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)

        # shift points to have action_gripper pose as the origin
        p_flat_4x1_action_origin[:, :3, :] = p_flat - action_trans_3x1

        # apply rotation
        perturbed_p_flat_4x1_action_origin = torch.bmm(
            p_flat_4x1_action_origin.transpose(2, 1), rot_shift_4x4
        ).transpose(2, 1)

        # apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(
            action_then_trans_3x1[:, 0], min=bounds_x_min, max=bounds_x_max
        )
        action_then_trans_3x1_y = torch.clamp(
            action_then_trans_3x1[:, 1], min=bounds_y_min, max=bounds_y_max
        )
        action_then_trans_3x1_z = torch.clamp(
            action_then_trans_3x1[:, 2], min=bounds_z_min, max=bounds_z_max
        )
        action_then_trans_3x1 = torch.stack(
            [action_then_trans_3x1_x, action_then_trans_3x1_y, action_then_trans_3x1_z],
            dim=1,
        )

        # shift back the origin
        perturbed_p_flat_3x1 = (
            perturbed_p_flat_4x1_action_origin[:, :3, :] + action_then_trans_3x1
        )

        perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
        perturbed_pcd.append(perturbed_p)
    return perturbed_pcd



def bimanual_apply_se3_augmentation(
    pcd,
    right_action_gripper_pose,
    right_action_trans,
    right_action_rot_grip,
    left_action_gripper_pose,
    left_action_trans,
    left_action_rot_grip,
    bounds,
    layer,
    trans_aug_range,
    rot_aug_range,
    rot_aug_resolution,
    voxel_size,
    rot_resolution,
    device,
):
    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose

    # center_action = (right_action_gripper_pose[:, :3] + left_action_gripper_pose[:, :3]) / 2
    center_action = right_action_gripper_pose[:, :3]
    # center_action = left_action_gripper_pose[:, :3]
    # center_action = torch.tensor([[0.0, 0.0, 0.0]]).to(device=device)

    right_action_gripper_trans = right_action_gripper_pose[:, :3]
    right_action_gripper_quat_wxyz = torch.cat(
        (
            right_action_gripper_pose[:, 6].unsqueeze(1),
            right_action_gripper_pose[:, 3:6],
        ),
        dim=1,
    )

    right_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
        right_action_gripper_quat_wxyz
    )
    right_action_gripper_4x4 = identity_4x4.detach().clone()
    right_action_gripper_4x4[:, :3, :3] = right_action_gripper_rot
    right_action_gripper_4x4[:, 0:3, 3] = right_action_gripper_trans

    right_perturbed_trans = torch.full_like(right_action_trans, -1.0)
    right_perturbed_rot_grip = torch.full_like(right_action_rot_grip, -1.0)

    left_action_gripper_trans = left_action_gripper_pose[:, :3]
    left_action_gripper_quat_wxyz = torch.cat(
        (left_action_gripper_pose[:, 6].unsqueeze(1), left_action_gripper_pose[:, 3:6]),
        dim=1,
    )

    left_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
        left_action_gripper_quat_wxyz
    )
    left_action_gripper_4x4 = identity_4x4.detach().clone()
    left_action_gripper_4x4[:, :3, :3] = left_action_gripper_rot
    left_action_gripper_4x4[:, 0:3, 3] = left_action_gripper_trans

    left_perturbed_trans = torch.full_like(left_action_trans, -1.0)
    left_perturbed_rot_grip = torch.full_like(left_action_rot_grip, -1.0)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    # while torch.any(right_perturbed_trans < 0) and torch.any(left_perturbed_trans < 0):
    while torch.any(right_perturbed_trans < 0) or torch.any(left_perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 100:
            return (
                right_action_trans,
                right_action_rot_grip,
                left_action_trans,
                left_action_rot_grip,
                pcd,
            )
            # raise Exception("Failing to perturb action and keep it within bounds.")

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(
            device=device
        )
        trans_shift = trans_range * utils.rand_dist((bs, 3)).to(device=device)
        # for debugging
        # trans_shift = torch.tensor([[0.125, 0.125, 0.125]]).to(device=device)
        # trans_shift = torch.tensor([[0.0, 0.0, 0.0]]).to(device=device)

        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = utils.rand_discrete(
            (bs, 1), min=-roll_aug_steps, max=roll_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        pitch = utils.rand_discrete(
            (bs, 1), min=-pitch_aug_steps, max=pitch_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        yaw = utils.rand_discrete(
            (bs, 1), min=-yaw_aug_steps, max=yaw_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        # for debugging
        # yaw = torch.tensor([[45.0]]).to(device=device)
        # yaw = torch.tensor([[0.0]]).to(device=device)

        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
            torch.cat((roll, pitch, yaw), dim=1), "XYZ"
        ).to(device=device)
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        def transform(matrix, C, translation, rotation_matrix):
            """
            Rotate and translate a 4x4 gripper pose matrix around the center C.
            TBD.

            Args:
                matrix: 4x4 matrix to be transformed. tensor.
                C: 3 Center of rotation. tensor.
                translation: 3 Translation vector. tensor.
                rotation_matrix: 3x3 rotation matrix. tensor.
            """
            # print(rotation_matrix.shape)
            matrix[:, 0:3, 3] -= C
            matrix[:, 0:3, 3] = torch.matmul(matrix[:, 0:3, 3].unsqueeze(1), rotation_matrix).squeeze(1)
            matrix[:, 0:3, 3] += C
            matrix[:, 0:3, 3] += translation

            matrix[:, :3, :3] = torch.matmul(matrix[:, :3, :3], rotation_matrix)

            # print(matrix.shape)
            return matrix

        # rotate then translate the 4x4 keyframe action
        right_perturbed_action_gripper_4x4 = torch.bmm(
            right_action_gripper_4x4, rot_shift_4x4
        )
        right_perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # right_perturbed_action_gripper_4x4 = right_action_gripper_4x4.detach().clone()
        # right_perturbed_action_gripper_4x4[:, 0:3, 3] -= center_action
        # right_perturbed_action_gripper_4x4 = torch.bmm(
        #     right_perturbed_action_gripper_4x4, rot_shift_4x4
        # )
        # right_perturbed_action_gripper_4x4[:, 0:3, 3] += center_action
        # right_perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # right_perturbed_action_gripper_4x4 = transform(
        #     right_action_gripper_4x4.detach().clone(),
        #     center_action,
        #     trans_shift,
        #     rot_shift_3x3,
        # )

        # convert transformation matrix to translation + quaternion
        right_perturbed_action_trans = (
            right_perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        )
        right_perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            right_perturbed_action_gripper_4x4[:, :3, :3]
        )
        right_perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    right_perturbed_action_quat_wxyz[:, 1:],
                    right_perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )
        
        # left_perturbed_action_gripper_4x4 = left_action_gripper_4x4.detach().clone().squeeze()  # [4, 4]
        # left_perturbed_action_gripper_4x4 = transform(
        #     left_perturbed_action_gripper_4x4,
        #     center_action.squeeze(),
        #     trans_shift.squeeze(),
        #     rot_shift_3x3.squeeze()
        # )

        left_perturbed_action_gripper_4x4 = left_action_gripper_4x4.detach().clone()  # [4, 4]
        left_perturbed_action_gripper_4x4 = transform(
            left_perturbed_action_gripper_4x4,
            center_action,
            trans_shift,
            rot_shift_3x3
        )

        # convert transformation matrix to translation + quaternion
        # print(left_perturbed_action_gripper_4x4.shape)
        left_perturbed_action_trans = (
            left_perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        )

        # print(left_perturbed_action_gripper_4x4.shape)
        left_perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            left_perturbed_action_gripper_4x4[:, :3, :3]
        )
        left_perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    left_perturbed_action_quat_wxyz[:, 1:],
                    left_perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        right_trans_indicies, right_rot_grip_indicies = [], []
        left_trans_indicies, left_rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            right_trans_idx = utils.point_to_voxel_index(
                right_perturbed_action_trans[b], voxel_size, bounds_np
            )
            right_trans_indicies.append(right_trans_idx.tolist())

            right_quat = right_perturbed_action_quat_xyzw[b]
            right_quat = utils.normalize_quaternion(right_perturbed_action_quat_xyzw[b])
            if right_quat[-1] < 0:
                right_quat = -right_quat
            right_disc_rot = utils.quaternion_to_discrete_euler(
                right_quat, rot_resolution
            )
            right_rot_grip_indicies.append(
                right_disc_rot.tolist()
                + [int(right_action_rot_grip[b, 3].cpu().numpy())]
            )

            left_trans_idx = utils.point_to_voxel_index(
                left_perturbed_action_trans[b], voxel_size, bounds_np
            )
            left_trans_indicies.append(left_trans_idx.tolist())

            left_quat = left_perturbed_action_quat_xyzw[b]
            left_quat = utils.normalize_quaternion(left_perturbed_action_quat_xyzw[b])
            if left_quat[-1] < 0:
                left_quat = -left_quat
            left_disc_rot = utils.quaternion_to_discrete_euler(
                left_quat, rot_resolution
            )
            left_rot_grip_indicies.append(
                left_disc_rot.tolist() + [int(left_action_rot_grip[b, 3].cpu().numpy())]
            )

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        right_perturbed_trans = torch.from_numpy(np.array(right_trans_indicies)).to(
            device=device
        )
        right_perturbed_rot_grip = torch.from_numpy(
            np.array(right_rot_grip_indicies)
        ).to(device=device)

        left_perturbed_trans = torch.from_numpy(np.array(left_trans_indicies)).to(
            device=device
        )
        left_perturbed_rot_grip = torch.from_numpy(np.array(left_rot_grip_indicies)).to(
            device=device
        )

    right_action_trans = right_perturbed_trans
    right_action_rot_grip = right_perturbed_rot_grip

    left_action_trans = left_perturbed_trans
    left_action_rot_grip = left_perturbed_rot_grip

    # apply perturbation to pointclouds
    # pcd = bimanual_perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, right_action_gripper_4x4, left_action_gripper_4x4, bounds)

    center_action_gripper_4x4 = identity_4x4.detach().clone()
    # center_action_gripper_4x4 = torch.bmm(center_action_gripper_4x4, rot_shift_4x4)
    # center_action_gripper_4x4 = rot_shift_4x4
    center_action_gripper_4x4[:, 0:3, 3] += center_action
    pcd = perturb_se3(
        pcd, trans_shift_4x4, rot_shift_4x4, center_action_gripper_4x4, bounds
    )

    return (
        right_action_trans,
        right_action_rot_grip,
        left_action_trans,
        left_action_rot_grip,
        pcd,
    )



def original_bimanual_apply_se3_augmentation(
    pcd,
    right_action_gripper_pose,
    right_action_trans,
    right_action_rot_grip,
    left_action_gripper_pose,
    left_action_trans,
    left_action_rot_grip,
    bounds,
    layer,
    trans_aug_range,
    rot_aug_range,
    rot_aug_resolution,
    voxel_size,
    rot_resolution,
    device,
):
    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    right_action_gripper_trans = right_action_gripper_pose[:, :3]
    right_action_gripper_quat_wxyz = torch.cat(
        (
            right_action_gripper_pose[:, 6].unsqueeze(1),
            right_action_gripper_pose[:, 3:6],
        ),
        dim=1,
    )

    right_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
        right_action_gripper_quat_wxyz
    )
    right_action_gripper_4x4 = identity_4x4.detach().clone()
    right_action_gripper_4x4[:, :3, :3] = right_action_gripper_rot
    right_action_gripper_4x4[:, 0:3, 3] = right_action_gripper_trans

    right_perturbed_trans = torch.full_like(right_action_trans, -1.0)
    right_perturbed_rot_grip = torch.full_like(right_action_rot_grip, -1.0)

    left_action_gripper_trans = left_action_gripper_pose[:, :3]
    left_action_gripper_quat_wxyz = torch.cat(
        (left_action_gripper_pose[:, 6].unsqueeze(1), left_action_gripper_pose[:, 3:6]),
        dim=1,
    )

    left_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
        left_action_gripper_quat_wxyz
    )
    left_action_gripper_4x4 = identity_4x4.detach().clone()
    left_action_gripper_4x4[:, :3, :3] = left_action_gripper_rot
    left_action_gripper_4x4[:, 0:3, 3] = left_action_gripper_trans

    left_perturbed_trans = torch.full_like(left_action_trans, -1.0)
    left_perturbed_rot_grip = torch.full_like(left_action_rot_grip, -1.0)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    # while torch.any(right_perturbed_trans < 0) and torch.any(left_perturbed_trans < 0):
    while torch.any(right_perturbed_trans < 0) or torch.any(left_perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 100:
            raise Exception("Failing to perturb action and keep it within bounds.")

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(
            device=device
        )
        # trans_shift = trans_range * utils.rand_dist((bs, 3)).to(device=device)
        # for debugging
        # trans_shift = torch.tensor([[0.125, 0.125, 0.125]]).to(device=device)
        trans_shift = torch.tensor([[0.0, 0.0, 0.0]]).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = utils.rand_discrete(
            (bs, 1), min=-roll_aug_steps, max=roll_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        pitch = utils.rand_discrete(
            (bs, 1), min=-pitch_aug_steps, max=pitch_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        # yaw = utils.rand_discrete(
        #     (bs, 1), min=-yaw_aug_steps, max=yaw_aug_steps
        # ) * np.deg2rad(rot_aug_resolution)
        # for debugging
        yaw = torch.tensor([[45.0]]).to(device=device)

        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
            torch.cat((roll, pitch, yaw), dim=1), "XYZ"
        )
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        # rotate then translate the 4x4 keyframe action
        right_perturbed_action_gripper_4x4 = torch.bmm(
            right_action_gripper_4x4, rot_shift_4x4
        )
        right_perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        right_perturbed_action_trans = (
            right_perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        )
        right_perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            right_perturbed_action_gripper_4x4[:, :3, :3]
        )
        right_perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    right_perturbed_action_quat_wxyz[:, 1:],
                    right_perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        # rotate then translate the 4x4 keyframe action
        left_perturbed_action_gripper_4x4 = torch.bmm(
            left_action_gripper_4x4, rot_shift_4x4
        )
        left_perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        left_perturbed_action_trans = (
            left_perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        )
        left_perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            left_perturbed_action_gripper_4x4[:, :3, :3]
        )
        left_perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    left_perturbed_action_quat_wxyz[:, 1:],
                    left_perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        right_trans_indicies, right_rot_grip_indicies = [], []
        left_trans_indicies, left_rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            right_trans_idx = utils.point_to_voxel_index(
                right_perturbed_action_trans[b], voxel_size, bounds_np
            )
            right_trans_indicies.append(right_trans_idx.tolist())

            right_quat = right_perturbed_action_quat_xyzw[b]
            right_quat = utils.normalize_quaternion(right_perturbed_action_quat_xyzw[b])
            if right_quat[-1] < 0:
                right_quat = -right_quat
            right_disc_rot = utils.quaternion_to_discrete_euler(
                right_quat, rot_resolution
            )
            right_rot_grip_indicies.append(
                right_disc_rot.tolist()
                + [int(right_action_rot_grip[b, 3].cpu().numpy())]
            )

            left_trans_idx = utils.point_to_voxel_index(
                left_perturbed_action_trans[b], voxel_size, bounds_np
            )
            left_trans_indicies.append(left_trans_idx.tolist())

            left_quat = left_perturbed_action_quat_xyzw[b]
            left_quat = utils.normalize_quaternion(left_perturbed_action_quat_xyzw[b])
            if left_quat[-1] < 0:
                left_quat = -left_quat
            left_disc_rot = utils.quaternion_to_discrete_euler(
                left_quat, rot_resolution
            )
            left_rot_grip_indicies.append(
                left_disc_rot.tolist() + [int(left_action_rot_grip[b, 3].cpu().numpy())]
            )

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        right_perturbed_trans = torch.from_numpy(np.array(right_trans_indicies)).to(
            device=device
        )
        right_perturbed_rot_grip = torch.from_numpy(
            np.array(right_rot_grip_indicies)
        ).to(device=device)

        left_perturbed_trans = torch.from_numpy(np.array(left_trans_indicies)).to(
            device=device
        )
        left_perturbed_rot_grip = torch.from_numpy(np.array(left_rot_grip_indicies)).to(
            device=device
        )

    right_action_trans = right_perturbed_trans
    right_action_rot_grip = right_perturbed_rot_grip

    left_action_trans = left_perturbed_trans
    left_action_rot_grip = left_perturbed_rot_grip

    # apply perturbation to pointclouds
    # pcd = bimanual_perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, right_action_gripper_4x4, left_action_gripper_4x4, bounds)

    pcd = perturb_se3(
        pcd, trans_shift_4x4, rot_shift_4x4, right_action_gripper_4x4, bounds
    )

    return (
        right_action_trans,
        right_action_rot_grip,
        left_action_trans,
        left_action_rot_grip,
        pcd,
    )



if __name__ == "__main__":
    from helpers.utils import visualise_voxel, stack_on_channel
    import visdom
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial.transform import Rotation as R
    def convert_to_numpy(pcd_tensor):
        pcd_np = pcd_tensor.squeeze().cpu().numpy() # [B, C, H, W]
        pcd_np = pcd_np.transpose(1, 2, 0) # [H, W, C]
        pcd_np = pcd_np.reshape(-1, 3) # [H*W, C]
        return pcd_np

    def plot_point_cloud(pcd, left_pose, right_pose, title, filename):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot points and end-effector poses
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='b', marker='o', s=1, alpha=0.01, label='Point Cloud')
        ax.scatter(left_pose[:, 0], left_pose[:, 1], left_pose[:, 2], c='r', marker='o', s=40, label='Left Gripper Pose')
        ax.scatter(right_pose[:, 0], right_pose[:, 1], right_pose[:, 2], c='g', marker='o', s=40, label='Right Gripper Pose')

        # Draw left and right gripper axes
        for pose in left_pose:
            trans = pose[:3]
            rot = pose[3:]  # wxyz
            rotation = R.from_quat(rot).as_matrix()
            for i, color in zip(range(3), ['r', 'g', 'b']):
                start = trans
                end = trans + rotation[:, i] * 0.1
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)

        for pose in right_pose:
            trans = pose[:3]
            rot = pose[3:]
            rotation = R.from_quat(rot).as_matrix()
            for i, color in zip(range(3), ['r', 'g', 'b']):
                start = trans
                end = trans + rotation[:, i] * 0.1
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)

        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # legend
        ax.scatter([0], [0], [0], c='y', marker='o', s=10, alpha=1, label='Origin')
        ax.legend()
        # bounding box
        ax.set_xlim([0, 1])
        ax.set_ylim([-1, 0])
        ax.set_zlim([-0.2, 0.8])

        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

    # Sample inputs
    batch_size = 1
    # pcd = [torch.rand(batch_size, 3, 128, 128) for _ in range(1)]

    # numpoints = 128 * 128
    # pcd_flat = torch.zeros(batch_size, 3, numpoints)
    # pcd_flat[0, 0, :] = torch.linspace(-1, 1, numpoints)
    # pcd_flat[0, 1, :] = torch.linspace(-1, 1, numpoints)
    # pcd_flat[0, 2, :] = 0.0
    # pcd_flat = pcd_flat.reshape(batch_size, 3, 128, 128)
    # pcd = [torch.rand(batch_size, 3, 1, 1) for _ in range(1)]
    # pcd = [pcd_flat]
    # right_action_gripper_pose = torch.tensor([[0.8, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]])
    # right_action_trans = right_action_gripper_pose[:, :3]
    # right_action_rot_grip = right_action_gripper_pose[:, 3:]
    # left_action_gripper_pose = torch.tensor([[0.2, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]])
    # left_action_trans = left_action_gripper_pose[:, :3]
    # left_action_rot_grip = left_action_gripper_pose[:, 3:]

    root_path = "/mnt/disk_1/tengbo/peract_bimanual-real/voxel/debug"
    pcd = np.load(os.path.join(root_path, "pcd.npy"))
    pcd = [torch.tensor(pcd)]
    right_action_gripper_pose = np.load(os.path.join(root_path, "right_action_gripper_pose.npy"))
    right_action_trans = np.load(os.path.join(root_path, "right_action_trans.npy"))
    right_action_rot_grip = np.load(os.path.join(root_path, "right_action_rot_grip.npy"))
    left_action_gripper_pose = np.load(os.path.join(root_path, "left_action_gripper_pose.npy"))
    left_action_trans = np.load(os.path.join(root_path, "left_action_trans.npy"))
    left_action_rot_grip = np.load(os.path.join(root_path, "left_action_rot_grip.npy"))

    right_action_gripper_pose = torch.tensor(right_action_gripper_pose)
    right_action_trans = torch.tensor(right_action_trans)
    right_action_rot_grip = torch.tensor(right_action_rot_grip)
    left_action_gripper_pose = torch.tensor(left_action_gripper_pose)
    left_action_trans = torch.tensor(left_action_trans)
    left_action_rot_grip = torch.tensor(left_action_rot_grip)


    # bs > 1 case
    right_action_gripper_pose = torch.cat([right_action_gripper_pose, right_action_gripper_pose], dim=0)
    right_action_trans = torch.cat([right_action_trans, right_action_trans], dim=0)
    right_action_rot_grip = torch.cat([right_action_rot_grip, right_action_rot_grip], dim=0)
    left_action_gripper_pose = torch.cat([left_action_gripper_pose, left_action_gripper_pose], dim=0)
    left_action_trans = torch.cat([left_action_trans, left_action_trans], dim=0)
    left_action_rot_grip = torch.cat([left_action_rot_grip, left_action_rot_grip], dim=0)
    pcd = [torch.cat([pcd[0], pcd[0]], dim=0)]

    # bounds = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    bounds = torch.tensor([[0.0, -1.0, -0.2, 1.0, 0.0, 0.8]])
    layer = 0
    trans_aug_range = torch.tensor([0.125, 0.125, 0.125])
    rot_aug_range = [0.0, 0.0, 45.0]
    rot_aug_resolution = 5
    voxel_size = 100
    rot_resolution = 5
    device = 'cpu'

    before_np = convert_to_numpy(pcd[0][0])
    plot_point_cloud(before_np, left_action_gripper_pose, right_action_gripper_pose, "Before Augmentation", os.path.join(root_path, "before.png"))

    # Call the function
    # outputs = original_bimanual_apply_se3_augmentation(
    outputs = bimanual_apply_se3_augmentation(
        pcd,
        right_action_gripper_pose,
        right_action_trans,
        right_action_rot_grip,
        left_action_gripper_pose,
        left_action_trans,
        left_action_rot_grip,
        bounds,
        layer,
        trans_aug_range,
        rot_aug_range,
        rot_aug_resolution,
        voxel_size,
        rot_resolution,
        device,
    )
    # Unpack outputs
    right_action_trans_out, right_action_rot_grip_out, left_action_trans_out, left_action_rot_grip_out, pcd_out = outputs

    # use index 0 if bs > 1
    right_action_trans_out = right_action_trans_out[0].unsqueeze(0)
    right_action_rot_grip_out = right_action_rot_grip_out[0].unsqueeze(0)
    left_action_trans_out = left_action_trans_out[0].unsqueeze(0)
    left_action_rot_grip_out = left_action_rot_grip_out[0].unsqueeze(0)
    # pcd_out = [pcd_out[0]]

    # Visualize point clouds
    after_np = convert_to_numpy(pcd_out[0][0])

    # Visualize and save before and after augmentation

    # left_action_gripper_pose = np.concatenate([left_action_trans_out, left_action_rot_grip_out], axis=1)
    # right_action_gripper_pose = np.concatenate([right_action_trans_out, right_action_rot_grip_out], axis=1)
    res = (bounds[:, 3:] - bounds[:, :3]) / voxel_size
    right_action_trans_out = bounds[:, :3] + res * right_action_trans_out + res / 2
    # right_action_rot_grip_out = right_action_rot_grip_out.squeeze().cpu().numpy()
    right_action_rot_grip_out = utils.discrete_euler_to_quaternion(right_action_rot_grip_out[:, -4:-1], rot_resolution)
    right_action_gripper_pose = np.concatenate([right_action_trans_out, right_action_rot_grip_out], axis=1)

    left_action_trans_out = bounds[:, :3] + res * left_action_trans_out + res / 2
    # left_action_rot_grip_out = left_action_rot_grip_out.squeeze().cpu().numpy()
    left_action_rot_grip_out = utils.discrete_euler_to_quaternion(left_action_rot_grip_out[:, -4:-1], rot_resolution)
    left_action_gripper_pose = np.concatenate([left_action_trans_out, left_action_rot_grip_out], axis=1)

    plot_point_cloud(after_np, left_action_gripper_pose, right_action_gripper_pose, "After Augmentation", os.path.join(root_path, "after.png"))


