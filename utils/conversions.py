# Copyright (c) 2023 Boston Dynamics AI Institute, LLC. All rights reserved.
import array
import struct
import sys
from typing import List, Optional

import cv2
import numpy
import numpy as np
import numpy.typing as npt
import open3d
import sensor_msgs_py.point_cloud2 as pc2
from bosdyn.api import geometry_pb2
from bosdyn.client.math_helpers import Quat, SE2Pose, SE3Pose, SE3Velocity
from builtin_interfaces.msg import Time as RosTimestamp
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Point as RosPoint,
)
from geometry_msgs.msg import (
    Pose as RosPose,
)
from geometry_msgs.msg import (
    Pose2D as RosPose2D,
)
from geometry_msgs.msg import (
    PoseStamped as RosPoseStamped,
)
from geometry_msgs.msg import (
    PoseWithCovariance as RosPoseWithCovariance,
)
from geometry_msgs.msg import (
    PoseWithCovarianceStamped as RosPoseWithCovarianceStamped,
)
from geometry_msgs.msg import (
    Quaternion as RosQuaternion,
)
from geometry_msgs.msg import (
    QuaternionStamped as RosQuaternionStamped,
)
from geometry_msgs.msg import (
    Transform as RosTransform,
)
from geometry_msgs.msg import (
    TransformStamped as RosTransformStamped,
)
from geometry_msgs.msg import (
    Twist as RosTwist,
)
from geometry_msgs.msg import (
    TwistStamped as RosTwistStamped,
)
from geometry_msgs.msg import (
    TwistWithCovariance as RosTwistWithCovariance,
)
from geometry_msgs.msg import (
    TwistWithCovarianceStamped as RosTwistWithCovarianceStamped,
)
from geometry_msgs.msg import (
    Vector3 as RosVector3,
)
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import PointCloud2 as RosPointCloud2
from sensor_msgs.msg import PointField as RosPointField
from spatialmath import (
    SE2,
    SE3,
    SpatialForce,
    Twist3,
    UnitQuaternion,
)
from spatialmath.base import R3

# prefix to the names of dummy fields we add to get byte alignment
# correct. this needs to not clash with any actual field names
_DUMMY_POINT_FIELD_PREFIX = "__"

# mappings between PointField types and numpy types
_pfnptype_mappings = [
    (RosPointField.INT8, numpy.dtype("int8")),
    (RosPointField.UINT8, numpy.dtype("uint8")),
    (RosPointField.INT16, numpy.dtype("int16")),
    (RosPointField.UINT16, numpy.dtype("uint16")),
    (RosPointField.INT32, numpy.dtype("int32")),
    (RosPointField.UINT32, numpy.dtype("uint32")),
    (RosPointField.FLOAT32, numpy.dtype("float32")),
    (RosPointField.FLOAT64, numpy.dtype("float64")),
]
_pftype_to_nptype = dict(_pfnptype_mappings)
_nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in _pfnptype_mappings)


def ros_pointcloud2_to_numpy_array(cloud_msg: RosPointCloud2, extract_color: bool = True) -> npt.NDArray:
    """Converts an ROS2 PointCloud2 message to numpy array.
    The point cloud will be can be either xyz (i.e. no color), or xyzrgb (i.e. with color).
    Color is represented as floating point RGB values between 0 and 1.

    Args:
        cloud_msg: ROS2 PointCloud2 message to convert into a numpy array
        extract_color: If True, color is extracted from ROS2 Message, if False color is ignored

    Returns:
        Numpy array containing point cloud. If extract_color==True, the point cloud is of size
            Nx6, with each row containing point information in the form of [x,y,z,r,g,b].
            If extract_color==False, the point cloud is of size Nx3, with each row containing
            point information in the form [x,y,z].
    """

    # This function assumes a particular structure to the message:
    if len(cloud_msg.fields) not in [3, 4, 6]:
        raise ValueError(f"Unexpected number of fields: {cloud_msg.fields}. Expecting 3, 4, or 6.")

    # First field is x
    if not cloud_msg.fields[0].name == "x":
        raise ValueError(f"Field 0 should be x, got {cloud_msg.fields[0].name}.")
    # Second field is y
    if not cloud_msg.fields[1].name == "y":
        raise ValueError(f"Field 1 should be y, got {cloud_msg.fields[1].name}.")
    # Third field is z
    if not cloud_msg.fields[2].name == "z":
        raise ValueError(f"Field 2 should be z, got {cloud_msg.fields[2].name}.")

    # All fields are float32
    for field in cloud_msg.fields:
        if not field.datatype == RosPointField.FLOAT32:
            raise ValueError(f"{field.name} field should be float32, got {field.datatype}.")
    # This uses the datatype information in the message to correctly interpret the data.  It will return an array
    # that is N x 4 or N x 6 where the first 3 columns are x, y, z as floating point (as we want) and the remaining
    # columns represent color. If the array is N x 4, fourth column is a floating point value that can be converted
    # into r, g, and b fields with some work. If the array is N x 6, the last three columns are floating point r, g,
    # and b.
    cloud_arr = pc2.read_points_numpy(cloud_msg, skip_nans=True)

    if extract_color and len(cloud_msg.fields) > 3:
        if len(cloud_msg.fields) == 4:
            # Last field is rgb all as one value
            if not cloud_msg.fields[3].name == "rgb":
                raise ValueError(f"Field 3 of 4d points should be rgb, got {cloud_msg.fields[3].name}.")

            # The fourth column is the floating point value for rgb
            rgb_arr = cloud_arr[:, 3]
            # For some reason PointCloud2 encodes the RGB value as a float.
            # But actually it's an integer where the first byte is r, the second is g,
            # and the third is b.  Cast to an integer
            rgb_arr.dtype = numpy.uint32
            # First byte of this integer is the r value
            r = numpy.asarray((rgb_arr >> 16) & 255, dtype=numpy.uint8)
            # Second byte of this integer is the b value
            g = numpy.asarray((rgb_arr >> 8) & 255, dtype=numpy.uint8)
            # Third byte of this integer is the g value
            b = numpy.asarray(rgb_arr & 255, dtype=numpy.uint8)
            # We want to return an N x 6 array because that's what open3d likes.
            # The first 3 columns are already x, y, z.
            # We want the last three to be r, g, b.  Since the array was already N x 4
            # overwrite the fourth column with the r values.
            cloud_arr[:, 3] = r / 255.0
            # Add the g values as the fifth column
            cloud_arr = numpy.append(cloud_arr, g.reshape(-1, 1) / 255.0, axis=1)
            # Add the b values as the sixth column
            cloud_arr = numpy.append(cloud_arr, b.reshape(-1, 1) / 255.0, axis=1)
        else:
            # Last fields are r, g, b
            if not (
                cloud_msg.fields[3].name == "r" and cloud_msg.fields[4].name == "g" and cloud_msg.fields[5].name == "b"
            ):
                raise ValueError(
                    "Fields 3-5 of 6d points should be r, g, b, respectively. "
                    + f"Got {cloud_msg.fields[3].name}, {cloud_msg.fields[4].name}, {cloud_msg.fields[5].name}"
                )
    else:
        # Drop 4th column if color is not requested
        cloud_arr = cloud_arr[:, :3]

    return cloud_arr

def merge_rgb_fields(cloud_arr: npt.NDArray) -> npt.NDArray:
    """Takes an array with named np.uint8 fields 'r', 'g', and 'b', and returns an array in
    which they have been merged into a single np.float32 'rgb' field. The first byte of this
    field is the 'r' uint8, the second is the 'g', uint8, and the third is the 'b' uint8.

    This is the way that pcl likes to handle RGB colors for some reason.
    """
    r = np.asarray(cloud_arr["r"] * 255, dtype=np.uint32)
    g = np.asarray(cloud_arr["g"] * 255, dtype=np.uint32)
    b = np.asarray(cloud_arr["b"] * 255, dtype=np.uint32)
    rgb_arr = np.array((r << 16) | (g << 8) | (b << 0), dtype=np.uint32)

    # not sure if there is a better way to do this. i'm changing the type of the array
    # from uint32 to float32, but i don't want any conversion to take place -jdb
    rgb_arr.dtype = np.float32

    # create a new array, without r, g, and b, but with rgb float32 field
    new_dtype = []
    for field_name in cloud_arr.dtype.names:
        field_type, field_offset = cloud_arr.dtype.fields[field_name]
        if field_name not in ("r", "g", "b"):
            new_dtype.append((field_name, field_type))
    new_dtype.append(("rgb", np.float32))
    new_cloud_arr = np.zeros(cloud_arr.shape, new_dtype)

    # fill in the new array
    for field_name in new_cloud_arr.dtype.names:
        if field_name == "rgb":
            new_cloud_arr[field_name] = rgb_arr
        else:
            new_cloud_arr[field_name] = cloud_arr[field_name]

    return new_cloud_arr


def numpy_array_to_ros_pointcloud2(
    cloud_arr: npt.NDArray,
    stamp: Optional[RosTimestamp] = None,
    frame_id: Optional[str] = None,
) -> RosPointCloud2:
    """Converts an numpy array into a ROS2 PointCloud2 message.
    The point cloud can be either xyz (i.e. no color), or xyzrgb (i.e. with color),
    depending on whether the size is Nx3 or Nx6, respectively.
    Color is represented as floating point RGB values between 0 and 1.

    Args:
        cloud_arr: Numpy array of size Nx3 or Nx6, with rows containing point information
            in the form of [x,y,z] or [x,y,z,r,g,b].
        stamp: ROS Timestamp for PointCloud2 message
        frame_id: Frame ID for PointCloud2 message

    Returns:
        ROS2 PointCloud2 message containing all point data, with header timestamp and frame_id
            if given

    """

    # convert the numpy array to a record array (specify field type for each column)
    if cloud_arr.shape[1] == 3:
        np_dtypes = [(name, numpy.float32) for name in ["x", "y", "z"]]
    elif cloud_arr.shape[1] == 6:
        np_dtypes = [(name, numpy.float32) for name in ["x", "y", "z", "r", "g", "b"]]
    else:
        raise ValueError(f"Unexpected dimensionality {cloud_arr.shape} of cloud array. Expecting (N, 3) or (N, 6)")

    cloud_record_arr = numpy.zeros(len(cloud_arr), dtype=np_dtypes)
    cloud_record_arr["x"] = cloud_arr[:, 0]
    cloud_record_arr["y"] = cloud_arr[:, 1]
    cloud_record_arr["z"] = cloud_arr[:, 2]
    if cloud_arr.shape[1] == 6:
        cloud_record_arr["r"] = cloud_arr[:, 3]
        cloud_record_arr["g"] = cloud_arr[:, 4]
        cloud_record_arr["b"] = cloud_arr[:, 5]
    cloud_record_arr = merge_rgb_fields(cloud_record_arr)

    cloud_msg = _numpy_recarray_to_ros_pointcloud2(cloud_record_arr, stamp=stamp, frame_id=frame_id)
    return cloud_msg


def _point_fields_to_numpy_dtype(fields: List, point_step: int) -> List:
    """Convert a list of PointFields to a numpy record datatype."""
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(("%s%d" % (_DUMMY_POINT_FIELD_PREFIX, offset), numpy.uint8))
            offset += 1

        dtype = _pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = numpy.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += _pftype_to_nptype[f.datatype].itemsize * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(("%s%d" % (_DUMMY_POINT_FIELD_PREFIX, offset), numpy.uint8))
        offset += 1

    return np_dtype_list


def _numpy_dtype_to_point_fields(dtype: numpy.dtype) -> List:
    """Convert a numpy record datatype into a list of PointFields."""
    fields = []
    for field_name in dtype.names:
        np_field_type, field_offset = dtype.fields[field_name]
        pf = RosPointField()
        pf.name = field_name
        if np_field_type.subdtype:
            item_dtype, shape = np_field_type.subdtype
            pf.count = int(numpy.prod(shape))
            np_field_type = item_dtype
        else:
            pf.count = 1

        pf.datatype = _nptype_to_pftype[np_field_type]
        pf.offset = field_offset
        fields.append(pf)
    return fields


def _ros2_pointcloud2_to_numpy_recarray(cloud_msg: RosPointCloud2, squeeze: bool = True) -> numpy.ndarray:
    """Converts a rospy PointCloud2 message to a numpy recordarray

    Reshapes the returned array to have shape (height, width), even if the
    height is 1.

    The reason for using np.frombuffer rather than struct.unpack is
    speed... especially for large point clouds, this will be <much> faster.
    """
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = _point_fields_to_numpy_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = numpy.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [
            fname
            for fname, _type in dtype_list
            if not (fname[: len(_DUMMY_POINT_FIELD_PREFIX)] == _DUMMY_POINT_FIELD_PREFIX)
        ]
    ]

    if squeeze and cloud_msg.height == 1:
        return numpy.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return numpy.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


def _numpy_recarray_to_ros_pointcloud2(
    cloud_arr: numpy.ndarray, stamp: Optional[RosTimestamp] = None, frame_id: Optional[str] = None
) -> RosPointCloud2:
    """Converts a numpy record array to a sensor_msgs.msg.PointCloud2."""
    # make it 2d (even if height will be 1)
    cloud_arr = numpy.atleast_2d(cloud_arr)

    cloud_msg = RosPointCloud2()

    if stamp is not None:
        cloud_msg.header.stamp = stamp
    if frame_id is not None:
        cloud_msg.header.frame_id = frame_id
    cloud_msg.height = cloud_arr.shape[0]
    cloud_msg.width = cloud_arr.shape[1]
    cloud_msg.fields = _numpy_dtype_to_point_fields(cloud_arr.dtype)
    cloud_msg.is_bigendian = sys.byteorder != "little"
    cloud_msg.point_step = cloud_arr.dtype.itemsize
    cloud_msg.row_step = cloud_msg.point_step * cloud_arr.shape[1]
    cloud_msg.is_dense = all([numpy.isfinite(cloud_arr[fname]).all() for fname in cloud_arr.dtype.names])

    # The PointCloud2.data setter will create an array.array object for you if you don't
    # provide it one directly. This causes very slow performance because it iterates
    # over each byte in python.
    # Here we create an array.array object using a memoryview, limiting copying and
    # increasing performance.
    memory_view = memoryview(cloud_arr)
    if memory_view.nbytes > 0:
        array_bytes = memory_view.cast("B")
    else:
        # Casting raises a TypeError if the array has no elements
        array_bytes = b""  # type: ignore
    as_array = array.array("B")  # type: ignore
    as_array.frombytes(array_bytes)
    cloud_msg.data = as_array
    return cloud_msg


def to_ros_quaternion(quaternion: geometry_pb2.Quaternion | Quat | UnitQuaternion) -> RosQuaternion:
    """Converts an object from a number of quaternion classes to the ROS msg for a quaternion"""
    if isinstance(quaternion, geometry_pb2.Quaternion):
        return RosQuaternion(w=float(quaternion.w), x=float(quaternion.x), y=float(quaternion.y), z=float(quaternion.z))
    elif isinstance(quaternion, Quat):
        return RosQuaternion(w=float(quaternion.w), x=float(quaternion.x), y=float(quaternion.y), z=float(quaternion.z))
    elif isinstance(quaternion, UnitQuaternion):
        return RosQuaternion(w=quaternion.A[0], x=quaternion.A[1], y=quaternion.A[2], z=quaternion.A[3])
    else:
        raise TypeError(f"Cannot convert from '{type(quaternion)}' to ROS's Quaternion msg")


def _bd_se3velocity_to_twist3(velocity: SE3Velocity) -> Twist3:
    """Internal function to convert an object from BD's SE3Velocity class to the Twist3 class from Spatial Math"""
    return Twist3(
        [velocity.linear_velocity_x, velocity.linear_velocity_y, velocity.linear_velocity_z],
        [velocity.angular_velocity_x, velocity.angular_velocity_y, velocity.angular_velocity_z],
    )


def _bd_se2pose_to_se2(pose: SE2Pose) -> SE2:
    """Internal function to convert an object from BD's SE2Pose to the SE2 class from Spatial Math"""
    return SE2(x=pose.x, y=pose.y, theta=pose.angle)


def to_se2(pose: RosPose | RosTransform | SE3Pose | RosPose2D | SE2Pose | geometry_pb2.SE2Pose | SE3) -> SE2:
    """Converts an object from a number of classes to the SE2 class from Spatial Math"""
    if isinstance(pose, RosPose) or isinstance(pose, RosTransform) or isinstance(pose, SE3Pose):
        return to_se3(pose).yaw_SE2()
    elif isinstance(pose, RosPose2D):
        return SE2(x=pose.x, y=pose.y, theta=pose.theta)
    elif isinstance(pose, SE2Pose):
        return _bd_se2pose_to_se2(pose)
    elif isinstance(pose, geometry_pb2.SE2Pose):
        return _bd_se2pose_to_se2(SE2Pose.from_proto(pose))
    elif isinstance(pose, SE3):
        return pose.yaw_SE2()
    else:
        raise TypeError(f"Cannot convert from '{type(pose)}' to SE2Pose")


def _bd_se3pose_to_ros_pose(pose: SE3Pose) -> RosPose:
    return RosPose(
        position=RosPoint(x=float(pose.x), y=float(pose.y), z=float(pose.z)),
        orientation=to_ros_quaternion(pose.rotation),
    )


def to_ros_pose(
    pose: RosTransform | SE3Pose | SE2Pose | geometry_pb2.SE3Pose | geometry_pb2.SE2Pose | SE3 | SE2,
) -> RosPose:
    """Converts an object from a number of classes to ROS' Pose msg class"""
    if isinstance(pose, RosTransform):
        return RosPose(
            position=RosPoint(x=pose.translation.x, y=pose.translation.y, z=pose.translation.z),
            orientation=pose.rotation,
        )
    elif isinstance(pose, SE3Pose):
        return _bd_se3pose_to_ros_pose(pose)
    elif isinstance(pose, SE2Pose):
        return _bd_se3pose_to_ros_pose(SE3Pose.from_se2(pose))
    elif isinstance(pose, geometry_pb2.SE3Pose):
        return _bd_se3pose_to_ros_pose(SE3Pose.from_proto(pose))
    elif isinstance(pose, geometry_pb2.SE2Pose):
        return _bd_se3pose_to_ros_pose(SE3Pose.from_se2(SE2Pose.from_proto(pose)))
    elif isinstance(pose, SE3):
        return RosPose(
            position=RosPoint(x=pose.x, y=pose.y, z=pose.z), orientation=to_ros_quaternion(UnitQuaternion(pose))
        )
    elif isinstance(pose, SE2):
        return to_ros_pose(pose.SE3())
    else:
        raise TypeError(f"Cannot convert from '{type(pose)}' to ROS's Pose msg")


def to_ros_pose_stamped(
    pose: RosTransform | SE3Pose | SE2Pose | geometry_pb2.SE3Pose | geometry_pb2.SE2Pose | SE3 | SE2,
    stamp: Optional[RosTimestamp] = None,
    frame_id: Optional[str] = None,
) -> RosPoseStamped:
    """Converts an object from a number of classes to ROS's PoseStamped msg class"""
    ros_pose = to_ros_pose(pose)
    ros_pose_stamped = RosPoseStamped()
    if stamp is not None:
        ros_pose_stamped.header.stamp = stamp
    if frame_id is not None:
        ros_pose_stamped.header.frame_id = frame_id
    ros_pose_stamped.pose = ros_pose

    return ros_pose_stamped


def to_ros_transform(
    transform: RosPose | SE3Pose | SE2Pose | geometry_pb2.SE3Pose | geometry_pb2.SE2Pose | SE3 | SE2,
) -> RosTransform:
    """Converts an object from a number of classes to ROS's Transform msg class"""
    if isinstance(transform, RosPose):
        return RosTransform(
            translation=RosVector3(x=transform.position.x, y=transform.position.y, z=transform.position.z),
            rotation=transform.orientation,
        )
    elif isinstance(transform, SE3Pose):
        return RosTransform(
            translation=RosVector3(x=float(transform.x), y=float(transform.y), z=float(transform.z)),
            rotation=to_ros_quaternion(transform.rotation),
        )
    elif isinstance(transform, SE2Pose):
        return to_ros_transform(SE3Pose.from_se2(transform))
    elif isinstance(transform, geometry_pb2.SE3Pose):
        return to_ros_transform(SE3Pose.from_proto(transform))
    elif isinstance(transform, geometry_pb2.SE2Pose):
        return to_ros_transform(SE2Pose.from_proto(transform))
    elif isinstance(transform, SE3):
        return RosTransform(
            translation=RosVector3(x=transform.x, y=transform.y, z=transform.z),
            rotation=to_ros_quaternion(UnitQuaternion(transform)),
        )
    elif isinstance(transform, SE2):
        return RosTransform(
            translation=RosVector3(x=transform.x, y=transform.y, z=0.0),
            rotation=to_ros_quaternion(UnitQuaternion.Rz(transform.theta())),
        )
    else:
        raise TypeError(f"Cannot convert from '{type(transform)}' to ROS's Transform msg")


def to_ros_vector3(v: R3) -> RosVector3:
    return RosVector3(x=v[0], y=v[1], z=v[2])


def to_ros_point(p: R3 | RosVector3 | RosPoint) -> RosPoint:
    if isinstance(p, RosPoint):
        return p
    elif isinstance(p, RosVector3):
        return RosPoint(x=p.x, y=p.y, z=p.z)
    else:
        return RosPoint(x=p[0], y=p[1], z=p[2])


def se2_to_se2pose(se2: SE2) -> SE2Pose:
    return SE2Pose(x=se2.x, y=se2.y, angle=se2.theta())


def se2_to_se2pose_proto(se2: SE2) -> geometry_pb2.SE2Pose:
    return se2_to_se2pose(se2).to_proto()


def se3_to_se3pose(transform: SE3) -> SE3Pose:
    """Converts the SE3 from spatial math to SE3Pose from the spot-sdk"""
    q = UnitQuaternion(transform)
    return SE3Pose(x=transform.x, y=transform.y, z=transform.z, rot=Quat(w=q.s, x=q.v[0], y=q.v[1], z=q.v[2]))


def se3_to_se3pose_proto(transform: SE3) -> geometry_pb2.SE3Pose:
    """Converts the SE3 from spatial math to SE3Pose protobuf from the spot-sdk"""
    return se3_to_se3pose(transform).to_proto()


def twist3_to_se3velocity_proto(velocity: Twist3) -> geometry_pb2.SE3Velocity:
    """Converts a Twist3 from spatial math into a SE3Velocity protobuf from the spot-sdk"""
    return SE3Velocity(
        lin_x=velocity.v[0],
        lin_y=velocity.v[1],
        lin_z=velocity.v[2],
        ang_x=velocity.w[0],
        ang_y=velocity.w[1],
        ang_z=velocity.w[2],
    ).to_proto()


def spatialforce_to_wrench_proto(spatialforce: SpatialForce) -> geometry_pb2.Wrench:
    """Converts a SpatialForce from spatialmath into a Wrench protobuf from the spot-sdk"""
    wrench_vec = spatialforce.data[0]
    force = geometry_pb2.Vec3(x=wrench_vec[0], y=wrench_vec[1], z=wrench_vec[2])
    torque = geometry_pb2.Vec3(x=wrench_vec[3], y=wrench_vec[4], z=wrench_vec[5])
    return geometry_pb2.Wrench(force=force, torque=torque)


def ros_camera_intrinsics_to_pinhole_open3d(msg: CameraInfo) -> open3d.camera.PinholeCameraIntrinsic:
    """Converts a ROS CameraInfo message to an open3d pinhole camera."""
    intrinsics_matrix = msg.k.reshape([3, 3])
    return open3d.camera.PinholeCameraIntrinsic(
        width=msg.width,
        height=msg.height,
        cx=intrinsics_matrix[0, 2],
        cy=intrinsics_matrix[1, 2],
        fx=intrinsics_matrix[0, 0],
        fy=intrinsics_matrix[1, 1],
    )


# There is no `to_ros_pose2d` because it is deprecated as of Foxy and might be removed in the future


def to_opencv_rvec_tvec(transform: SE3) -> tuple[npt.NDArray, npt.NDArray]:
    """Converts a spatialmath SE3 pose to the rodrigues rotational representation and translation vector required by
        OpenCV.

    Returns:
        A tuple of (rvec, tvec), using the OpenCV API naming conventions.
    """
    rvec, _ = cv2.Rodrigues(transform.R)
    tvec = transform.t
    return rvec, tvec
