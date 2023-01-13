import typing as t

from pipelime.stages import SampleStage
from pydantic import Field

import numpy as np
import trimesh

if t.TYPE_CHECKING:
    from pipelime.sequences import Sample


class DepthToMetersStage(SampleStage, title="depth2mt"):
    """Converts 16bit depth images to meters as floating point numpy array."""

    depth_key: str = Field(
        "depth", description="The key of the input depth image item."
    )
    mind_key_address: str = Field(
        "info_depth.normalization.min",
        description="The pydash address of the minimum depth value in the sample's item.",
    )
    maxd_key_address: str = Field(
        "info_depth.normalization.max",
        description="The pydash address of the maximum depth value in the sample's item.",
    )

    use_float64: bool = Field(
        False, description="Whether to use 64-bit floats for the output metric depth."
    )

    out_depth_key_format: str = Field(
        "*",
        description="The name of the item containing the output depth image. Any `*` will be replaced with the input item key.",
    )

    def __call__(self, x: "Sample") -> "Sample":
        import numpy as np
        from pipelime.items import NpyNumpyItem

        mind = x.deep_get(self.mind_key_address)
        maxd = x.deep_get(self.maxd_key_address)
        if (
            self.depth_key not in x
            or not isinstance(mind, float)
            or not isinstance(maxd, float)
        ):
            return x

        dimg: np.ndarray = x[self.depth_key]()  # type: ignore
        dimg = dimg.astype(np.float64 if self.use_float64 else np.float32)
        dimg = dimg / 65535.0 * (maxd - mind) + mind

        x = x.set_item(
            self.out_depth_key_format.replace("*", self.depth_key), NpyNumpyItem(dimg)
        )
        return x


class DepthToPCStage(SampleStage, title="depth2pc"):
    """Converts floating point numpy array representing depth to pointcloud."""

    depth_key: str = Field(
        "depth", description="The key of the input metric depth item."
    )
    image_key: str = Field(
        "image_0", description="The key of the input image to use to color the PC"
    )
    pose_key: str = Field("pose", description="The key of the pose matrix")
    focal_length: str = Field(50, description="The camera focal length [mm]")
    sensor_size: str = Field(36, description="The sensor size [mm]")
    out_pc_key: str = Field("pc", description="The key of the output pointcloud item.")

    def __call__(self, x: "Sample") -> "Sample":
        import open3d as o3d
        import pipelime.items as pli

        # if self.depth_key not in x:
        #     return x

        dimg: np.ndarray = x[self.depth_key]()
        color: np.ndarray = x[self.image_key]()
        extrinsics: np.ndarray = x[self.pose_key]()

        width, height, _ = color.shape
        fx = self.focal_length / self.sensor_size * width
        fy = self.focal_length / self.sensor_size * height
        cx = width / 2
        cy = height / 2

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(dimg),
            convert_rgb_to_intensity=False,
        )

        # load intrinsics
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(width, height, fx, fy, cx, cy)

        # Create pointcloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd_image, intrinsic=intrinsics, extrinsic=extrinsics
        )
        # Flip it, otherwise the pointcloud will be upside down.
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # convert from open3d to trimesh geometry for pipelime item compatibility
        pcd = self.o3d_to_trimesh_pointcloud(pcd)
        x = x.set_item(self.out_pc_key, pli.PLYModel3DItem(pcd))
        return x

    @staticmethod
    def o3d_to_trimesh_pointcloud(o3d_pcd):
        RGB = np.asarray(o3d_pcd.colors)
        RGBA = np.concatenate(
            [RGB, np.zeros((RGB.shape[0], 1)).astype("uint8")], axis=-1
        )
        pcd = trimesh.PointCloud(vertices=np.asarray(o3d_pcd.points), colors=RGBA)

        return pcd

    @staticmethod
    def trimesh_to_o3d_pointcloud(trimesh_pcd):
        import open3d as o3d

        o3d_points_vec = o3d.utility.Vector3dVector(np.asarray(trimesh_pcd.vertices))
        # RGBA 2 RGB
        o3d_colors_vec = o3d.utility.Vector3dVector(
            np.asarray(trimesh_pcd.colors)[:, 0:3].astype("float64") / 255
        )
        print(np.asarray(o3d_points_vec))
        pcd = o3d.geometry.PointCloud(points=o3d_points_vec)
        pcd.colors = o3d_colors_vec
        return pcd
