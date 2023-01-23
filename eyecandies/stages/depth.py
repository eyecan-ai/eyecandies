import typing as t
import numpy as np

from pipelime.stages import SampleStage
from pydantic import Field

if t.TYPE_CHECKING:
    from pipelime.sequences import Sample


class DepthToMetersStage(SampleStage, title="depth2mt"):
    """Converts 16bit depth images to meters as floating point numpy array."""

    depth_key: str = Field(
        "depth", description="The key of the input depth image item."
    )
    mind_key_address: str = Field(
        "info_depth.normalization.min",
        description=(
            "The pydash address of the minimum depth value in the sample's item."
        ),
    )
    maxd_key_address: str = Field(
        "info_depth.normalization.max",
        description=(
            "The pydash address of the maximum depth value in the sample's item."
        ),
    )

    use_float64: bool = Field(
        False, description="Whether to use 64-bit floats for the output metric depth."
    )

    out_depth_key_format: str = Field(
        "*",
        description=(
            "The name of the item containing the output depth image. "
            "Any `*` will be replaced with the input item key."
        ),
    )

    def __call__(self, x: "Sample") -> "Sample":
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
    """Converts a metric depth to pointcloud."""

    depth_key: str = Field(
        "depth", description="The key of the input metric depth item"
    )
    image_key: t.Optional[str] = Field(
        "image_0",
        description="The optional key of the input image to use to color the PC",
    )
    normals_key: t.Optional[str] = Field(
        "normals", description="The optional key of the input normals"
    )
    pose_key: t.Optional[str] = Field(
        "pose", description="The optional key of the pose matrix"
    )
    focal_length: float = Field(711.11, description="The camera focal length")
    use_float64: bool = Field(
        False, description="Whether to use 64-bit floats for points and normals."
    )
    out_pcd_key: str = Field(
        "pcd", description="The key of the output pointcloud item."
    )

    def __call__(self, x: "Sample") -> "Sample":
        import trimesh
        from sklearn.preprocessing import normalize
        from pipelime.items import PLYModel3DItem

        if self.depth_key not in x:
            return x

        pcd, valid_mask, pose = self._depth_to_pointcloud(x)

        if self.image_key is not None and self.image_key in x:
            # [
            #   [red_0, green_0, blue_0, alpha_0],
            #   [red_1, green_1, blue_1, alpha_1],
            #   ...
            # ]
            colors = x[self.image_key]()
            colors = colors.reshape(-1, 3)[valid_mask]  # type: ignore
            colors = np.hstack(
                [colors, 255 * np.ones((colors.shape[0], 1), dtype=colors.dtype)]
            )
        else:
            colors = None

        if self.normals_key is not None and self.normals_key in x:
            # [
            #   [nx_0, ny_0, nz_0],
            #   [nx_1, ny_1, nz_1],
            #   ...
            # ]
            normals = x[self.normals_key]()
            normals = normals.reshape(-1, 3)[valid_mask]  # type: ignore
            normals = normalize(normals.astype(pcd.dtype) / 127.5 - 1.0, norm="l2")

            # normals reference frame has  Z and Y flipped compared to the camera frame
            normals = normals @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

            # move to world reference frame
            normals = normals @ pose[:3, :3].T  # type: ignore
        else:
            normals = None

        # pipelime uses trimesh for 3D models
        pcd = trimesh.Trimesh(
            vertices=pcd, vertex_colors=colors, vertex_normals=normals
        )

        # set the pointcloud on the sample
        x = x.set_item(self.out_pcd_key, PLYModel3DItem(pcd))
        return x

    def _depth_to_pointcloud(self, x: "Sample"):
        depth_mt: np.ndarray = x[self.depth_key]()  # type: ignore

        if self.pose_key is not None and self.pose_key in x:
            pose: np.ndarray = x[self.pose_key]()  # type: ignore
        else:
            pose = np.eye(4)

        # camera intrinsics
        height, width = depth_mt.shape[:2]
        intrinsics_4x4 = np.array(
            [
                [self.focal_length, 0, width / 2, 0],
                [0, self.focal_length, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # [ depth_0, depth_1, ... ]
        flattened_depth = depth_mt.reshape(-1)
        valid_mask = flattened_depth > 0
        flattened_depth = flattened_depth[valid_mask]

        # [ 0, 1, ..., width-1 ] repeated `height` times
        xcoords = np.tile(np.arange(width, dtype=depth_mt.dtype), height)[valid_mask]

        # [ 0, ..., 0, 1, ..., 1, ..., height-1, ..., height-1 ]
        # each element repeated `width` times
        ycoords = np.repeat(np.arange(height, dtype=depth_mt.dtype), width)[valid_mask]

        # ┌      u0         ... ┐
        # │      v0         ... │
        # │       1         ... │
        # └ 1/depth[v0, u0] ... ┘
        coord_grid = np.stack(
            [
                xcoords,
                ycoords,
                np.ones(xcoords.shape[0], dtype=depth_mt.dtype),
                1.0 / flattened_depth,
            ],
            axis=0,
        )

        # project to 3D and move to world reference frame
        hom_3d_pts = pose @ np.linalg.inv(intrinsics_4x4) @ coord_grid

        # remove the homogeneous coordinate and return
        pcd = flattened_depth.reshape(-1, 1) * hom_3d_pts.T
        return (
            pcd[:, :3].astype(np.float64 if self.use_float64 else np.float32),
            valid_mask,
            pose,
        )
