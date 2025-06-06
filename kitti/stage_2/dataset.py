from vision3d.datasets.registration import OdometryKittiPairDataset
from vision3d.utils.collate import GraphPyramid2D3DRegistrationCollateFn
from vision3d.utils.dataloader import build_dataloader, calibrate_neighbors_pack_mode


def train_valid_data_loader(cfg):
    train_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_dir,
        "train",
        num_pc=cfg.train.max_points,
    )
 
    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        GraphPyramid2D3DRegistrationCollateFn,
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
    )

    collate_fn = GraphPyramid2D3DRegistrationCollateFn(
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
        neighbor_limits,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_dir,
        "val",
        num_pc=cfg.test.max_points,
    )

    valid_loader = build_dataloader(
        valid_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg):
    train_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_dir,
        "train",
        num_pc=cfg.train.max_points,
    )

    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        GraphPyramid2D3DRegistrationCollateFn,
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
    )

    collate_fn = GraphPyramid2D3DRegistrationCollateFn(
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
        neighbor_limits,
    )

    test_dataset = OdometryKittiPairDataset(
        cfg.data.dataset_dir,
        "val",
        num_pc=cfg.test.max_points,
    )

    test_loader = build_dataloader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return test_loader, neighbor_limits