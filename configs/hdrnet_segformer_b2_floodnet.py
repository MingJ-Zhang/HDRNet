_base_ = './hdrnet_segformer_b2_common.py'

data_root = 'data/FloodNet'
dataset_type = 'FloodNetDataset'
num_classes = 10

# FloodNet class ids:
# 0 background, 1 building_flooded, 2 building_non_flooded,
# 3 road_flooded, 4 road_non_flooded, 5 water, 6 tree, 7 vehicle, 8 pool, 9 grass
model = dict(
    decode_head=dict(
        num_classes=num_classes,
        num_object_classes=7,
        num_damage_classes=2,
        damage_ordinal=False,
        # object ids: 0 bg, 1 building, 2 road, 3 water, 4 vegetation, 5 vehicle, 6 pool
        object_class_mapping=[0, 1, 1, 2, 2, 3, 4, 5, 6, 4],
        # coarse ids: 0 bg, 1 building, 2 road, 3 water, 4 other foreground
        coarse_class_mapping=[0, 1, 1, 2, 2, 3, 4, 4, 4, 4],
        # damage/status ids: 0 normal/other, 1 flooded/water-related
        damage_class_mapping=[0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        damage_semantic_ids=[1, 3, 5]))

train_dataloader = dict(dataset=dict(
    _delete_=True,
    type=dataset_type,
    data_root=data_root,
    img_suffix='.jpg',
    data_prefix=dict(
        img_path='train_only_crop1024/images',
        seg_map_path='train_only_crop1024/labels'),
    pipeline={{_base_.train_pipeline}}))

val_dataloader = dict(dataset=dict(
    _delete_=True,
    type=dataset_type,
    data_root=data_root,
    img_suffix='.jpg',
    seg_map_suffix='_lab.png',
    data_prefix=dict(
        img_path='test/test-org-img',
        seg_map_path='test/test-label-img'),
    pipeline={{_base_.test_pipeline}}))

test_dataloader = val_dataloader
work_dir = './work_dirs/hdrnet_segformer_b2_floodnet_gated'
