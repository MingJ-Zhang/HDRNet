_base_ = './hdrnet_segformer_b2_common.py'

data_root = 'data/FloodNet'
dataset_type = 'FloodNetDataset'
num_classes = 10

# Coarse disaster classes used by DRCA:
# 0 water/flood, 1 building/infrastructure, 2 road, 3 vegetation, 4 vehicle/other
class_cooccurrence_matrix = [
    [1.0, 0.8, 0.8, 0.4, 0.3],
    [0.8, 1.0, 0.7, 0.5, 0.4],
    [0.8, 0.7, 1.0, 0.4, 0.5],
    [0.4, 0.5, 0.4, 1.0, 0.3],
    [0.3, 0.4, 0.5, 0.3, 1.0],
]

# FloodNet class ids:
# 0 background, 1 building_flooded, 2 building_non_flooded,
# 3 road_flooded, 4 road_non_flooded, 5 water, 6 tree, 7 vehicle, 8 pool, 9 grass
model = dict(
    decode_head=dict(
        num_classes=num_classes,
        num_coarse_classes=5,
        num_object_classes=7,
        num_damage_classes=2,
        damage_ordinal=False,
        class_cooccurrence_matrix=class_cooccurrence_matrix,
        # coarse ids: water/flood, building/infra, road, vegetation, vehicle/other
        coarse_class_mapping=[4, 1, 1, 2, 2, 0, 3, 4, 0, 3],
        # object ids: 0 bg, 1 building, 2 road, 3 water, 4 vegetation, 5 vehicle, 6 pool
        object_class_mapping=[0, 1, 1, 2, 2, 3, 4, 5, 6, 4],
        # damage/status ids: 0 normal/other, 1 flooded/water-related
        damage_class_mapping=[0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        damage_semantic_ids=[1, 3, 5],
        loss_weights=dict(seg=1.0, coarse=0.1, obj=0.4, damage=0.6, hier=0.3, boundary=0.4)))

train_dataloader = dict(dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
    pipeline={{_base_.train_pipeline}}))

val_dataloader = dict(dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='images/test', seg_map_path='annotations/test'),
    pipeline={{_base_.test_pipeline}}))

test_dataloader = val_dataloader
work_dir = './work_dirs/hdrnet_segformer_b2_floodnet'
