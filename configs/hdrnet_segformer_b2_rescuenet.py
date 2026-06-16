_base_ = './hdrnet_segformer_b2_common.py'

data_root = 'data/RescueNet'
dataset_type = 'RescueNetDataset'
num_classes = 11

# Coarse disaster classes used by DRCA:
# 0 water/flood, 1 building/infrastructure, 2 road, 3 vegetation, 4 vehicle/other
class_cooccurrence_matrix = [
    [1.0, 0.8, 0.8, 0.4, 0.3],
    [0.8, 1.0, 0.7, 0.5, 0.4],
    [0.8, 0.7, 1.0, 0.4, 0.5],
    [0.4, 0.5, 0.4, 1.0, 0.3],
    [0.3, 0.4, 0.5, 0.3, 1.0],
]

# Check your official RescueNet mask index order before training.
# Here: 0 bg, 1 water, 2 building_no_damage, 3 building_minor_damage,
# 4 building_major_damage, 5 building_total_destruction, 6 vehicle,
# 7 road_clear, 8 road_blocked, 9 tree, 10 pool
model = dict(
    decode_head=dict(
        num_classes=num_classes,
        num_coarse_classes=5,
        num_object_classes=7,
        num_damage_classes=4,
        damage_ordinal=True,
        class_cooccurrence_matrix=class_cooccurrence_matrix,
        # coarse ids: water/flood, building/infra, road, vegetation, vehicle/other
        coarse_class_mapping=[4, 0, 1, 1, 1, 1, 4, 2, 2, 3, 0],
        # object ids: 0 bg, 1 water, 2 building, 3 vehicle, 4 road, 5 vegetation, 6 pool
        object_class_mapping=[0, 1, 2, 2, 2, 2, 3, 4, 4, 5, 6],
        # damage ids: 0 none/clear, 1 minor/blocked, 2 major, 3 total destruction
        damage_class_mapping=[0, 0, 0, 1, 2, 3, 0, 0, 1, 0, 0],
        damage_semantic_ids=[3, 4, 5, 8],
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
work_dir = './work_dirs/hdrnet_segformer_b2_rescuenet'
