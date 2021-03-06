# Import PCL module
import pcl
# from pcl_helper import *

# Load Point Cloud file
cloud = pcl.load_XYZRGB('tabletop.pcd')


# Voxel Grid filter
vox = cloud.make_voxel_grid_filter()

LEAF_SIZE = 0.01

vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

cloud_filtered = vox.filter()
filename = 'voxel_downsampled.pcd'
pcl.save(cloud_filtered, filename)

# PassThrough filter
passthrough = cloud_filtered.make_passthrough_filter()

filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.5
axis_max = 1.1
passthrough.set_filter_limits(axis_min, axis_max)

cloud_filtered = passthrough.filter()
filename = 'pass_through_filtered.pcd'
pcl.save(cloud_filtered, filename)

# RANSAC plane segmentation
# Create the segmentation object
seg = cloud_filtered.make_segmenter()

# Set the model you wish to fit 
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

# Max distance for a point to be considered fitting the model
# Experiment with different values for max_distance 
# for segmenting the table
max_distance = .04
seg.set_distance_threshold(max_distance)

# Call the segment function to obtain set of inlier indices and model coefficients
inliers, coefficients = seg.segment()

# Extract inliers
extracted_inliers = cloud_filtered.extract(inliers, negative=True)
filename = 'extracted_inliers.pcd'
pcl.save(extracted_inliers, filename)
# Save pcd for table
# pcl.save(cloud, filename)

# Extract outliers


# Save pcd for tabletop objects


# white_cloud = XYZRGB_to_XYZ(cloud)
# tree = white_cloud.make_kdtree()

# ec = white_cloud.make_EuclideanClusterExtraction()

# ec.set_ClusterTolerance(0.001)
# ec.set_MinClusterSize(10)
# ec.set_MaxClusterSize(250)

# ec.set_SearchMethod(tree)
# cluster_indices = ec.Extract()

# cluster_color = get_color_list(len(cluster_indices))

# color_cluster_point_list = []

# for j, indices in enumerate(cluster_indices):
# 	for i, indice in enumerate(indices):
# 		color_cluster_point_list.append([white_cloud[indice][0], 
# 										white_cloud[indice][1], 										
# 										white_cloud[indice][2], 
# 										rgb_to_float(cluster_color[j])])

# cluster_cloud = pcl.PointCloud_PointXYZRGB()
# cluster_cloud.from_list(color_cluster_point_list)

# filename = 'clustered_cloud.pcd'
# pcl.save(cluster_cloud, filename)