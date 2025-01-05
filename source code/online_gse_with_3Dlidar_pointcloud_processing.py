#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point,Quaternion, Vector3
from std_msgs.msg import ColorRGBA
import numpy as np
import math
import heapq
import time
import random
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
from nav_msgs.msg import Odometry
import tf
from geometry_msgs.msg import Point, Quaternion,Twist
from math import pow, atan2, sqrt, pi, degrees
from sklearn.cluster import DBSCAN  # Faster clustering with DBSCAN
from sensor_msgs import point_cloud2 as pc2 
from std_msgs.msg import String  # Import the message type
from threading import Event
import json

# Global variables to store the robot pose and point cloud message
robot_pose = np.array([0.0, 0.0, 0.0])
robot_pose_received = False
pointcloud_msg = None
listener = None

def get_robot_pose(data):
    global robot_pose, robot_pose_received
    robot_pose[0] = data.pose.pose.position.x
    robot_pose[1] = data.pose.pose.position.y
    robot_pose[2] = data.pose.pose.position.z
    robot_pose_received = True
    return robot_pose[0],robot_pose[1],robot_pose[2]

def pointcloud_callback(msg):
    global pointcloud_msg
    pointcloud_msg = msg

def process_pointcloud(msg):
    global listener, robot_pose, marker_pub

    try:
        listener.waitForTransform('/world', msg.header.frame_id, rospy.Time(), rospy.Duration(0.5))#1.0
        (trans, rot) = listener.lookupTransform('/world', msg.header.frame_id, rospy.Time(0))
        transform_matrix = listener.fromTranslationRotation(trans, rot)
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logwarn("TF lookup failed")
        return {}

    # Extract points from the point cloud
    points = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))))

    if len(points) == 0:
        
        return 0

    # Transform points to the world frame
    points = np.dot(points, transform_matrix[:3, :3].T) + transform_matrix[:3, 3]

    # Create and downsample the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=0.2)  # Adjust voxel size for performance

    # Convert to numpy array for clustering
    cluster_points = np.asarray(pcd.points)

    # Split points into two vertical segments
    z_min, z_max = cluster_points[:, 2].min(), cluster_points[:, 2].max()
    z_split = (z_min + z_max) / 2  # Midpoint for vertical splitting

    marker_array = MarkerArray()
    marker_id = 0
    obstacle_info = []

    # Function to detect overlap between two bounding boxes
    def bbox_overlap(bbox1, bbox2):
        min1, max1 = bbox1.get_min_bound(), bbox1.get_max_bound()
        min2, max2 = bbox2.get_min_bound(), bbox2.get_max_bound()
        return np.all(max1 >= min2) and np.all(max2 >= min1)

    # Function to merge two bounding boxes
    def merge_two_bboxes(bbox1, bbox2):
        min_bound = np.minimum(bbox1.get_min_bound(), bbox2.get_min_bound())
        max_bound = np.maximum(bbox1.get_max_bound(), bbox2.get_max_bound())
        merged_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        return merged_bbox

    # Function to merge all overlapping bounding boxes
    def merge_bounding_boxes(bboxes):
        merged_boxes = []
        for bbox in bboxes:
            merged = False
            for i, existing_box in enumerate(merged_boxes):
                if bbox_overlap(existing_box, bbox):
                    # Merge the boxes if they overlap
                    merged_boxes[i] = merge_two_bboxes(existing_box, bbox)
                    merged = True
                    break
            if not merged:
                merged_boxes.append(bbox)
        return merged_boxes

    # Process both vertical segments
    for i, segment_points in enumerate([
        cluster_points[cluster_points[:, 2] < z_split],  # Lower segment
        cluster_points[cluster_points[:, 2] >= z_split]  # Upper segment
    ]):
        if len(segment_points) == 0:
            continue

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=1.0, min_samples=10).fit(segment_points)
        labels = clustering.labels_

        unique_labels = set(labels)
        unique_labels.discard(-1)  # Ignore noise

        #rospy.loginfo(f"Segment {i + 1}: Detected {len(unique_labels)} obstacles")

        bboxes = []
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster = segment_points[cluster_indices]

            # Create an Open3D point cloud for the cluster
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster)

            # Compute the bounding box for the cluster
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bboxes.append(bbox)

        # Merge overlapping bounding boxes
        merged_bboxes = merge_bounding_boxes(bboxes)

        for bbox in merged_bboxes:
            center = bbox.get_center()
            extents = bbox.get_extent()
    
            # Get the minimum and maximum bounds of the bounding box
            min_bound = bbox.get_min_bound()
            max_bound = bbox.get_max_bound()
            
            # Compute the 8 vertices of the bounding box
            vertices = [
                (min_bound[0], min_bound[1], min_bound[2]),
                (max_bound[0], min_bound[1], min_bound[2]),
                (min_bound[0], max_bound[1], min_bound[2]),
                (max_bound[0], max_bound[1], min_bound[2]),
                (min_bound[0], min_bound[1], max_bound[2]),
                (max_bound[0], min_bound[1], max_bound[2]),
                (min_bound[0], max_bound[1], max_bound[2]),
                (max_bound[0], max_bound[1], max_bound[2]),
            ]
            
             
            # Compute the closest point to the robot
            distances = np.linalg.norm(np.asarray(bbox.get_box_points()) - robot_pose, axis=1)
            closest_point = np.asarray(bbox.get_box_points())[np.argmin(distances)]

            #rospy.loginfo(
            #    f"Obstacle {marker_id}: Center (x, y, z) = {center}, "
            #    f"Horizontal size = {extents[0]:.3f}, Vertical size = {extents[2]:.3f}"
            #)

            # Publish bounding box and center markers
            publish_box(marker_array, bbox, marker_id)
            publish_point(marker_array, center, marker_id)
            outside_point = get_center_outside(bbox, center)
            publish_outside_point(marker_array, outside_point, marker_id)

            
            # Store obstacle info
            obstacle_info.append({
                "distance": np.min(distances),
                "outside_point": closest_point.tolist(),
                "center": center.tolist(),
                "horizontal_size": extents[0],
                "vertical_size": extents[2],
                #"vertices": vertices  # Include vertices in the dictionary
                #"vertices": [min_bound.tolist(), max_bound.tolist()]
                "box_min": min_bound.tolist(),
                "box_max": max_bound.tolist()
            })
            
            marker_id += 1

    # Publish all markers at once
    marker_pub.publish(marker_array)

    # Sort obstacles by distance
    obstacle_info.sort(key=lambda x: x["distance"])

    # Create the final dictionary of obstacles
    sorted_obstacle_info = {f"obstacle{idx}": info for idx, info in enumerate(obstacle_info)}

    return sorted_obstacle_info

def get_obstacles_info():
    global marker_pub, pointcloud_msg, robot_pose_received

    if pointcloud_msg and robot_pose_received:
        return process_pointcloud(pointcloud_msg)
    else:
        rospy.logwarn("Pointcloud or robot pose not received yet")
        return 0
 
 
def get_center_outside_random( random_point ,center, offset=0.5):
    
    # Calculate the direction vector from the robot to the center of the obstacle
    direction_vector = center - random_point
    direction_vector /= np.linalg.norm(direction_vector)

    # Adjust direction vector to ensure the outside point is always in front of the drone
    direction_vector[2] = 0  # Ensure the outside point remains at the same height as the obstacle center
    if np.linalg.norm(direction_vector) > 0:
        direction_vector /= np.linalg.norm(direction_vector)

    # Calculate the outside point
    outside_point_rd = center - direction_vector * offset

    #rospy.loginfo("Outside center point (x, y, z) = [%.3f, %.3f, %.3f]" % (outside_point_rd[0], outside_point_rd[1], outside_point_rd[2]))
    return outside_point_rd

def get_center_outside(bbox, center, offset=0.5):
    global robot_pose

    # Get the extents of the bounding box
    extents = bbox.get_extent()

    # Calculate the direction vector from the drone to the center of the obstacle
    direction_vector = center - robot_pose
    direction_vector[2] = 0  # Ensure the outside point remains at the same height as the obstacle center

    if np.linalg.norm(direction_vector) > 0:
        direction_vector /= np.linalg.norm(direction_vector)

    # Project the extents onto the direction vector
    extent_projection = np.abs(extents[0] * direction_vector[0]) + np.abs(extents[1] * direction_vector[1])
    
    # Adjust the offset to ensure the outside point is just outside the bounding box
    adjusted_offset = offset + extent_projection / 2.0

    # Calculate the outside point
    outside_point = center - direction_vector * adjusted_offset

    #rospy.loginfo("Outside center point (x, y, z) = [%.3f, %.3f, %.3f]" % (outside_point[0], outside_point[1], outside_point[2]))
    return outside_point


def publish_box(marker_array, bbox, marker_id):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "obstacles"
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    center = bbox.get_center()
    extents = bbox.get_extent()
    marker.pose.position.x = center[0]
    marker.pose.position.y = center[1]
    marker.pose.position.z = center[2]
    marker.pose.orientation.w = 1.0
    marker.scale.x = extents[0]
    marker.scale.y = extents[1]
    marker.scale.z = extents[2]
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.5  # Transparency
    marker_array.markers.append(marker)

def publish_point(marker_array, position, marker_id):
    point_marker = Marker()
    point_marker.header.frame_id = "world"
    point_marker.header.stamp = rospy.Time.now()
    point_marker.ns = "obstacle_centers"
    point_marker.id = marker_id + 1000  # Ensure unique IDs for points
    point_marker.type = Marker.SPHERE
    point_marker.action = Marker.ADD
    point_marker.pose.position.x = position[0]
    point_marker.pose.position.y = position[1]
    point_marker.pose.position.z = position[2]
    point_marker.pose.orientation.w = 1.0
    point_marker.scale.x = 0.1  # Size of the point marker
    point_marker.scale.y = 0.1
    point_marker.scale.z = 0.1
    point_marker.color.r = 0.0
    point_marker.color.g = 0.0
    point_marker.color.b = 1.0  # Blue color
    point_marker.color.a = 1.0
    marker_array.markers.append(point_marker)

    rospy.loginfo("Center point (x, y, z) = [%.3f, %.3f, %.3f]" % (position[0], position[1], position[2]))

def publish_outside_point(marker_array, position, marker_id):
    outside_marker = Marker()
    outside_marker.header.frame_id = "world"
    outside_marker.header.stamp = rospy.Time.now()
    outside_marker.ns = "obstacle_outside_points"
    outside_marker.id = marker_id + 2000  # Ensure unique IDs for outside points
    outside_marker.type = Marker.SPHERE
    outside_marker.action = Marker.ADD
    outside_marker.pose.position.x = position[0]
    outside_marker.pose.position.y = position[1]
    outside_marker.pose.position.z = position[2]
    outside_marker.pose.orientation.w = 1.0
    outside_marker.scale.x = 0.1  # Size of the outside point marker
    outside_marker.scale.y = 0.1
    outside_marker.scale.z = 0.1
    outside_marker.color.r = 1.0
    outside_marker.color.g = 0.0
    outside_marker.color.b = 0.0  # Red color
    outside_marker.color.a = 1.0
    marker_array.markers.append(outside_marker)

    rospy.loginfo("Outside point (x, y, z) = [%.3f, %.3f, %.3f]" % (position[0], position[1], position[2]))



class MarkerPublisher:
    def __init__(self):
        #rospy.init_node("two_points_and_vector_rviz", anonymous=True)  # Added anonymous=True to ensure unique node names
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
        self.marker_pub= rospy.Publisher("visualization_marker", Marker, queue_size=10)
        self.marker_pub1 = rospy.Publisher("visualization_marker2", Marker, queue_size=10)
        self.pub_workspace = rospy.Publisher('visualization_marker3', Marker, queue_size=10)
        
        
        self.rate = rospy.Rate(100)  # Added a rate to control the publishing frequency
        
   
        

        
        

    def publish_two_points(self, point1, point2):
        # Create a marker for the points
        points_marker = Marker()
        points_marker.header.frame_id = "world"  # Ensure this matches the fixed frame in RViz
        points_marker.header.stamp = rospy.Time.now()
        points_marker.ns = "two_points"
        points_marker.action = Marker.ADD
        points_marker.pose.orientation.w = 1.0
        points_marker.id = 0
        points_marker.type = Marker.POINTS
        points_marker.scale.x = 0.1  # Point size
        points_marker.scale.y = 0.1
        points_marker.color.g = 1.0  # Green color
        points_marker.color.a = 1.0  # Alpha

        # Add the two points to the marker
        points_marker.points.append(point1)
        points_marker.points.append(point2)

        # Publish the points marker
        self.marker_pub.publish(points_marker)
        self.rate.sleep()  # Sleep to maintain the publishing rate

    def publish_vector_between_points(self, point1, point2):
        # Create a marker for the vector arrow
        vector_marker = Marker()
        vector_marker.header.frame_id = "world"  # Ensure this matches the fixed frame in RViz
        vector_marker.header.stamp = rospy.Time.now()
        vector_marker.ns = "vector_between_points"
        vector_marker.action = Marker.ADD
        vector_marker.pose.orientation.w = 1.0
        vector_marker.id = 1  # Changed ID to be unique for each marker
        vector_marker.type = Marker.ARROW
        vector_marker.scale.x = 0.05  # Arrow shaft diameter
        vector_marker.scale.y = 0.1  # Arrow head diameter
        vector_marker.scale.z = 0.2  # Arrow head length
        vector_marker.color.r = 1.0  # Red color
        vector_marker.color.a = 1.0  # Alpha

        # Set the start and end points of the vector
        vector_marker.points.append(point1)
        vector_marker.points.append(point2)

        # Publish the vector marker
        self.marker_pub.publish(vector_marker)
        self.rate.sleep()  # Sleep to maintain the publishing rate
        
    def publish_single_point(self,point3):
        
    
        marker_pub = rospy.Publisher("visualization_marker2", Marker, queue_size=10)
        rate = rospy.Rate(100)  # 1 Hz

        # Create a marker for the single point
        point_marker = Marker()
        point_marker.header.frame_id = "world"  # Replace with a valid frame ID
        point_marker.header.stamp = rospy.Time.now()
        point_marker.ns = "single_point"
        point_marker.action = Marker.ADD
        point_marker.pose.orientation.w = 1.0
        point_marker.id = 0
        point_marker.type = Marker.POINTS
        point_marker.scale.x = 0.1  # Point size
        point_marker.scale.y = 0.1
        point_marker.color.g = 1.0  # Green color
        point_marker.color.a = 1.0  # Alpha

        # Add the point to the marker
        point_marker.points.append(point3)
 
        
        self.marker_pub.publish(point_marker)
        self.rate.sleep() 
        
        
    def publish_axes(self):
        
        # Create a marker for the axes
        axes_marker = Marker()
        axes_marker.header.frame_id = "world"
        axes_marker.header.stamp = rospy.Time.now()
        axes_marker.ns = "axes"
        axes_marker.action = Marker.ADD
        axes_marker.pose.orientation.w = 1.0
        axes_marker.id = 0
        axes_marker.type = Marker.LINE_LIST
        axes_marker.scale.x = 0.02  # Line width
        axes_marker.color.a = 1.0  # Alpha

        # Define colors for x, y, z axes
        red = ColorRGBA(1.0, 0.0, 0.0, 1.0)    # Red for x-axis
        green = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Green for y-axis
        blue = ColorRGBA(0.0, 0.0, 1.0, 1.0)   # Blue for z-axis

        # Add x, y, z axes
        origin = Point()
        end_x = Point(x=1.0)
        end_y = Point(y=1.0)
        end_z = Point(z=1.0)

        axes_marker.points.extend([origin, end_x, origin, end_y, origin, end_z])

        # Assign colors to axes
        axes_marker.colors.extend([red, red, green, green, blue, blue])

        # Publish the axes marker
        
        self.marker_pub.publish(axes_marker)
        self.rate.sleep()
        
        
    def quaternion_from_euler(self,roll, pitch, yaw):
        
        qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
        qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
        qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        return Quaternion(qx, qy, qz, qw)

    def euler_from_vector(self,vector):
        
        x, y, z = vector.x, vector.y, -vector.z
        yaw = math.atan2(y, x)
        pitch = math.atan2(z, math.sqrt(x**2 + y**2))
        roll = 0
        return roll, pitch, yaw
    
    def calculate_orientation_and_geometry(self,X, pix, lix):
        
        # Calculate direction vector
        direction_vector = Point(pix.x - X.x, pix.y - X.y, pix.z - X.z)
    
        # Calculate Euler angles and quaternion orientation
        roll, pitch, yaw = self.euler_from_vector(direction_vector)
        orientation = self.quaternion_from_euler(roll, pitch, yaw)
    
        # Calculate geometry
        rix =( math.sqrt((pix.x - X.x)**2 + (pix.y - X.y)**2 + (pix.z - X.z)**2))
        half_angle = math.atan2(lix, rix)  # Compute vertex angle /2

        if half_angle > math.pi / 2:
            
            half_angle = math.pi - half_angle
        elif half_angle < -math.pi / 2:
            
            half_angle = -math.pi - half_angle
        else:
            
            half_angle = half_angle

        # This modification ensures that half_angle always falls within the range [−π/2,π/2], 
        # thus preventing math.cos(half_angle) from becoming negative. This keeps the computed height hc non-negative

        hc = rix * math.cos(half_angle)
        rcb = hc * math.tan(half_angle)
        # slant_height = math.sqrt(hc**2 + rcb**2)
        slant_height = rix
        hsc = slant_height - math.sqrt(slant_height**2 - rcb**2)
 
        rsc = math.sqrt(slant_height**2 - (slant_height - hsc)**2)
        return orientation, slant_height, hsc, rsc, rix, half_angle, rcb, hc
    
    def create_cone_marker(self,frame_id, marker_id, X, orientation, hc, rcb):
        
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "cone"
        marker.id = marker_id
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        n_segments = 36
        delta_theta = 2 * math.pi / n_segments

        for i in range(n_segments):
            
            theta = i * delta_theta
            p1 = Point(0, 0, 0)
            p2 = Point(hc, rcb * math.cos(theta), rcb * math.sin(theta))
            p3 = Point(hc, rcb * math.cos(theta + delta_theta), rcb * math.sin(theta + delta_theta))
            marker.points.extend([p3, p2, p1])

        marker.pose.position = X
        marker.pose.orientation = orientation
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0, 1.0, 0, 0.5)
        return marker
    
    
    def create_spherical_cap_marker(self,frame_id, marker_id, X, orientation, slant_height, hsc, rsc, rix):
        
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "spherical_cap"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        n_segments = 90
        delta_theta = 2 * math.pi / n_segments
        delta_phi = math.pi / 360

        for i in range(n_segments):
            
            theta1 = i * delta_theta
            theta2 = (i + 1) % n_segments * delta_theta
            for j in range(int(math.pi / delta_phi)):
                
            
                phi1 = j * delta_phi
                phi2 = (j + 1) * delta_phi

                if phi1 > math.asin(rsc / slant_height):
                    
                    break

                p1 = Point(slant_height * math.cos(phi1), slant_height * math.sin(phi1) * math.cos(theta1), slant_height * math.sin(phi1) * math.sin(theta1))
                p2 = Point(slant_height * math.cos(phi1), slant_height * math.sin(phi1) * math.cos(theta2), slant_height * math.sin(phi1) * math.sin(theta2))
                p3 = Point(slant_height * math.cos(phi2), slant_height * math.sin(phi2) * math.cos(theta1), slant_height * math.sin(phi2) * math.sin(theta1))
                p4 = Point(slant_height * math.cos(phi2), slant_height * math.sin(phi2) * math.sin(theta2), slant_height * math.sin(phi2) * math.cos(theta2))

                marker.points.extend([p1, p2, p3, p1, p3, p4])

        marker.pose.position = X
        marker.pose.orientation = orientation
        marker.scale.x = 0.01
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 0, 0, 0.25)
        return marker

def update_markers(g, start, goal, shortest_path, marker_pub):
    markers = MarkerArray()

    # Create markers for points
    for idx, (vertex_name, (x, y, z, color)) in enumerate(g.vertices.items()):
        # Create a marker for each vertex
        point_marker = Marker()
        point_marker.header.frame_id = "world"
        point_marker.header.stamp = rospy.Time.now()
        point_marker.ns = "graph"
        point_marker.id = idx
        point_marker.type = Marker.SPHERE
        point_marker.action = Marker.ADD
        point_marker.pose.position = Point(x=x, y=y, z=z)
        point_marker.scale = Vector3(0.2, 0.2, 0.2)
        point_marker.color = color  # Use color from the vertex
        markers.markers.append(point_marker)

        # Create a marker for the text above the point
        text_marker = Marker()
        text_marker.header.frame_id = "world"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "graph"
        text_marker.id = idx + len(g.vertices)  # Offset by number of vertices
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position = Point(x=x, y=y, z=z + 0.5)  # Adjust z coordinate to position above the point
        text_marker.pose.orientation = Quaternion()  # Initialize to identity quaternion
        text_marker.scale = Vector3(0.1, 0.1, 0.1)
        text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)  # White color
        text_marker.text = vertex_name
        markers.markers.append(text_marker)

    # Create a marker array for the lines
    lines_markers = MarkerArray()

    # Add the edges to the lines markers
    for idx, (start_vertex, end_vertex, _) in enumerate(g.edges):
        lines_marker = Marker()
        lines_marker.header.frame_id = "world"  # Replace with a valid frame ID
        lines_marker.header.stamp = rospy.Time.now()
        lines_marker.ns = "graph"
        lines_marker.id = idx + 2 * len(g.vertices)  # Offset by number of vertices
        lines_marker.type = Marker.LINE_LIST
        lines_marker.action = Marker.ADD
        lines_marker.pose.orientation = Quaternion()  # Initialize to identity quaternion
        lines_marker.scale = Vector3(0.05, 0.05, 0.05)  # Line width
        lines_marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # Blue color for edges

        start_point = g.vertices[start_vertex][:3]
        end_point = g.vertices[end_vertex][:3]
        lines_marker.points.append(Point(x=start_point[0], y=start_point[1], z=start_point[2]))
        lines_marker.points.append(Point(x=end_point[0], y=end_point[1], z=end_point[2]))

        lines_markers.markers.append(lines_marker)

    # Highlight the shortest path
    path_marker = Marker()
    path_marker.header.frame_id = "world"
    path_marker.header.stamp = rospy.Time.now()
    path_marker.ns = "graph"
    path_marker.id = 2 * len(g.vertices) + len(g.edges)  # Offset by number of vertices and edges
    path_marker.type = Marker.LINE_STRIP
    path_marker.action = Marker.ADD
    path_marker.pose.orientation = Quaternion()  # Initialize to identity quaternion
    path_marker.scale = Vector3(0.07, 0.07, 0.07)  # Line width
    path_marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)  # Red color for path

    if shortest_path:
        for vertex in shortest_path:
            x, y, z, _ = g.vertices[vertex]
            path_marker.points.append(Point(x=x, y=y, z=z))
        markers.markers.append(path_marker)

    marker_pub.publish(markers)
    marker_pub.publish(lines_markers)



class gse:
    
    
        
        
        
        
        
    def calculate_distance(self,point, point2):
        
        # Calculate Euclidean distance between points X and Pi_X
        # Calculate the vector components
        vector_x = point2.x - point.x
        vector_y = point2.y - point.y
        vector_z = point2.z - point.z
    
        magnitude = math.sqrt(vector_x**2 + vector_y**2 + vector_z**2)
        #print("magnitude inside distance function:",magnitude)
    
    
        return magnitude
    
    
    def shape(self,p,x,xobs ):# p- random point , xnearest- nearest point , xbos obstacle list
        
    
        angle_difference=[]#dot product of ri
        ri=[]
        sat_fi_angle=[]
        sat_fi_distance=[] 
    
    
        for _ in range(len(xobs)):
            
        
            angle_difference.append(None)
            ri.append(None)
            sat_fi_angle.append(None)
            sat_fi_distance.append(None)
        
        
        (angle_difference,ri,sat_fi_angle,sat_fi_distance)=self.shape_obstacle_iterator(p,x,xobs )
        #print("fi:",fi)
        #print("ri:",ri)
        ri_value = [int(value) for value in ri]    #ri  int
        sat_fi_int = [int(value) for value in sat_fi_angle]
        sat_fi_distance_int = [int(value) for value in sat_fi_distance]
        #ri_value = [1 if value else 0 for value in ri]
        #sat_fi_int = [1 if value else 0 for value in sat_fi_angle]
        #sat_fi_distance_int = [1 if value else 0 for value in sat_fi_distance]
        # Add corresponding integer values and store in a new list
        fi = [int(a) + int(b) for a, b in zip(sat_fi_int, sat_fi_distance_int)]#fi int
        #print("fi:",fi)
        #print("ri:",ri)
    
        if all(ri):
            
            print("ALL RI IS 1 SO")
            return 0
    
        else:
            
            
         
            gi = []  # Initialize the list for g_i values
            for i in range(1, len(fi) + 1):
                
                # Calculate the summation part for each i
                summation = sum(ri_value[:i-1])
                # Calculate g_i and append to the list
                gi.append(fi[i-1] + summation - (i + 1))

            #print("gi list", gi)

            result = 1  # Start with 1 because it's the identity value for multiplication
            for element in gi:
                
                result *= element
                #print("result", result)

            # Ensure we return the final result correctly
            final_result = result
            print("Final result:", final_result)
            
            return final_result
        
        
    def shape_obstacle_iterator(self,p,x,xobs):
            
        
    
        angle_difference=[]
        ri=[]
        sat_fi_angle=[]
        sat_fi_distance=[]
        rix=[]
        pix=[]
        lix=[]
        nix=[]
    
    
        for _ in range(len(xobs)):
            
        
            angle_difference.append(None)
            ri.append(None)
            sat_fi_angle.append(None)
            sat_fi_distance.append(None)
            
            
        for i in range(len(xobs)):
            
            rix.append(xobs[f"xobs{i}"][0])
            pix.append(xobs[f"xobs{i}"][1])
            lix.append(xobs[f"xobs{i}"][2])
            nix.append(xobs[f"xobs{i}"][3])
        
        #print("shape iterator")
        #print("rix:",rix)
        #print("pix:",pix)
        #print("lix:",lix)
        #print("nix:",nix)
        
            
        
        
        
    
        #return ri,sat_ri,sat_fi,sat_fi_distance
    
        for i in range(len(xobs)):
            
            
        
            #print("pix_list[i]",xobs[f"xobs{i}"])
            first_element = pix[i]
            #print(first_element)
            #print(first_element.x)
            #print(first_element.y)
            #print(first_element.z)
            pix_point_point = Point()
            pix_point_point.x = (first_element.x -0)
            pix_point_point.y = (first_element.y -0)
            pix_point_point.z = (first_element.z -0)
            #ri_X = self.calculate_distance(x, pix_point_point)
            #print("Minimum distance from point X to obstacle i (ri_X):", ri_X)

            #height = float(ri_X)
            height = float(rix[i])
            #radius = float(input("Enter the radius of the cone: "))
            #radius = 0.5
            radius = float(lix[i])
            theta_i = math.atan((radius / height))
            #print("angletheta_i:",theta_i)
        
        
            (angle_difference[i],ri[i],sat_fi_angle[i],sat_fi_distance[i])=self.shape_classifier(p,x,pix_point_point,nix[i],theta_i )
            frame_id = "world"
            shape_publisher = MarkerPublisher()
            #shape_publisher.marker_pub1.publish(x, pix_point_point, height, radius)#
            shape_publisher.publish_two_points(x, pix_point_point)
            
            shape_publisher.publish_vector_between_points(x, pix_point_point)
            
            orientation, slant_height, hsc, rsc, rix_c, half_angle, rcb, hc = shape_publisher.calculate_orientation_and_geometry(x, pix_point_point, lix[i])
            
                
            cone_marker = shape_publisher.create_cone_marker(frame_id, 0, x, orientation, hc, rcb)
            cap_marker = shape_publisher.create_spherical_cap_marker(frame_id, 1, x, orientation, slant_height, hsc, rsc, rix_c)
            
            #marker = shape_publisher.publish_marker(x, pix_point_point, height, radius)
            shape_publisher.marker_pub1.publish(cone_marker)
            shape_publisher.marker_pub1.publish(cap_marker)
            #vis=input("gse shape press enter")
            #time.sleep(0.4)
            # Code execution resumes after 5 seconds
            #print("Delay complete.")
        
        #print("angle_difference",angle_difference)
       # print("ri",ri)
        #print("sat_fi_angle",sat_fi_angle)
        #print("sat_fi_distance",sat_fi_distance)
    
        return angle_difference,ri,sat_fi_angle,sat_fi_distance
    
    
    
    
    def shape_classifier(self,p,x,xobs,nix, theta_i):
    
        
   
        ri=0
        sat_fi_distance=0
        #print("shape classifer")
    
       # print("start points:",x.x,x.y,x.z)
       # print("goal points",p.x,p.y,p.z)
    
        #nix=Vector3()
        #nix.x= (xobs.x - x.x)
        #nix.y= (xobs.y - x.y)
        #nix.z= (xobs.z - x.z)
        
       
    
        #vector for startpoint and random point
        xp=Vector3()#(P-X)
        xp.x=(p.x - x.x)
        xp.y=(p.y - x.y)
        xp.z=(p.z - x.z)
       # print("v_i_x",xp.x,xp.y,xp.z)
    
        # Define the components of vectors A and B
        A_nix = [nix.x, nix.y,nix.z]
        B_xp = [xp.x, xp.y, xp.z]

        # Calculate the dot product of vectors A and B
        dot_product = sum(a * b for a, b in zip(A_nix, B_xp))
       #print("dot_product",dot_product)

        # Calculate the magnitudes of vectors A and B
        magnitude_A_nix = (sum(a**2 for a in A_nix))**0.5
        magnitude_B_xp = (sum(b**2 for b in B_xp))**0.5
    
        #print("magnitude_ni_X: ",magnitude_A_nix )
    
        #print("magnitude_B_xp: ",magnitude_B_xp)
        
        # Check for zero magnitudes to avoid division by zero
        if magnitude_A_nix == 0 or magnitude_B_xp == 0:
            
            #print("One of the vectors is a zero vector; cannot compute cosine of the angle.")
            cos_theta = 0  # or any other appropriate value or handling
        else:
            cos_theta = dot_product / (magnitude_A_nix * magnitude_B_xp)

    
        # Calculate the angle in radians
       # cos_theta = (dot_product / (magnitude_A_nix * magnitude_B_xp))
    
        #cos_theta = max(min(cos_theta, 1.0), -1.0)
    
        theta_radians = math.acos(cos_theta)

        # Convert the angle to degrees
        theta_degrees = math.degrees(theta_radians)

        #print("Angle between vectors A and B (in radians):", theta_radians)
        #print("Angle between vectors A and B (in degrees):", theta_degrees)

        # Calculate ri
        angle_difference = (theta_radians - theta_i)
        #ri = np.arccos((dot_product) / ((magnitude_ni_X) * (magnitude_V_i_X))) - (theta_i)
        #print("ri:",angle_difference )
    
  
    
    
        #ri = angle - theta_i

        # Determine if Pi_X lies within the angle theta_i
        #if ri <= theta_i:
        if angle_difference < 0:
            
            #print("Point Pi_X lies within the angle theta_i")
            ri=False
        else:
            
            #print("Point Pi_X lies outside the angle theta_i")
            ri=True
    
        if magnitude_A_nix > magnitude_B_xp:
            
            #print("magnitude_ni_X > magnitude_B_xp: inside range")
            sat_fi_distance=True
        else:
            
           # print("magnitude_ni_X < magnitude_B_xp: outside range")
            sat_fi_distance=False
    
        sat_fi_angle = not ri
        return angle_difference,ri,sat_fi_angle,sat_fi_distance
    
    
    def xobstacle(self,x,obstacles_info):
        
        
        #print("indise obs------------------------------")
        
        for key, value in obstacles_info.items():
            
            print(f"{key}: {value}")
            
        #print("outside obs------------------------------")
        
        num_keys = len(obstacles_info)

        # Extract 'outside_point' values and 'vertical_size' for each obstacle
        xobs = []
        offset = []
        for obstacles_data in obstacles_info.values():
            
            
            
            xobs.append(obstacles_data['outside_point'])
            #offset.append(obstacles_data['vertical_size'])
            #if (obstacles_data['vertical_size']>obstacles_data['horizontal_size']):
                
                #offset.append(obstacles_data['vertical_size'])
                
            #else:
                
                #offset.append(obstacles_data['horizontal_size'])
            offset.append(obstacles_data['horizontal_size'])

        #print(f"Number of obstacles: {num_keys}")
        #print(f"Outside points: {xobs}")
        #print(f"Vertical sizes: {offset}")
        
        
        
        
        
        xobs_dict = {}
        
        for i in range(len(xobs)):
            
            key = f"xobs{i}"
            first_element = xobs[i]
            pix=Point()
            pix.x=(first_element[0])
            pix.y=(first_element[1])
            pix.z=(first_element[2])
            #pix=scale_vector(pix, 0.5)
            rix=self.calculate_distance(x, pix)
            lix=0.5#dummy
            nix=Vector3()
            nix.x= (first_element[0] - x.x)
            nix.y= (first_element[1] - x.y)
            nix.z= (first_element[2] - x.z)
            #nix=scale_vector(nix, 0.5)
            
            #xobs_dict[key] = [rix, pix,lix,nix]
            #xobs_dict[key] = [rix, pix,offset[i],nix]
            xobs_dict[key] = [rix, pix,(offset[i]+0.0),nix]
            
            #print("obstacle data:", xobs_dict)
            
            
            
        return xobs_dict
    
    def get_center_outside(self, random_point, center, horizontal_size):
        
        offset = 0.5

        # Convert Point objects to NumPy arrays
        random_point_array = np.array([random_point.x, random_point.y, random_point.z])
        center_array = np.array([center.x, center.y, center.z])

        # Calculate the direction vector from the random point to the center of the obstacle
        direction_vector = random_point_array - center_array
        direction_vector[2] = 0  # Ensure the direction vector remains at the same height as the obstacle center
        
        if np.linalg.norm(direction_vector) > 0:
            direction_vector /= np.linalg.norm(direction_vector)

        # Calculate the distance from the center to the edge of the cuboid along the direction vector
        distance_to_edge = horizontal_size / 2.0 + offset

        # Calculate the outside point
        outside_point_rd = center_array + direction_vector * distance_to_edge

        #rospy.loginfo("Outside point (x, y, z) = [%.3f, %.3f, %.3f]" % (outside_point_rd[0], outside_point_rd[1], outside_point_rd[2]))
        #return Point(outside_point_rd[0], outside_point_rd[1], outside_point_rd[2])

        return outside_point_rd
    
    
    






    def get_center_outside1(self, drone_position, center, size, epsilon=0.1):
        
        
        # Convert Point objects to NumPy arrays for mathematical operations
        drone_pos = np.array([drone_position.x, drone_position.y, drone_position.z])
        center_pos = np.array([center.x, center.y, center.z])

        # Calculate the direction vector from the center to the drone's position
        direction = drone_pos - center_pos

        # Normalize the direction vector to have a unit length
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        else:
            # If the drone is exactly at the center, choose an arbitrary direction (e.g., positive x-axis)
            direction = np.array([1.0, 0.0, 0.0])

        # Define the axis-aligned bounding box (AABB) based on the center and size
        # Assuming 'size' is the largest dimension; other dimensions can be scaled accordingly if needed
        half_size = size / 2.0
        bbox_min = center_pos - half_size
        bbox_max = center_pos + half_size

        # Ray-box intersection algorithm (Slab Method)
        t_min = -np.inf
        t_max = np.inf

        for i in range(3):  # Iterate over x, y, z axes
            if direction[i] != 0:
                t1 = (bbox_min[i] - center_pos[i]) / direction[i]
                t2 = (bbox_max[i] - center_pos[i]) / direction[i]

                t_near = min(t1, t2)
                t_far = max(t1, t2)

                t_min = max(t_min, t_near)
                t_max = min(t_max, t_far)
            else:
                # If the direction is parallel to the slab and the origin is not within the slab, no intersection
                if center_pos[i] < bbox_min[i] or center_pos[i] > bbox_max[i]:
                    rospy.logwarn(f"Ray parallel to axis {i} and outside the bounding box.")
                    # Place the outside point at a default distance along this axis
                    t_min = t_max = half_size + epsilon

        if t_max < t_min:
            # No valid intersection; place the outside point at a default distance
            rospy.logwarn("No valid intersection found. Placing outside point at default distance.")
            distance_to_edge = half_size + epsilon
        else:
            # Intersection occurs; place the outside point slightly beyond the intersection
            distance_to_edge = t_max + epsilon

        # Compute the outside point
        outside_point = center_pos + direction * distance_to_edge

        # Optional: Log the outside point for debugging purposes
        rospy.loginfo(f"Computed Outside Point: [{outside_point[0]:.3f}, {outside_point[1]:.3f}, {outside_point[2]:.3f}]")

        return outside_point







    def get_center_outside2(self, drone_position, center, box_min,box_max):
        """
        Computes a point outside the bounding box (defined by 'vertices' = [bbox_min, bbox_max]) 
        along the line from 'center' to 'drone_position' with enough lateral clearance.

        Parameters:
        -----------
        self           : reference to the class instance (typical for a class method).
        drone_position : array-like or object with .x, .y, .z (3D coordinate of the drone).
        center         : array-like or object with .x, .y, .z (3D coordinate of the bounding box center).
        vertices       : list of two elements: [bbox_min, bbox_max], each a [x, y, z] array.

        Returns:
        --------
        outside_point  : np.ndarray
            A NumPy array [x_out, y_out, z_out], guaranteed to lie outside the bounding box 
            and in front of the drone, with enough lateral clearance.
        """
        # Convert bounding box min/max
        #bbox_min, bbox_max = vertices
        bbox_min = np.array(box_min, dtype=float)
        bbox_max = np.array(box_max, dtype=float)

        # Convert drone_position and center to NumPy arrays
        drone_pos = np.array([drone_position.x, drone_position.y, drone_position.z], dtype=float)
        center_bbox = np.array([center.x, center.y, center.z], dtype=float)

        # Internal parameters for the logic (larger buffer ensures outside point is not close)
        initial_buffer  = 1.0
        buffer_increment = 0.5
        max_iterations   = 20
        delta            = 0.1  # Lateral clearance distance

        # Compute the direction vector from the bounding box center to the drone
        direction_vector = drone_pos - center_bbox
        norm_dir = np.linalg.norm(direction_vector)
        if norm_dir == 0:
            raise ValueError("Drone position coincides with bounding box center. No valid direction.")

        direction_unit = direction_vector / norm_dir

        # Helper function: Ray-box intersection using the Slab Method
        def compute_ray_bbox_intersection(ray_origin, ray_dir, bmin, bmax):
            t_min = -np.inf
            t_max_val = np.inf
            for i in range(3):
                if ray_dir[i] != 0:
                    t1 = (bmin[i] - ray_origin[i]) / ray_dir[i]
                    t2 = (bmax[i] - ray_origin[i]) / ray_dir[i]
                    t_entry = min(t1, t2)
                    t_exit  = max(t1, t2)
                    t_min = max(t_min, t_entry)
                    t_max_val = min(t_max_val, t_exit)
                else:
                    # If parallel but origin is outside this axis range, no intersection
                    if ray_origin[i] < bmin[i] or ray_origin[i] > bmax[i]:
                        return None
            if t_max_val < t_min or t_max_val < 0:
                return None
            return t_max_val

        t_max = compute_ray_bbox_intersection(center_bbox, direction_unit, bbox_min, bbox_max)
        if t_max is None:
            # If no intersection, fallback to zero intersection
            t_max = 0.0

        # Helper to generate two perpendicular vectors to direction_unit
        def compute_perpendicular_vectors(dir_unit):
            if abs(dir_unit[0]) < 1e-6 and abs(dir_unit[1]) < 1e-6:
                arbitrary = np.array([1, 0, 0], dtype=float)
            else:
                arbitrary = np.array([0, 0, 1], dtype=float)
            p1 = np.cross(dir_unit, arbitrary)
            p1 /= np.linalg.norm(p1)
            p2 = np.cross(dir_unit, p1)
            p2 /= np.linalg.norm(p2)
            return p1, p2

        perp1, perp2 = compute_perpendicular_vectors(direction_unit)

        # Helper to check if a point is outside the bounding box
        def is_point_outside(pt):
            return np.any(pt < bbox_min) or np.any(pt > bbox_max)

        buffer           = initial_buffer
        iteration        = 0
        best_outside_point = None

        # Iterate, adjusting buffer if needed
        while iteration < max_iterations:
            t_out = t_max + buffer
            # Ensure the outside point doesn't go beyond the drone
            if t_out > norm_dir:
                t_out = norm_dir

            outside_point = center_bbox + t_out * direction_unit

            # Must be outside bounding box
            if not is_point_outside(outside_point):
                buffer += buffer_increment
                iteration += 1
                continue

            # Lateral clearance check
            moved_points = [
                outside_point + delta * perp1,
                outside_point - delta * perp1,
                outside_point + delta * perp2,
                outside_point - delta * perp2
            ]

            intersection_found = False
            for pt in moved_points:
                if not is_point_outside(pt):
                    intersection_found = True
                    break

            if not intersection_found:
                return outside_point  # Found valid outside point
            else:
                buffer += buffer_increment
                iteration += 1
                best_outside_point = outside_point

        # If no valid outside point found after max_iterations
        if best_outside_point is not None:
            print("Warning: Max buffer adjustments reached. Returning best found outside point.")
            return best_outside_point
        else:
            print("Warning: Could not find a suitable outside point. Returning fallback.")
            return center_bbox + (t_max + buffer) * direction_unit



    

   

    def get_center_outside3(self, drone_position, center, box_min, box_max):
        """
        Computes a point outside the bounding box (defined by [bbox_min, bbox_max]) 
        along the line from 'center' to 'drone_position' with enough lateral clearance
        and ensuring the outside point is closer to the drone than any bounding box vertex.

        Parameters:
        -----------
        self           : reference to the class instance.
        drone_position : object with .x, .y, .z or array-like. 
                        3D coordinate of the drone.
        center         : object with .x, .y, .z or array-like.
                        3D coordinate of the bounding box center.
        box_min        : array-like or object with min bounding box corner [x_min, y_min, z_min].
        box_max        : array-like or object with max bounding box corner [x_max, y_max, z_max].

        Returns:
        --------
        outside_point  : np.ndarray
            A NumPy array [x_out, y_out, z_out], guaranteed to lie outside the bounding box, 
            in front of the drone, with lateral clearance. Also ensures the drone->outside point distance 
            is less than drone->any vertex distance.
        """

        # Convert bounding box min/max to NumPy arrays
        bbox_min = np.array(box_min, dtype=float)
        bbox_max = np.array(box_max, dtype=float)

        # Convert drone_position and center to NumPy arrays
        if hasattr(drone_position, 'x'):  # If it's a ROS Point or similar
            drone_pos = np.array([drone_position.x, drone_position.y, drone_position.z], dtype=float)
        else:
            drone_pos = np.array(drone_position, dtype=float)

        if hasattr(center, 'x'):
            center_bbox = np.array([center.x, center.y, center.z], dtype=float)
        else:
            center_bbox = np.array(center, dtype=float)

        # Internal parameters (larger buffer ensures the outside point is not too close)
        initial_buffer  = 1.0
        buffer_increment = 0.5
        max_iterations   = 30
        delta            = 0.1   # Lateral clearance distance

        # 1) Compute all 8 bounding box vertices to enforce distance constraints
        def get_bbox_vertices(bmin, bmax):
            x_min, y_min, z_min = bmin
            x_max, y_max, z_max = bmax
            verts = np.array([
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_min, y_max, z_min],
                [x_max, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_min, y_max, z_max],
                [x_max, y_max, z_max]
            ])
            return verts

        box_vertices = get_bbox_vertices(bbox_min, bbox_max)

        # Distances from drone to each bounding box vertex
        dist_to_vertices = np.linalg.norm(box_vertices - drone_pos, axis=1)
        min_vertex_dist  = np.min(dist_to_vertices)  # The drone->vertex minimum distance

        # 2) Compute direction vector from bounding box center to drone
        direction_vector = drone_pos - center_bbox
        norm_dir = np.linalg.norm(direction_vector)
        if norm_dir == 0:
            raise ValueError("Drone position coincides with bounding box center. No valid direction.")

        direction_unit = direction_vector / norm_dir

        # Helper: Ray-box intersection using the Slab Method
        def compute_ray_bbox_intersection(ray_origin, ray_dir, bmin, bmax):
            t_min = -np.inf
            t_max_val = np.inf
            for i in range(3):
                if abs(ray_dir[i]) > 1e-12:
                    t1 = (bmin[i] - ray_origin[i]) / ray_dir[i]
                    t2 = (bmax[i] - ray_origin[i]) / ray_dir[i]
                    t_entry = min(t1, t2)
                    t_exit  = max(t1, t2)
                    t_min = max(t_min, t_entry)
                    t_max_val = min(t_max_val, t_exit)
                else:
                    # If parallel but origin is outside this axis range, no intersection
                    if ray_origin[i] < bmin[i] or ray_origin[i] > bmax[i]:
                        return None
            if t_max_val < t_min or t_max_val < 0:
                return None
            return t_max_val

        t_max = compute_ray_bbox_intersection(center_bbox, direction_unit, bbox_min, bbox_max)
        if t_max is None:
            t_max = 0.0  # fallback if no intersection

        # Helper to generate two perpendicular vectors for lateral clearance checks
        def compute_perpendicular_vectors(dir_unit):
            if abs(dir_unit[0]) < 1e-6 and abs(dir_unit[1]) < 1e-6:
                arbitrary = np.array([1, 0, 0], dtype=float)
            else:
                arbitrary = np.array([0, 0, 1], dtype=float)
            p1 = np.cross(dir_unit, arbitrary)
            if np.linalg.norm(p1) < 1e-12:
                p1 = np.array([1, 0, 0], dtype=float)
            else:
                p1 /= np.linalg.norm(p1)
            p2 = np.cross(dir_unit, p1)
            p2 /= np.linalg.norm(p2)
            return p1, p2

        perp1, perp2 = compute_perpendicular_vectors(direction_unit)

        # Check outside bounding box
        def is_point_outside(pt):
            return np.any(pt < bbox_min) or np.any(pt > bbox_max)

        buffer           = initial_buffer
        iteration        = 0
        best_outside_point = None

        while iteration < max_iterations:
            t_out = t_max + buffer
            # Ensure the outside point doesn't go beyond the drone
            if t_out > norm_dir:
                t_out = norm_dir

            outside_point = center_bbox + t_out * direction_unit

            # 3) Must lie outside bounding box
            if not is_point_outside(outside_point):
                buffer += buffer_increment
                iteration += 1
                continue

            # 4) Lateral clearance check
            moved_points = [
                outside_point + delta * perp1,
                outside_point - delta * perp1,
                outside_point + delta * perp2,
                outside_point - delta * perp2
            ]
            intersection_found = False
            for pt in moved_points:
                if not is_point_outside(pt):
                    intersection_found = True
                    break
            if intersection_found:
                buffer += buffer_increment
                iteration += 1
                best_outside_point = outside_point
                continue

            # 5) Distance check: drone->outside < min(drone->all box vertices)
            dist_drone_outside = np.linalg.norm(outside_point - drone_pos)
            if dist_drone_outside >= min_vertex_dist:
                # Fails the condition, need more buffer or we can dynamically increment 'delta' maybe
                buffer += buffer_increment
                iteration += 1
                best_outside_point = outside_point
                continue

            # If all conditions pass:
            return outside_point

        # If no valid outside point found after max_iterations
        if best_outside_point is not None:
            print("Warning: Max iterations reached. Returning best found outside point.")
            return best_outside_point
        else:
            print("Warning: Could not find a suitable outside point. Returning fallback.")
            return center_bbox + (t_max + buffer) * direction_unit



    

    
    def xobstacle_near_point(self,x,obstacles_info,random_pt,goal):
        
        
        #print("indise obs------------------------------")
        
        for key, value in obstacles_info.items():
            
            print(f"{key}: {value}")
            
        #print("outside obs------------------------------")
        
        num_keys = len(obstacles_info)

        # Extract 'outside_point' values and 'vertical_size' for each obstacle
        xobs = []
        xobs1 = []
        center=[]
        offset = []
        vertices=[]
        box_min=[]
        box_max=[]
        for obstacles_data in obstacles_info.values():
            
            center.append(obstacles_data['center'])
            
                
            xobs.append(obstacles_data['outside_point'])
            #vertices.append(obstacles_data['vertices'])
            box_min.append(obstacles_data['box_min'])
            box_max.append(obstacles_data['box_max'])
            
            
            #if (obstacles_data['vertical_size']>obstacles_data['horizontal_size']):
                
                #offset.append(obstacles_data['vertical_size'])
                
            #else:
                
                #offset.append(obstacles_data['horizontal_size'])
            offset.append(obstacles_data['horizontal_size'])
                
        #print("-------------------------------------------vertices:",vertices)
        #print("-############################################vertices:",vertices[1])

        #print(f"Number of obstacles: {num_keys}")
        #print(f"Outside points: {xobs}")
       #print(f"Vertical sizes: {offset}")
        #print(f"random outside point: {xobs1}")
        i=0
        for value in center:
            
            a=value
            center_tf=Point(a[0],a[1],a[2])
            
            
            #def get_center_outside1(self, drone_position, center, vertices):
            #xobs1.append(self.get_center_outside1(goal, center_tf, offset[i]))
            #xobs1.append(self.get_center_outside1(goal, center_tf, offset[i]))
            xobs1.append(self.get_center_outside3(goal, center_tf, box_min[i],box_max[i]))
            #xobs1.append(self.get_center_outside2(random_pt, center_tf, box_min[i],box_max[i]))
            i=i+1
        #for value in xobs:
            
        #    a=value
        #    center_tf=Point(a[0],a[1],a[2])
            
        #    xobs1.append(self.get_center_outside(goal, center_tf))
            
        
        
        
        xobs_dict = {}
        xobs_dict1 = {}
        
        for i in range(len(xobs)):
            
            key = f"xobs{i}"
            first_element = xobs[i]
            pix=Point()
            pix.x=(first_element[0])
            pix.y=(first_element[1])
            pix.z=(first_element[2])
            #pix=scale_vector(pix, 0.5)
            rix=self.calculate_distance(x, pix)
            lix=0.5#dummy
            nix=Vector3()
            nix.x= (first_element[0] - x.x)
            nix.y= (first_element[1] - x.y)
            nix.z= (first_element[2] - x.z)
            #nix=scale_vector(nix, 0.5)
            
            #xobs_dict[key] = [rix, pix,lix,nix]
            #xobs_dict[key] = [rix, pix,offset[i],nix]
            xobs_dict[key] = [rix, pix,(offset[i]+0.0),nix]
            
            #print("obstacle data:", xobs_dict)
            
            
        for i in range(len(xobs)):
            
            key = f"xobs{i}"
            first_element = xobs1[i]
            pix=Point()
            pix.x=(first_element[0])
            pix.y=(first_element[1])
            pix.z=(first_element[2])
            #pix=scale_vector(pix, 0.5)
            rix=self.calculate_distance(x, pix)
            lix=0.5#dummy
            nix=Vector3()
            nix.x= (first_element[0] - x.x)
            nix.y= (first_element[1] - x.y)
            nix.z= (first_element[2] - x.z)
            #nix=scale_vector(nix, 0.5)
            
            #xobs_dict[key] = [rix, pix,lix,nix]
            #xobs_dict[key] = [rix, pix,offset[i],nix]
            xobs_dict1[key] = [rix, pix,(offset[i]+0.0),nix]
            
            #print("obstacle data:", xobs_dict)
            
            
            
        return xobs_dict,xobs_dict1
    
    
    def generate_random_point(self):
        
        # Generate a random point within the specified ranges
        random_x = random.randint(0, 15)
        random_y = random.randint(0, 15)
        random_z = random.randint(0, 5)
        return  random_x,  random_y,  3.5
    
    def generate_point_towards_goal(self,robot_pose_x, robot_pose_y, robot_pose_z, goal_x, goal_y, goal_z):
        
        # Vector from sensor to goal
        dx = goal_x - robot_pose_x
        dy = goal_y - robot_pose_y
        dz = goal_z - robot_pose_z

        # Distance to goal
        distance_to_goal = math.sqrt(dx**2 + dy**2 + dz**2)

        # Normalize the direction vector
        direction_x = dx / distance_to_goal
        direction_y = dy / distance_to_goal
        direction_z = dz / distance_to_goal

        # Use the maximum distance within the sensor range (0.1 to 5 meters)
        max_sensor_range = 6
        r = min(distance_to_goal, max_sensor_range)

        # Generate the point in the direction of the goal at the maximum distance
        x_translated = robot_pose_x + direction_x * r
        y_translated = robot_pose_y + direction_y * r
        z_translated = robot_pose_z + direction_z * r

        return x_translated, y_translated, z_translated









class Graph:
    """
    A Graph class to manage vertices and edges, ensuring no duplicate edges are added.
    """
    def __init__(self):
        self.vertices = {}     # Stores vertex data: {vertex_name: (x, y, z, color)}
        self.edges = []        # Stores edges as tuples: (vertex1, vertex2, cost)
        self.edge_set = set()  # Stores sorted tuples of edges for quick duplicate checks

    def add_vertex(self, vertex, x, y, z, color):
        """
        Adds a new vertex to the graph.

        Parameters:
            vertex (str): Identifier for the vertex.
            x (float): X-coordinate.
            y (float): Y-coordinate.
            z (float): Z-coordinate.
            color (ColorRGBA): Color of the vertex.

        Returns:
            bool: True if the vertex was added successfully, False if it already exists.
        """
        if vertex in self.vertices:
            rospy.logwarn(f"Vertex '{vertex}' already exists. Skipping addition.")
            return False  # Vertex already exists
        self.vertices[vertex] = (x, y, z, color)
        rospy.loginfo(f"Vertex '{vertex}' added successfully.")
        return True

    def add_edge(self, start_vertex, end_vertex, cost):
        """
        Adds an undirected edge between two vertices with the given cost.
        Prevents duplicate edges using a set for efficient checks.

        Parameters:
            start_vertex (str): Identifier for the start vertex.
            end_vertex (str): Identifier for the end vertex.
            cost (float): Cost or weight of the edge.

        Returns:
            bool: True if the edge was added successfully, False if it already exists or vertices are missing.
        """
        if start_vertex not in self.vertices:
            rospy.logerr(f"Start vertex '{start_vertex}' does not exist. Cannot add edge.")
            return False  # Cannot add edge if start vertex doesn't exist
        if end_vertex not in self.vertices:
            rospy.logerr(f"End vertex '{end_vertex}' does not exist. Cannot add edge.")
            return False  # Cannot add edge if end vertex doesn't exist

        # Create a sorted tuple to handle undirected edges consistently
        edge = tuple(sorted([start_vertex, end_vertex]))

        if edge in self.edge_set:
            rospy.logwarn(f"Edge between '{start_vertex}' and '{end_vertex}' already exists. Skipping addition.")
            return False  # Duplicate edge detected

        # Add the edge to the graph
        self.edges.append((start_vertex, end_vertex, cost))
        self.edge_set.add(edge)
        rospy.loginfo(f"Edge from '{start_vertex}' to '{end_vertex}' with cost {cost:.2f} added successfully.")
        return True

    def get_neighbors(self, vertex):
        """
        Retrieves all neighbors of a given vertex along with the cost to reach them.

        Parameters:
            vertex (str): The vertex identifier.

        Returns:
            list of tuples: Each tuple contains (neighbor_vertex, cost).
        """
        if vertex not in self.vertices:
            rospy.logerr(f"Vertex '{vertex}' does not exist. Cannot retrieve neighbors.")
            return []  # Return empty list if vertex doesn't exist

        neighbors = []
        for edge in self.edges:
            if edge[0] == vertex:
                neighbors.append((edge[1], edge[2]))  # (neighbor, cost)
            elif edge[1] == vertex:
                neighbors.append((edge[0], edge[2]))  # (neighbor, cost)
        return neighbors

    def remove_vertex(self, vertex):
        """
        Removes a vertex and all connected edges from the graph.

        Parameters:
            vertex (str): The vertex identifier to remove.

        Returns:
            bool: True if the vertex was removed successfully, False if it does not exist.
        """
        if vertex not in self.vertices:
            rospy.logwarn(f"Vertex '{vertex}' does not exist. Cannot remove.")
            return False  # Vertex does not exist

        # Remove the vertex
        del self.vertices[vertex]
        rospy.loginfo(f"Vertex '{vertex}' removed successfully.")

        # Remove all edges connected to this vertex
        edges_to_remove = [edge for edge in self.edges if edge[0] == vertex or edge[1] == vertex]
        for edge in edges_to_remove:
            sorted_edge = tuple(sorted([edge[0], edge[1]]))
            self.edge_set.discard(sorted_edge)
            self.edges.remove(edge)
            rospy.loginfo(f"Edge between '{edge[0]}' and '{edge[1]}' removed due to vertex deletion.")

        return True

    def remove_edge(self, start_vertex, end_vertex):
        """
        Removes an undirected edge between two vertices.

        Parameters:
            start_vertex (str): Identifier for the start vertex.
            end_vertex (str): Identifier for the end vertex.

        Returns:
            bool: True if the edge was removed successfully, False if it does not exist.
        """
        edge = tuple(sorted([start_vertex, end_vertex]))
        if edge not in self.edge_set:
            rospy.logwarn(f"Edge between '{start_vertex}' and '{end_vertex}' does not exist. Cannot remove.")
            return False  # Edge does not exist

        # Remove the edge from the edges list
        for existing_edge in self.edges:
            if tuple(sorted([existing_edge[0], existing_edge[1]])) == edge:
                self.edges.remove(existing_edge)
                break

        # Remove from the edge_set
        self.edge_set.discard(edge)
        rospy.loginfo(f"Edge between '{start_vertex}' and '{end_vertex}' removed successfully.")
        return True

    def display_graph(self):
        """
        Displays the current state of the graph, including all vertices and edges.
        """
        rospy.loginfo("Vertices:")
        for vertex, properties in self.vertices.items():
            x, y, z, color = properties
            rospy.loginfo(f"  {vertex}: Coordinates=({x}, {y}, {z}), Color=({color.r}, {color.g}, {color.b}, {color.a})")

        rospy.loginfo("Edges:")
        if not self.edges:
            rospy.loginfo("  No edges in the graph.")
        else:
            for edge in self.edges:
                rospy.loginfo(f"  {edge[0]} --({edge[2]:.2f})-- {edge[1]}")

    def publish_graph_markers(self, marker_pub):
        """
        Publishes visualization markers for the graph's vertices and edges to RViz.

        Parameters:
            marker_pub (rospy.Publisher): ROS publisher for MarkerArray messages.
        """
        marker_array = MarkerArray()
        marker_id = 0

        # Add markers for vertices
        for vertex, (x, y, z, color) in self.vertices.items():
            marker = Marker()
            marker.header.frame_id = "world"  # Adjust frame as needed
            marker.header.stamp = rospy.Time.now()
            marker.ns = "vertices"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2  # Diameter of the sphere
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = color.r
            marker.color.g = color.g
            marker.color.b = color.b
            marker.color.a = color.a
            marker.lifetime = rospy.Duration(0)  # 0 means forever
            marker_array.markers.append(marker)
            marker_id += 1

        # Add markers for edges
        for edge in self.edges:
            start_vertex, end_vertex, cost = edge
            start_pos = self.vertices[start_vertex][:3]
            end_pos = self.vertices[end_vertex][:3]

            marker = Marker()
            marker.header.frame_id = "world"  # Adjust frame as needed
            marker.header.stamp = rospy.Time.now()
            marker.ns = "edges"
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05  # Line width
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.points = [
                Point(x=start_pos[0], y=start_pos[1], z=start_pos[2]),
                Point(x=end_pos[0], y=end_pos[1], z=end_pos[2])
            ]
            marker.lifetime = rospy.Duration(0)
            marker_array.markers.append(marker)
            marker_id += 1

        # Publish the MarkerArray
        marker_pub.publish(marker_array)









class PDController:
    def __init__(self, kp, kd, max_value, min_value):
        self.kp = kp
        self.kd = kd
        self.max_value = max_value
        self.min_value = min_value
        self.prev_error = 0.0
    
    def calculate(self, setpoint, current_value, dt):
        
        # Error calculation
        error = setpoint - current_value
        
        # Derivative calculation
        derivative = (error - self.prev_error) / dt
        
        # PD output
        output = (self.kp * error) + (self.kd * derivative)
        
        # Clamp output to max and min values
        output = max(self.min_value, min(self.max_value, output))
        
        # Store error for next derivative calculation
        self.prev_error = error
        
        return output    
    
    def publish_graph(g, start, goal, shortest_path):
        
        #rospy.init_node("graph_rviz")
        marker_pub_graph = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
        rate = rospy.Rate(100)  # 1 Hz
        update_markers(g, start, goal, shortest_path, marker_pub_graph)
        rate.sleep()




class MazebotGTG:
    def __init__(self):
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.robot_pose_subscriber = rospy.Subscriber('/ground_truth/state', Odometry, self.get_mazebot_pose)
        self.robot_pose = Point()
        self.goal_pose = Point()
        self.vel_msg = Twist()
        self.distance_to_goal = 0.0
        self.angle_to_goal = 0.0
        self.angular_velocity_scale = 8
        self.linear_velocity_scale = 8
        self.goal_reached_threshold = 0.1
        
         # PD controllers for linear and angular velocity
        self.angular_pd = PDController(kp=2.0, kd=0.2, max_value=1.5, min_value=-1.5)
        self.linear_pd = PDController(kp=1.0, kd=0.1, max_value=2.0, min_value=0.0)  

    def get_mazebot_pose(self, data):
        self.robot_pose.x = data.pose.pose.position.x
        self.robot_pose.y = data.pose.pose.position.y
        self.robot_pose.z = data.pose.pose.position.z
        quaternion = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w)
        (_, _, self.robot_pose_yaw) = self.euler_from_quaternion(*quaternion)

    def return_pose(self):
    
        robot_x=self.robot_pose.x
        robot_y=self.robot_pose.y
        robot_z=self.robot_pose.z
        robot_yaw=self.robot_pose_yaw
        return robot_x,robot_y,robot_z,robot_yaw

    def goal_movement(self, goal_x, goal_y):
        
        self.goal_pose.x = goal_x
        self.goal_pose.y = goal_y

        rate = rospy.Rate(100)  # 30 Hz
        prev_time = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            current_time = rospy.Time.now().to_sec()
            dt = current_time - prev_time
            if dt == 0:
                
                continue

            # Calculate distance to goal
            self.distance_to_goal = sqrt((self.goal_pose.x - self.robot_pose.x)**2 + (self.goal_pose.y - self.robot_pose.y)**2)
            
            # Calculate angle to goal
            self.angle_to_goal = atan2(self.goal_pose.y - self.robot_pose.y, self.goal_pose.x - self.robot_pose.x)
            
            # Get current yaw from quaternion
            current_yaw = self.robot_pose_yaw
            
            # Calculate angle difference
            angle_difference = self.angle_to_goal - current_yaw
            
            # Normalize angle difference to be within -pi to pi
            while angle_difference > pi:
                angle_difference -= 2 * pi
            while angle_difference < -pi:
                angle_difference += 2 * pi

            # Calculate angular velocity using PD controller
            angular_velocity = self.angular_pd.calculate(setpoint=self.angle_to_goal, current_value=current_yaw, dt=dt)

            # Linear velocity control
            if abs(angle_difference) < 0.3:  # Only move forward if almost aligned
                linear_velocity = self.linear_pd.calculate(setpoint=self.distance_to_goal, current_value=0, dt=dt)
            else:
                linear_velocity = 0  # Stop moving forward if not aligned

            # Update the velocities
            self.vel_msg.angular.z = 1*angular_velocity
            self.vel_msg.linear.x = 1*linear_velocity

            # Publish the velocities
            self.velocity_publisher.publish(self.vel_msg)

            #rospy.loginfo(f"Current Yaw: {current_yaw}, Goal Angle: {self.angle_to_goal}, Angular Vel: {angular_velocity}, Linear Vel: {linear_velocity}")

            # Check if the goal is reached
            if self.distance_to_goal < self.goal_reached_threshold:
                self.vel_msg.linear.x = 0
                self.vel_msg.angular.z = 0
                self.velocity_publisher.publish(self.vel_msg)
                rospy.loginfo("Reached the goal!")
                rospy.sleep(0.3)
                return 1
                break

            prev_time = current_time
            rate.sleep()
            
    def euler_from_quaternion(self, x, y, z, w):
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)

        return roll_x, pitch_y, yaw_z


def euclidean_distance(point1, point2):
    
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def euclidean_distance_point(point1, point2):
    return math.sqrt((point2.x - point1.x)**2 + 
                     (point2.y - point1.y)**2 + 
                     (point2.z - point1.z)**2)
    
def euclidean_distance_near(point1, point2):
    
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5

    


def is_connected(graph, start, goal):
    visited = set()
    stack = [start]

    while stack:
        vertex = stack.pop()
        if vertex == goal:
            return True
        if vertex not in visited:
            visited.add(vertex)
            neighbors = graph.get_neighbors(vertex)
            for neighbor, _ in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
    return False

def dijkstra(graph, start, goal):
    queue = [(0, start, [])]  # (cost, current_vertex, path)
    visited = set()

    while queue:
        cost, current, path = heapq.heappop(queue)
        if current not in visited:
            visited.add(current)
            path = path + [current]
            if current == goal:
                return path
            for neighbor, neighbor_cost in graph.get_neighbors(current):
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + neighbor_cost, neighbor, path))
    return None



def steer(first_nearest_vertex,second_nearest_vertex,x1nearest,x2nearest,xrand,graph, marker_pub,node_name,i):
    #steer(first_nearest_vertex,second_nearest_vertex,first_nearest_coords,second_nearest_coords,xrand,g, marker_pub,node_name,i)
    
    vertex_name =node_name
    x = xrand[0]
    y = xrand[1]
    z = xrand[2]
    r = 0
    g_value = 0
    b = 1
    a = 1
    point = (x, y, z)
    #first_nearest_vertex, first_nearest_coords, second_nearest_vertex, second_nearest_coords = nearest(graph, point)
    #print("First nearest point:", first_nearest_vertex, "with coordinates:", first_nearest_coords)
    #print("Second nearest point:", second_nearest_vertex, "with coordinates:", second_nearest_coords)
    
    color = ColorRGBA(r, g_value, b, a)
    graph.add_vertex(vertex_name, x, y, z, color)
    update_markers(graph, None, None, None, marker_pub)
    add_edge(graph, marker_pub, vertex_name,x1nearest,x2nearest,i,xrand,first_nearest_vertex,second_nearest_vertex)
    return vertex_name, first_nearest_vertex, x1nearest, second_nearest_vertex, x2nearest

#def add_edge(graph, marker_pub, vertex_name, first_nearest_vertex, second_nearest_vertex):
def add_edge(graph, marker_pub, vertex_name,x1nearest,x2nearest,i,xrand,first_nearest_vertex,second_nearest_vertex):
    if len(graph.vertices) < 2:
        print("At least two vertices are required to add an edge.")
        return
    
    
        
    first_nearest = vertex_name
    second_nearest = first_nearest_vertex
    
    #edge_cost_to_first_vertex = euclidean_distance(xrand, xrand)
    #edge_cost_to_second_vertex = euclidean_distance(xrand, x2nearest)
    edge_cost_to_second_vertex = euclidean_distance_near(xrand, x1nearest)
    print("inside edge x1nearest:",x1nearest)
    print("inside edge x2nearest:",x2nearest)
    #edge_cost_to_first_vertex = 1
    #edge_cost_to_second_vertex = 1
    
    
    #graph.add_edge(vertex_name, first_nearest, edge_cost_to_first_vertex)
    graph.add_edge(first_nearest, second_nearest, edge_cost_to_second_vertex)
        
    update_markers(graph, None, None, None, marker_pub)
    print("steer confirm")
    #rand=input("steer press enter to contiue ") 
    #time.sleep(5)
    
def add_edge_only(graph, marker_pub, vertex_name,x1nearest,i,xrand,first_nearest_vertex):
    if len(graph.vertices) < 2:
        print("At least two vertices are required to add an edge.")
        return
    
    
        
    first_nearest = vertex_name
    second_nearest = first_nearest_vertex
    
    edge_cost_to_first_vertex = euclidean_distance_near(xrand, xrand)
    edge_cost_to_second_vertex = euclidean_distance_near(xrand, x1nearest)
    print("inside edge x2nearest:",x1nearest)
    #edge_cost_to_first_vertex = 1
    #edge_cost_to_second_vertex = 1
    
    
    #graph.add_edge(vertex_name, first_nearest, edge_cost_to_first_vertex)
    graph.add_edge(first_nearest, second_nearest, edge_cost_to_second_vertex)
        
    update_markers(graph, None, None, None, marker_pub)

def get_vertex_positions(graph):
    vertex_positions = {}
    for vertex, (x, y, z, _) in graph.vertices.items():
        vertex_positions[vertex] = (x, y, z)
    return vertex_positions 

def nearest_intersected_shape(robot_pose_xnear,obstacles_info1,f,random_point,xobs,gse_sample,g,marker_pub,node_name,xrand):
    gi_temp=None
    #flag=input("inside near intersect  START")
    for index, (key, value) in enumerate(f.items()):
        
                    
        if index == len(f) - 1:
            
            break
                    
        print(f"Key: {key}, Value: {value}")
                    
        print("key:",key)                                                                                                                                                   
        print("value:",value)
                    #time.sleep(20)
                    
        value_point=Point()
        value_point.x=value[0]
        value_point.y=value[1]
        value_point.z=value[2]
                    
        #xobs,xobs1=gse_sample.xobstacle_near_point(robot_pose_xnear,obstacles_info1,random_point,value_point) 
        xobs,xobs1=gse_sample.xobstacle_near_point(value_point,obstacles_info1,random_point,value_point) 
        #xobs,xobs1=gse_sample.xobstacle_near_point(robot_pose_xnear,obstacles_info1,value_point,random_point)          
        #gi_temp= gse_sample.shape(random_point,value_point,xobs1 )
        print("graph:",f)
        print("random point",random_point)
        print(f"Key: {key}, Value: {value}")
        
        print("flag value_point,random_point",value_point,random_point)
        
        #gi_temp= gse_sample.shape(value_point,random_point,xobs1 )
        gi_temp= gse_sample.shape(random_point,value_point,xobs1 )
        print("gi_vale",gi_temp)
        #flag1=input("inside near intersect gi value")
                    
        if gi_temp==0:
            
            #add_edge_only(graph, marker_pub, vertex_name,x1nearest,i,xrand,first_nearest_vertex,second_nearest_vertex)
            add_edge_only(g, marker_pub, node_name,value,i,xrand,key)
            #graph_to_string(g)
            print("nearest confirm")
            #time.sleep(10)
            #flag=input("END near intersect")


def nearest(g, point):
    # Initialize the first and second nearest distances to a large number
    first_nearest_dist = float('inf')
    second_nearest_dist = float('inf')
    
    # Initialize the first and second nearest vertices and coordinates
    first_nearest_vertex = None
    first_nearest_coords = None
    second_nearest_vertex = None
    second_nearest_coords = None
    
    for vertex, (x, y, z, _) in g.vertices.items():
        distance = euclidean_distance_near((x, y, z), point)
        
        if distance < first_nearest_dist:
            # Update second nearest before updating the first nearest
            second_nearest_dist = first_nearest_dist
            second_nearest_vertex = first_nearest_vertex
            second_nearest_coords = first_nearest_coords
            
            # Update first nearest
            first_nearest_dist = distance
            first_nearest_vertex = vertex
            first_nearest_coords = (x, y, z)
        elif distance < second_nearest_dist:
            # Update second nearest only
            second_nearest_dist = distance
            second_nearest_vertex = vertex
            second_nearest_coords = (x, y, z)
    
    return first_nearest_vertex, first_nearest_coords, second_nearest_vertex, second_nearest_coords



def generate_random_point_in_sphere(start, goal):
    """Generate a random point inside the sphere defined by start and goal points."""
    # Calculate the center of the sphere
    center = [(start[0] + goal[0]) / 2,
              (start[1] + goal[1]) / 2,
              (start[2] + goal[2]) / 2]
    
    # Calculate the radius (half the distance between start and goal points)
    radius = np.linalg.norm(np.array(goal) - np.array(start)) / 2

    # Generate a random point inside the sphere
    while True:
        x = random.uniform(-radius, radius)
        y = random.uniform(-radius, radius)
        z = random.uniform(-radius, radius)

        # Check if the point is inside the sphere
        if x**2 + y**2 + z**2 <= radius**2:
            #return [center[0] + x, center[1] + y, center[2] + z]
            return Point(center[0] + x, center[1] + y, 3.5)
            break
        

def calculate_spherical_sector_angles(X, pix, lix):
    # Calculate the distance between X and pix
    distance = math.sqrt((pix.x - X.x)**2 + (pix.y - X.y)**2 + (pix.z - X.z)**2)
    
    # Central azimuthal angle (phi_a)
    phi_a = math.atan2(pix.y - X.y, pix.x - X.x)
    
    # Constrain lix / distance within the range [-1, 1] for asin function
    asin_input = lix / distance
    if asin_input < -1.0:
        asin_input = -1.0
    elif asin_input > 1.0:
        asin_input = 1.0

    # Azimuthal angle range (d_phi)
    d_phi = 2 * math.asin(asin_input)
    
    # Central polar angle (theta_a)
    theta_a = math.acos((pix.z - X.z) / distance)
    
    # Polar angle range (d_theta)
    d_theta = 2 * math.asin(asin_input)
    
    return phi_a, d_phi, theta_a, d_theta

def generate_random_point_in_spherical_sector(apex, rix, phi_a, d_phi, theta_a, d_theta):
    r = random.uniform(0, rix)
    phi = phi_a + random.uniform(-d_phi / 2, d_phi / 2)
    theta = theta_a + random.uniform(-d_theta / 2, d_theta / 2)

    # Spherical to Cartesian conversion
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)

    # Translate point by the apex coordinates
    translated_x = x + apex.x
    translated_y = y + apex.y
    translated_z = z + apex.z
    point = (int(translated_x), int(translated_y), 3.5)
    return point


class ObstacleDataSubscriber:
    def __init__(self, topic_name):
        """Initialize the subscriber to receive data on request."""
        self.data_available = Event()  # Event to signal data availability
        self.latest_data = None  # Store the latest received data
        self.subscriber = rospy.Subscriber(topic_name, String, self._callback)

    def _callback(self, msg):
        """Callback to store the received obstacle data."""
        #rospy.loginfo(f"Raw obstacle data: {msg.data}")
        try:
            # Sanitize and parse the JSON string to a dictionary
            sanitized_data = self.sanitize_data(msg.data)
            self.latest_data = json.loads(sanitized_data)
            self.data_available.set()  # Signal that data is available
        except json.JSONDecodeError as e:
            rospy.logerr(f"Failed to parse JSON: {e}")

    @staticmethod
    def sanitize_data(data):
        """Ensure the data has valid JSON-like syntax."""
        # Replace single quotes with double quotes (if any)
        return data.replace("'", '"')

    def get_latest_data(self):
        """Block until new data is available, then return it."""
       # rospy.loginfo("Waiting for obstacle data...")
        self.data_available.wait()  # Block until data is received
        self.data_available.clear()  # Reset the event for future calls
        #rospy.loginfo(f"Latest obstacle data: {self.latest_data}")
        return self.latest_data



def main():
    try:
        global marker_pub, listener
        rospy.init_node("obstacle_detection_node")
        marker_publisher = MarkerPublisher()#done
        mazebot_gtg = MazebotGTG()#done
        
        marker_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
        pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
        rospy.Subscriber("/velodyne_points", PointCloud2, pointcloud_callback)
        rospy.Subscriber("/ground_truth/state", Odometry, get_robot_pose)
        rospy.sleep(0.1)  # Give some time to receive the messages
        
        
        rate = rospy.Rate(100)
        print("-----------------------START----------------------------")
        
        gse_sample = gse()#done
        subscriber = ObstacleDataSubscriber("obstacle_data")
        
        global i , j
        i=0
        j=0

     

        gial_no=0
        coordinates = [(4,11,3.5), (42,12,3.5),(43,42,3.5),(2,40,3.5),(4,11,3.5)]
        #coordinates = [(-37,1,0), (-37,20,0),(-58,20,0),(-58,1,0),(-37,1,0)]
        #coordinates = [(-42,-9,3), (-42,29,3),(-70,29,3),(-70,-10,3),(-42,-9,3)]

        for a, b, c in coordinates:
            
          
            goal_vertex = f"node{gial_no}"   
            x2 = float(a)
            y2 = float(b)
            z2 = float(3.5)
            goal_point=Point(x2,y2,z2)
            
            
            waypoints={}
            
            
            
            marker_publisher.publish_axes()
            
            while not rospy.is_shutdown():
                marker_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
                #rate = rospy.Rate(1)
                rospy.sleep(0.1)  # Give some time to receive the messages
                rate = rospy.Rate(100)# 1 H
                j=0
                i=i+1
               
                node_name = f"node{i},{j}"

                
                pos_x,pos_y,pos_z,pos_yaw=mazebot_gtg.return_pose()
                pos_z=z2
                print("--------------------new position ----------------------------------------------")
                #time.sleep(5)
                point_a = (pos_x, pos_y)
                point_b = (x2, y2)
                distance = euclidean_distance(point_a, point_b)#euclidean_distance_apex_to_goal/tuple
                print("iteration i number",i)
                if distance<=1:
                    print("\nReached goal")
                    break
                listener = tf.TransformListener()

                rospy.sleep(0.5)  # Give some time to receive the messages
            
                obstacles_info = get_obstacles_info()
                
                #obstacles_info = subscriber.get_latest_data()
                
                print(obstacles_info)
                ##time.sleep(10)
                
                if (obstacles_info==0):
                    print("inside if")
                    
                    print("----------------no obstacle block----------")
                    j=j+1
                    print("iteration i number",i)
                    print("iteration j number",j)
                    #random_x,random_y,random_z=gse_sample.generate_random_point(pos_x, pos_y,pos_z)
                    random_x,random_y,random_z=gse_sample.generate_point_towards_goal(pos_x,pos_y,pos_z, x2, y2, pos_z)
                    
                    random_point = Point()
                    random_point.x = (random_x)
                    random_point.y = (random_y)
                    random_point.z = (random_z)
                    marker_publisher.publish_single_point(random_point)
                        
                    obs1_values = (random_x,random_y,random_z) 
                    waypoints[node_name] = obs1_values
                    print("waypoints:",waypoints)
                    
                                    
                    reached=mazebot_gtg.goal_movement(random_x,random_y)
                    if (reached==0):
                        #rand=input("if (reached==0): press enter to contiue ")    
                            
                                        
                        print("node_name:",node_name)
                        print("coordinate:",random_point)
                        print("reached!!!!!")
                        
                    distance_goal = euclidean_distance(point_a, point_b)
                    
                    if distance_goal<=5:
                        print("\nInside sensor zone")
                        rand=input("distance_goal<=5: press enter to contiue ")  
                        reached_goal=mazebot_gtg.goal_movement(x2,y2)
                        if (reached_goal==0):
                            
                            print("node_name:",node_name)
                            print("coordinate:",random_point)
                        
                        
                            print("\nReached goal")
                            
                else:
                    
                    
                    g = Graph()#
                    r2 = 0
                    g_value2 = 0
                    b2 = 1
                    a2 = 1
                    goal_color = ColorRGBA(r2, g_value2, b2, a2)
                    g.add_vertex(goal_vertex, x2, y2, z2, goal_color)
                    
                    while True:
                        
                        #rand=input("inside while press enter to contiue ") 
                        j=j+1
                        print("j",j)
                        #time.sleep(2)
                        
                        node_name = f"node{i},{j}"
                
                
                        pos_x,pos_y,pos_z,pos_yaw=mazebot_gtg.return_pose()
                        robot_pose_xnear=Point(pos_x,pos_y,pos_z)
                        print("pos_x",pos_x)
                        print("pos_y",pos_y)
                        print("pos_z",pos_z)
                        print("pos_yaw",pos_yaw)
                        
                        start_vertex = "s"
                        x1 = float(pos_x)
                        y1= float(pos_y)
                        z1 = float(pos_z)
                        r1 = 0
                        g_value1 = 0
                        b1 = 1
                        a1 = 1
                        start_color = ColorRGBA(r1, g_value1, b1, a1)
                        g.add_vertex(start_vertex, x1, y1, z1, start_color)
            
                        x = Point()
                        x.x = (x1-0)
                        x.y = (y1-0)
                        x.z = (z1-0)
                        
                        point_a = (pos_x, pos_y)
                        point_b = (x2, y2)
                        start_rand=(pos_x, pos_y,3.5)
                        goal_rand=(x2, y2,3.5)
                        if is_connected(g, start_vertex, goal_vertex):#
                            print("\nConnected")
                            #time.sleep(5)
                            break
                        
                        print("\nNo path found. Please add more vertices and edges.")
                        #xobs_to_goal=gse_sample.xobstacle(robot_pose_xnear,obstacles_info)
                        xobs0,xobs_to_goal=gse_sample.xobstacle_near_point(robot_pose_xnear,obstacles_info,1,robot_pose_xnear)
                        rix_distance = euclidean_distance_point(robot_pose_xnear, goal_point)
                        print("rix_distance:",rix_distance)
                        frame_id = "world"
                        raduis_lix=40.0
                        listener = tf.TransformListener()

                        rospy.sleep(0.5) 
                        
                        obstacles_info = get_obstacles_info()
                        #obstacles_info = subscriber.get_latest_data()
                        while (obstacles_info==0):
                            
                            print("inside if")
                            
                    
                            print("----------------no obstacle block----------")
                            
                            print("iteration i number",i)
                            print("iteration j number",j)
                            #rand=input("while (obstacles_info==0): press enter to contiue ")
                            #note
                            random_x,random_y,random_z=gse_sample.generate_point_towards_goal(pos_x,pos_y,pos_z, x2, y2, pos_z)
                        
                            random_point = Point()
                            random_point.x = (random_x)
                            random_point.y = (random_y)
                            random_point.z = (random_z)
                            marker_publisher.publish_single_point(random_point)
                            
                            
                            
                            
                            #rix_distance = euclidean_distance_point(robot_pose_xnear, goal_point)
                            print("rix_distance:",rix_distance)
                            frame_id = "world"
                            raduis_lix=40.0
                           # phi_a, d_phi, theta_a, d_theta=calculate_spherical_sector_angles(robot_pose_xnear, goal_point, raduis_lix)
                           # print("phi_a, d_phi, theta_a, d_theta:",phi_a, d_phi, theta_a, d_theta)
                            #random_point=generate_random_point_in_spherical_sector(robot_pose_xnear,rix_distance, phi_a, d_phi, theta_a, d_theta)
                            #xrand=(random_point.x,random_point.y,random_point.z)
                            
                            
                            
                                
                            obs1_values = (random_x,random_y,random_z) 
                            waypoints[node_name] = obs1_values
                            print("waypoints:",waypoints)
                            
                                            
                            reached=mazebot_gtg.goal_movement(random_x,random_y)
                            if (reached==0):
                                
                                    
                                                
                                print("node_name:",node_name)
                                print("coordinate:",random_point)
                                print("reached!!!!!")
                            obstacles_info = get_obstacles_info()  
                            #obstacles_info = subscriber.get_latest_data() 
                        
                        random_point=generate_random_point_in_sphere(start_rand,goal_rand)
                        
                        xrand=(random_point.x,random_point.y,random_point.z)
                        
                        random_point = Point()
                        random_point.x = (xrand[0]-0)
                        random_point.y = (xrand[1]-0)
                        random_point.z = (xrand[2]-0)
                        
                        marker_publisher.publish_single_point(random_point)
                        print("-------------------------------------------")
                        
                        
                        print("rand",xrand)
                        #rand=input("random point press enter to contiue ")
                        
                            
                        #time.sleep(10)
                        first_nearest_vertex, first_nearest_coords, second_nearest_vertex, second_nearest_coords=nearest(g,xrand)#
                        print("First nearest point:", first_nearest_vertex, "with coordinates:", first_nearest_coords)
                        print("Second nearest point:", second_nearest_vertex, "with coordinates:", second_nearest_coords)
                        #rand=input("first_nearest_vertex, first_nearest_coords, second_nearest_vertex, second_nearest_coords press enter to contiue ") 
                        xnearest=Point()
                        xnearest.x=first_nearest_coords[0]
                        xnearest.y=first_nearest_coords[1]
                        xnearest.z=first_nearest_coords[2]
                        listener = tf.TransformListener()

                        rospy.sleep(0.5)  # Give some time to receive the messages
                    
                        
                            
                        obstacles_info = get_obstacles_info()
                        #obstacles_info = subscriber.get_latest_data()    
                        xobs0,xobs=gse_sample.xobstacle_near_point(xnearest,obstacles_info,random_point,xnearest)# note
                        #xobs=gse_sample.xobstacle(xnearest,obstacles_info)
                        gi_value= gse_sample.shape(random_point,xnearest,xobs )
                        print("gi value:",gi_value)
                        print("-------------------------------------------")
                        #time.sleep(5)
                        if gi_value==0:
                            
                            
                            print("gi value:is ZERO")
                            #steer(first_nearest_coords,xrand,g, marker_pub,node_name)
                            steer(first_nearest_vertex,second_nearest_vertex,first_nearest_coords,second_nearest_coords,xrand,g, marker_pub,node_name,i)
                            
                            f=get_vertex_positions(g)
                            print("f",f)
                            #rand=input("f press enter to contiue ")
                            #time.sleep(10)
                            #nearest_intersected_shape(robot_pose_xnear,obstacles_info,f,random_point,xobs,gse_sample,g,marker_pub,node_name,xrand)
                            nearest_intersected_shape(xnearest,obstacles_info,f,random_point,xobs,gse_sample,g,marker_pub,node_name,xrand) 
                            
                            f=get_vertex_positions(g)
                            print("f",f)
                            #rand=input("f press enter to contiue ")
                            #time.sleep(10)
                            nearest_intersected_shape(robot_pose_xnear,obstacles_info,f,random_point,xobs,gse_sample,g,marker_pub,node_name,xrand)
                            #nearest_intersected_shape(xnearest,obstacles_info,f,random_point,xobs,gse_sample,g,marker_pub,node_name,xrand)#note
                            #rand=input("after press enter to contiue ")    
                        else:
                            
                            print("inside else")
                            continue
                        marker_publisher.rate.sleep()  # Sleep to maintain the loop rate
                        print("----------------------------------\nGraph:----------------------")
                        g.display_graph()#changed
                    #time.sleep(10)
                    
                    
                    print("\nGraph:")
                    g.display_graph()
                    shortest_path = dijkstra(g, start_vertex, goal_vertex)
                    vertex_coordinates = {}
                    if shortest_path:
                        
                        print("\nShortest path:", shortest_path)
                        print("\nShortest path coordinates:")
                        for vertex in shortest_path:
                            
                            x, y, z, _ = g.vertices[vertex]
                            print(f"Vertex: {vertex}, Coordinates: ({x}, {y}, {z})")
                            vertex_coordinates[vertex] = (x, y, z)
                            
                        print("Vertex coordinates dictionary111:", vertex_coordinates)
                    else:
                        print("No path found.")  
                        
                    #publish_graph(g, start_vertex, goal_vertex, shortest_path)#
                    g.display_graph()
                    print("Vertex coordinates dictionary:", vertex_coordinates)
                    distance = euclidean_distance(point_a, point_b)#changes
                    print("iteration i number",i)
                    gi_value_goal= gse_sample.shape(goal_point,robot_pose_xnear,xobs_to_goal )
                    if (gi_value_goal==0) and (distance<=5):
                        #rand=input("if (gi_value_goal==0) and (distance<4): press enter to contiue ")
                        print("direct path to goal")
                        reached=mazebot_gtg.goal_movement(x2,y2)
                        if (reached==0):
                            
                        
                
                            print("node_name:",node_name)
                            print("coordinate:",random_point)
                            print("reached goal!!!!!")
                        break
                    
                    
                    value_of_first_entry = list(vertex_coordinates.items())[1][1]
                    value_of_first_entry1 = list(vertex_coordinates.items())[2][1]

                    # Store it in a list
                    value_list = list(value_of_first_entry)
                    value_list1 = list(value_of_first_entry1)

                    print(value_list)
                    #rand=input("print(value_list) press enter to contiue ")
                    distance = euclidean_distance(point_a, point_b)#changed
                    
                    if (distance>=5):
                        
                    
                                        
                        random_x,random_y,random_z=gse_sample.generate_point_towards_goal(pos_x,pos_y,pos_z, value_list[0], value_list[1], pos_z)#
                        
                        point_rand_marker=Point(random_x,random_y,pos_z)
                        marker_publisher.publish_single_point(point_rand_marker)
                        obs1_values = (random_x,random_y,random_z) 
                        waypoints[node_name] = obs1_values
                        print("waypoints:",waypoints)
                        reached=mazebot_gtg.goal_movement(random_x,random_y)
                        if (reached==0):
                            
                                
                                            
                            print("node_name:",node_name)
                            print("coordinate:",random_point)
                            print("reached!!!!!")
                
                    distance = euclidean_distance(point_a, point_b)
                    
                    if (distance<=5):
                        
                    
                        obs1_values = (value_list[0], value_list[1],pos_z) 
                        waypoints[node_name] = obs1_values
                        print("waypoints:",waypoints)
                        reached=mazebot_gtg.goal_movement(value_list[0],value_list[1])
                        reached=mazebot_gtg.goal_movement(x2,y2)
                        if (reached==0):
                            
                            print("else")   
                                
                                            
                            print("node_name:",node_name)
                            print("coordinate:",random_point)
                            print("reached!!!!!")
                          
               
                
    except rospy.ROSInterruptException:
        
        pass

if __name__ == "__main__":
    
    
    #workspace()
    main()
