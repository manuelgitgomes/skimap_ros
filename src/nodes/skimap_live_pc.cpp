/*
 * Copyright (C) 2017 daniele de gregorio, University of Bologna - All Rights
 * Reserved
 * You may use, distribute and modify this code under the
 * terms of the GNU GPLv3 license.
 *
 * please write to: d.degregorio@unibo.it
 */

#include <boost/thread/thread.hpp>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ROS
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>

// OPENCV
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>

// Skimap
#include <skimap/SkiMap.hpp>
#include <skimap/voxels/VoxelDataRGBW.hpp>

// skimap
typedef skimap::VoxelDataRGBW<uint16_t, float> VoxelDataColor;
typedef skimap::SkiMap<VoxelDataColor, int16_t, float> SKIMAP;
typedef skimap::SkiMap<VoxelDataColor, int16_t, float>::Voxel3D Voxel3D;
typedef skimap::SkiMap<VoxelDataColor, int16_t, float>::Tiles2D Tiles2D;
SKIMAP *map;

// Ros
ros::NodeHandle *nh;
tf::TransformListener *tf_listener;
ros::Publisher cloud_publisher;
ros::Publisher map_publisher;
ros::Publisher map_2d_publisher;

// Live Params
std::string base_frame_name = "world";

/**
 */
struct MapParameters {
  float ground_level;
  float agent_height;
  float map_resolution;
  int min_voxel_weight;
  bool enable_chisel;
  bool height_color;
  int chisel_step;
} mapParameters;

/**
 */
struct CameraParameters {
  double fx, fy, cx, cy;
  int cols, rows;
  double min_distance;
  double max_distance;
  int point_cloud_downscale;
} camera;

/**
 */
struct ColorPoint {
  cv::Point3f point;
  cv::Vec4b color;
  int w;
};

/**
 */
struct SensorMeasurement {
  ros::Time stamp;
  std::vector<ColorPoint> points;
  std::vector<ColorPoint> chisel_points;

  void addChiselPoints(CameraParameters &camera, float resolution) {
    chisel_points.clear();

#pragma omp parallel
    {
      std::vector<ColorPoint> new_points;

#pragma omp for nowait
      for (int i = 0; i < points.size(); i++) {
        cv::Point3f dir = points[i].point - cv::Point3f(0, 0, 0);
        dir = dir * (1 / cv::norm(dir));
        for (float dz = camera.min_distance; dz < points[i].point.z;
             dz += resolution) {
          cv::Point3f dp = dir * dz;
          ColorPoint colorPoint;
          colorPoint.point = dp;
          colorPoint.w = -mapParameters.chisel_step;
          new_points.push_back(colorPoint);
        }
      }

#pragma omp critical
      points.insert(points.end(), new_points.begin(), new_points.end());
    }

    //        points.insert(points.end(), chisel_points.begin(),
    //        chisel_points.end());
  }
};

std::queue<SensorMeasurement> measurement_queue;
int measurement_queue_max_size = 2;

/**
 */
struct Timings {
  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::milliseconds ms;
  typedef std::chrono::microseconds us;
  typedef std::chrono::duration<float> fsec;

  std::map<std::string, std::chrono::time_point<std::chrono::system_clock>>
      times;

  void startTimer(std::string name) {
    times[name] = Time::now(); // IS NOT ROS TIME!
  }

  us elapsedMicroseconds(std::string name) {
    fsec elaps = Time::now() - times[name];
    return std::chrono::duration_cast<us>(elaps);
  }

  ms elapsedMilliseconds(std::string name) {
    fsec elaps = Time::now() - times[name];
    return std::chrono::duration_cast<ms>(elaps);
  }

  void printTime(std::string name) {
    ROS_INFO("Time for %s: %f ms", name.c_str(),
             float(elapsedMicroseconds(name).count()) / 1000.0f);
  }
} timings;

/**
 * Extracts point cloud from RGB-D Frame
 * @param pcl_cloud PCL PointCloud
 * @param output_points OUTPUT vector containing points
 */
void extractColorCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_cloud,
                       std::vector<ColorPoint> &output_points) {


  output_points.clear();

  // Removing NaN and Inf
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::PointCloud<pcl::PointXYZ>::PointType p_nan;
  p_nan.x = std::numeric_limits<float>::quiet_NaN();
  p_nan.y = std::numeric_limits<float>::quiet_NaN();
  p_nan.z = std::numeric_limits<float>::quiet_NaN();
  pcl_cloud->push_back(p_nan);

  pcl::PointCloud<pcl::PointXYZ>::PointType p_valid;
  p_valid.x = 1.0f;
  pcl_cloud->push_back(p_valid);

  pcl::Indices indices;
  pcl::removeNaNFromPointCloud(*pcl_cloud, *filtered_cloud, indices);

  // Adapting PCL PointCloud to SensorMeasurement
  for (int nIndex = 0; nIndex < filtered_cloud->size (); nIndex++)
  {
    ColorPoint cp;
    cp.point.x = filtered_cloud->points[nIndex].x;
    cp.point.y = filtered_cloud->points[nIndex].y;
    cp.point.z = filtered_cloud->points[nIndex].z;
    cp.w = 1;
    output_points.push_back(cp);
  }


}

/**
 * Visualization types for Markers
 */
enum VisualizationType {
  POINT_CLOUD,
  VOXEL_MAP,
  VOXEL_GRID,
};

/**
 * Creates a "blank" visualization marker with some attributes
 * @param frame_id Base TF Origin for the map points
 * @param time Timestamp for relative message
 * @param id Unique id for marker identification
 * @param type Type of Marker.
 * @return
 */
visualization_msgs::Marker createVisualizationMarker(std::string frame_id,
                                                     ros::Time time, int id,
                                                     VisualizationType type) {

  /**
   * Creating Visualization Marker
   */
  visualization_msgs::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = time;
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = id;

  if (type == VisualizationType::POINT_CLOUD) {
    marker.type = visualization_msgs::Marker::POINTS;
    marker.scale.x = 0.01;
    marker.scale.y = 0.01;
    marker.scale.z = 0.01;
  } else if (type == VisualizationType::VOXEL_MAP) {
    marker.type = visualization_msgs::Marker::CUBE_LIST;
    marker.scale.x = mapParameters.map_resolution;
    marker.scale.y = mapParameters.map_resolution;
    marker.scale.z = mapParameters.map_resolution;
  } else if (type == VisualizationType::VOXEL_GRID) {
    marker.type = visualization_msgs::Marker::CUBE_LIST;
    marker.scale.x = mapParameters.map_resolution;
    marker.scale.y = mapParameters.map_resolution;
    marker.scale.z = mapParameters.map_resolution;
  }
  return marker;
}

/**
 * Creates a Visualization Marker representing a Voxel Map of the environment
 * @param voxels_marker Marker to fill
 * @param voxels 3D Voxel list
 * @param min_weight_th Minimum weight for a voxel to be displayed
 */
void fillVisualizationMarkerWithVoxels(
    visualization_msgs::Marker &voxels_marker, std::vector<Voxel3D> &voxels,
    int min_weight_th) {

  cv::Mat colorSpace(1, voxels.size(), CV_32FC3);
  if (mapParameters.height_color) {
    for (int i = 0; i < voxels.size(); i++) {
      colorSpace.at<cv::Vec3f>(i)[0] = 180 - (voxels[i].z / 2) * 180;
      colorSpace.at<cv::Vec3f>(i)[1] = 1;
      colorSpace.at<cv::Vec3f>(i)[2] = 1;
    }
    cv::cvtColor(colorSpace, colorSpace, CV_HSV2BGR);
  }

  for (int i = 0; i < voxels.size(); i++) {

    if (voxels[i].data->w < min_weight_th)
      continue;
    /**
     * Create 3D Point from 3D Voxel
     */
    geometry_msgs::Point point;
    point.x = voxels[i].x;
    point.y = voxels[i].y;
    point.z = voxels[i].z;

    /**
     * Assign Cube Color from Voxel Color
     */
    std_msgs::ColorRGBA color;
    if (mapParameters.height_color) {
      color.r = colorSpace.at<cv::Vec3f>(i)[2];
      color.g = colorSpace.at<cv::Vec3f>(i)[1];
      color.b = colorSpace.at<cv::Vec3f>(i)[0];
    } else {
      color.r = float(voxels[i].data->r) / 255.0;
      color.g = float(voxels[i].data->g) / 255.0;
      color.b = float(voxels[i].data->b) / 255.0;
    }
    color.a = 1;

    voxels_marker.points.push_back(point);
    voxels_marker.colors.push_back(color);
  }
}

/**
 * Fills Visualization Marker with 2D Tiles coming from a 2D Query in SkiMap.
 * Represent in a black/white chessboard the occupied/free space respectively
 *
 * @param voxels_marker Marker to fill
 * @param tiles Tiles list
 */
void fillVisualizationMarkerWithTiles(visualization_msgs::Marker &voxels_marker,
                                      std::vector<Tiles2D> &tiles) {
  for (int i = 0; i < tiles.size(); i++) {

    /**
     * Create 3D Point from 3D Voxel
     */
    geometry_msgs::Point point;
    point.x = tiles[i].x;
    point.y = tiles[i].y;
    point.z = tiles[i].z;

    /**
     * Assign Cube Color from Voxel Color
     */
    std_msgs::ColorRGBA color;
    if (tiles[i].data != NULL) {
      color.r = color.g = color.b =
          tiles[i].data->w >= mapParameters.min_voxel_weight ? 0.0 : 1.0;
      color.a = 1;
    } else {
      color.r = color.g = color.b = 1.0;
      color.a = 1;
    }

    voxels_marker.points.push_back(point);
    voxels_marker.colors.push_back(color);
  }
}

/**
 * Fills a Visualization Marker with points coming from a SensorMeasuremetn
 * object. It's used
 * to show the Live Cloud
 *
 * @param voxels_marker
 * @param measurement
 * @param min_weight_th
 */
void fillVisualizationMarkerWithSensorMeasurement(
    visualization_msgs::Marker &voxels_marker, SensorMeasurement measurement) {
  for (int i = 0; i < measurement.points.size(); i++) {

    /**
     * Create 3D Point from 3D Voxel
     */
    geometry_msgs::Point point;
    point.x = measurement.points[i].point.x;
    point.y = measurement.points[i].point.y;
    point.z = measurement.points[i].point.z;

    /**
     * Assign Cube Color from Voxel Color
     */
    std_msgs::ColorRGBA color;
    color.r = measurement.points[i].color[2] / 255.0;
    color.g = measurement.points[i].color[1] / 255.0;
    color.b = measurement.points[i].color[0] / 255.0;
    color.a = 1;

    voxels_marker.points.push_back(point);
    voxels_marker.colors.push_back(color);
  }
}

/**
 */
struct IntegrationParameters {
  std::vector<VoxelDataColor> voxels_to_integrate;
  std::vector<tf::Vector3> poses_to_integrate;
  std::vector<bool> tiles_mask;
  int integration_counter;

  IntegrationParameters() { integration_counter = 0; }
} integrationParameters;

/**
 * Integrates measurements in global Map. Integration is made with OpenMP if
 * possibile so
 * real integration function is surrounded by "startBatchIntegration" and
 * "commitBatchIntegration". By removing
 * these two lines the integration will be launched in single-thread mode
 * @param measurement
 * @param map
 * @param base_to_camera
 */
void integrateMeasurement(SensorMeasurement measurement, SKIMAP *&map,
                          tf::Transform base_to_sensor) {

  std::vector<ColorPoint> points = measurement.points;

  map->enableConcurrencyAccess(true);
  // #pragma omp parallel shared(points, map)
  for (int i = 0; i < points.size(); i++) {
    // tf::Vector3 p = measurement.points[i].point;
    float x = points[i].point.x;
    float y = points[i].point.y;
    float z = points[i].point.z;

    tf::Vector3 base_to_point(x, y, z);
    base_to_point = base_to_sensor * base_to_point;

    VoxelDataColor voxel(points[i].color[2], points[i].color[1],
                         points[i].color[0], 1.0);

    map->integrateVoxel(float(base_to_point.x()), float(base_to_point.y()),
                        float(base_to_point.z()), &voxel);
  }

  integrationParameters.integration_counter++;
}

// PointCloud2 callback
void pcCallback(const sensor_msgs::PointCloud2ConstPtr &pc2_msg) {
  
  // Retrieve TF from base to sensor
  tf::StampedTransform base_to_sensor;
  try {
    tf_listener->lookupTransform(base_frame_name, pc2_msg->header.frame_id,
                                 pc2_msg->header.stamp, base_to_sensor);
  } catch (tf::TransformException ex) {
    ROS_ERROR("%s", ex.what());
    return;
  }


  // Convert PointCloud2 to PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*pc2_msg, *pcl_cloud);

  // Change PCL to Sensor Measurement
  SensorMeasurement measurement;
  measurement.stamp = pc2_msg->header.stamp;
  extractColorCloud(pcl_cloud, measurement.points);

  /**
   * Map Integration
   */
  timings.startTimer("Integration");
  integrateMeasurement(measurement, map, base_to_sensor);
  timings.printTime("Integration");

  /**
   * 3D Map Publisher
   */
  std::vector<Voxel3D> voxels;
  map->fetchVoxels(voxels);
  visualization_msgs::Marker map_marker = createVisualizationMarker(
      base_frame_name, pc2_msg->header.stamp, 1, VisualizationType::VOXEL_MAP);
  fillVisualizationMarkerWithVoxels(map_marker, voxels,
                                    mapParameters.min_voxel_weight);
  map_publisher.publish(map_marker);

  /**
   * 2D Grid Publisher
   */
  std::vector<Tiles2D> tiles;
  map->fetchTiles(tiles, mapParameters.agent_height);
  visualization_msgs::Marker map_2d_marker = createVisualizationMarker(
      base_frame_name, pc2_msg->header.stamp, 1, VisualizationType::VOXEL_GRID);
  fillVisualizationMarkerWithTiles(map_2d_marker, tiles);
  map_2d_publisher.publish(map_2d_marker);

  /**
   * Cloud publisher
   */
  visualization_msgs::Marker cloud_marker =
      createVisualizationMarker(pc2_msg->header.frame_id, pc2_msg->header.stamp, 1,
                                VisualizationType::POINT_CLOUD);
  fillVisualizationMarkerWithSensorMeasurement(cloud_marker, measurement);
  cloud_publisher.publish(cloud_marker);


}

/**
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {

  // Initialize ROS
  ros::init(argc, argv, "skimap_live_pc");
  nh = new ros::NodeHandle("~");
  tf_listener = new tf::TransformListener();

  // Cloud Publisher
  std::string map_cloud_publisher_topic =
      nh->param<std::string>("map_cloud_publisher_topic", "live_cloud");
  std::string map_topic =
      nh->param<std::string>("map_publisher_topic", "live_map");
  std::string map_2d_topic =
      nh->param<std::string>("map_2d_publisher_topic", "live_map_2d");
  cloud_publisher =
      nh->advertise<visualization_msgs::Marker>(map_cloud_publisher_topic, 1);
  map_publisher = nh->advertise<visualization_msgs::Marker>(map_topic, 1);
  map_2d_publisher = nh->advertise<visualization_msgs::Marker>(map_2d_topic, 1);

  int hz;
  nh->param<int>("hz", hz, 30);

  bool viz;
  nh->param<bool>("viz", viz, true);

  // SkiMap
  nh->param<float>("map_resolution", mapParameters.map_resolution, 0.05f);
  nh->param<float>("ground_level", mapParameters.ground_level, 0.15f);
  nh->param<int>("min_voxel_weight", mapParameters.min_voxel_weight, 10);
  nh->param<bool>("enable_chisel", mapParameters.enable_chisel, false);
  nh->param<bool>("height_color", mapParameters.height_color, false);
  nh->param<int>("chisel_step", mapParameters.chisel_step, 10);
  nh->param<float>("agent_height", mapParameters.agent_height, 0.0f);
  map = new SKIMAP(mapParameters.map_resolution, mapParameters.ground_level);

  // Topics
  std::string pc2_topic;
  nh->param<std::string>("pc2_topic", pc2_topic,
                         "/all_points");
  nh->param<std::string>("base_frame_name", base_frame_name, "world");
  nh->param<int>("point_cloud_downscale", camera.point_cloud_downscale, 2);

  // PC2 subscriber
  ros::Subscriber sub = nh->subscribe("/all_points", 1, pcCallback);

  // Spin & Time
  ros::Rate r(hz);

  // Spin
  while (nh->ok()) {
    ros::spinOnce();
    r.sleep();
  }
}
