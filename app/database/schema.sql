-- ==============================================================================
-- Normalized MySQL Schema for Pothole Detection & Road Condition Mapping System
-- ==============================================================================

CREATE DATABASE IF NOT EXISTS pothole_detection_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE pothole_detection_db;

-- ------------------------------------------------------------------------------
-- 1. Table: videos (Stores metadata of uploaded and processed video streams)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS videos (
    video_id VARCHAR(64) PRIMARY KEY,
    original_filename VARCHAR(255) NOT NULL,
    stored_filename VARCHAR(255) NOT NULL,
    status ENUM('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED') DEFAULT 'PENDING',
    duration_sec FLOAT DEFAULT 0.0,
    total_frames INT DEFAULT 0,
    fps FLOAT DEFAULT 0.0,
    resolution_width INT DEFAULT 0,
    resolution_height INT DEFAULT 0,
    has_gps BOOLEAN DEFAULT FALSE,
    gps_latitude DOUBLE DEFAULT NULL,
    gps_longitude DOUBLE DEFAULT NULL,
    total_detections INT DEFAULT 0,
    total_unique_potholes INT DEFAULT 0,
    processing_time_sec FLOAT DEFAULT 0.0,
    annotated_video_path VARCHAR(500) DEFAULT NULL,
    error_message TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_video_status (status),
    INDEX idx_video_created (created_at)
) ENGINE=InnoDB;

-- ------------------------------------------------------------------------------
-- 2. Table: detections (Raw frame-level YOLO detections)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS detections (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    video_id VARCHAR(64) NOT NULL,
    frame_number INT NOT NULL,
    timestamp_sec FLOAT NOT NULL,
    class_id INT DEFAULT 0,
    class_name VARCHAR(50) DEFAULT 'pothole',
    confidence FLOAT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    bbox_x1 FLOAT NOT NULL,
    bbox_y1 FLOAT NOT NULL,
    bbox_x2 FLOAT NOT NULL,
    bbox_y2 FLOAT NOT NULL,
    area_ratio FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_detections_video
        FOREIGN KEY (video_id) REFERENCES videos(video_id)
        ON DELETE CASCADE,
    INDEX idx_det_video_frame (video_id, frame_number),
    INDEX idx_det_severity (severity)
) ENGINE=InnoDB;

-- ------------------------------------------------------------------------------
-- 3. Table: road_conditions (Aggregated unique physical pothole events)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS road_conditions (
    event_id VARCHAR(64) PRIMARY KEY,
    video_id VARCHAR(64) NOT NULL,
    start_frame INT NOT NULL,
    end_frame INT NOT NULL,
    start_time_sec FLOAT NOT NULL,
    end_time_sec FLOAT NOT NULL,
    observation_count INT NOT NULL,
    max_confidence FLOAT NOT NULL,
    overall_severity VARCHAR(20) NOT NULL,
    bbox_x1 FLOAT NOT NULL,
    bbox_y1 FLOAT NOT NULL,
    bbox_x2 FLOAT NOT NULL,
    bbox_y2 FLOAT NOT NULL,
    max_area_ratio FLOAT NOT NULL,
    latitude DOUBLE DEFAULT NULL,
    longitude DOUBLE DEFAULT NULL,
    flag_status ENUM('FLAGGED', 'VERIFIED', 'RESOLVED') DEFAULT 'FLAGGED',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_conditions_video
        FOREIGN KEY (video_id) REFERENCES videos(video_id)
        ON DELETE CASCADE,
    INDEX idx_cond_video (video_id),
    INDEX idx_cond_severity (overall_severity),
    INDEX idx_cond_location (latitude, longitude),
    INDEX idx_cond_status (flag_status)
) ENGINE=InnoDB;
