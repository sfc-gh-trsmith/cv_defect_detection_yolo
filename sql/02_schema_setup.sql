--------------------------------------------------------------------------------
-- 02_schema_setup.sql - Schema-Level Setup for CV Defect Detection YOLO Demo
--
-- Run as: CV_DEFECT_DETECTION_YOLO_ROLE (or $PROJECT_ROLE)
-- Purpose: Create database, schema, stages, and tables
--
-- Session variables (set by deploy.sh before execution):
--   FULL_PREFIX (database name), PROJECT_SCHEMA
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- 1. Create Database and Schema
--------------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS IDENTIFIER($FULL_PREFIX)
    COMMENT = 'Database for CV Defect Detection YOLOv12 demo';

USE DATABASE IDENTIFIER($FULL_PREFIX);

CREATE SCHEMA IF NOT EXISTS IDENTIFIER($PROJECT_SCHEMA)
    COMMENT = 'Schema for CV defect detection assets';

USE SCHEMA IDENTIFIER($PROJECT_SCHEMA);

--------------------------------------------------------------------------------
-- 2. Create Stages
--------------------------------------------------------------------------------
-- Stage for raw dataset and trained models
CREATE STAGE IF NOT EXISTS MODEL_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for Deep PCB dataset and trained YOLOv12 models';

-- Stage for notebook files
CREATE STAGE IF NOT EXISTS NOTEBOOKS
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for Jupyter notebook files';

--------------------------------------------------------------------------------
-- 3. Create Tables
--------------------------------------------------------------------------------

-- PCB Metadata: Tracks individual PCB boards
CREATE TABLE IF NOT EXISTS PCB_METADATA (
    BOARD_ID VARCHAR(50) NOT NULL,
    MANUFACTURING_DATE TIMESTAMP_NTZ,
    FACTORY_LINE_ID VARCHAR(50),
    PRODUCT_TYPE VARCHAR(100),
    IMAGE_PATH VARCHAR(500),
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT PK_PCB_METADATA PRIMARY KEY (BOARD_ID)
)
COMMENT = 'Metadata for PCB boards processed through defect detection';

-- Defect Logs: Stores inference results from YOLOv12
CREATE TABLE IF NOT EXISTS DEFECT_LOGS (
    INFERENCE_ID VARCHAR(36) NOT NULL,
    BOARD_ID VARCHAR(50),
    INFERENCE_TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    DETECTED_CLASS VARCHAR(50) NOT NULL,
    CONFIDENCE_SCORE FLOAT,
    BBOX_X_CENTER FLOAT,
    BBOX_Y_CENTER FLOAT,
    BBOX_WIDTH FLOAT,
    BBOX_HEIGHT FLOAT,
    IMAGE_PATH VARCHAR(500),
    MODEL_VERSION VARCHAR(50),
    CONSTRAINT PK_DEFECT_LOGS PRIMARY KEY (INFERENCE_ID)
)
COMMENT = 'Inference results from YOLOv12 defect detection model';

-- Create index-like clustering for common query patterns
ALTER TABLE DEFECT_LOGS CLUSTER BY (DETECTED_CLASS, INFERENCE_TIMESTAMP);

--------------------------------------------------------------------------------
-- 4. Create Views for Analytics
--------------------------------------------------------------------------------

-- Defect Summary by Class
CREATE OR REPLACE VIEW DEFECT_SUMMARY AS
SELECT 
    DETECTED_CLASS,
    COUNT(*) AS DEFECT_COUNT,
    AVG(CONFIDENCE_SCORE) AS AVG_CONFIDENCE,
    MIN(INFERENCE_TIMESTAMP) AS FIRST_DETECTED,
    MAX(INFERENCE_TIMESTAMP) AS LAST_DETECTED
FROM DEFECT_LOGS
GROUP BY DETECTED_CLASS;

-- Daily Defect Trends
CREATE OR REPLACE VIEW DAILY_DEFECT_TRENDS AS
SELECT 
    DATE_TRUNC('DAY', INFERENCE_TIMESTAMP) AS DETECTION_DATE,
    DETECTED_CLASS,
    COUNT(*) AS DEFECT_COUNT,
    AVG(CONFIDENCE_SCORE) AS AVG_CONFIDENCE
FROM DEFECT_LOGS
GROUP BY DATE_TRUNC('DAY', INFERENCE_TIMESTAMP), DETECTED_CLASS
ORDER BY DETECTION_DATE DESC, DEFECT_COUNT DESC;

-- Factory Line Performance (for demo, generate synthetic data later)
CREATE OR REPLACE VIEW FACTORY_LINE_DEFECTS AS
SELECT 
    COALESCE(m.FACTORY_LINE_ID, 'UNKNOWN') AS FACTORY_LINE_ID,
    d.DETECTED_CLASS,
    COUNT(*) AS DEFECT_COUNT,
    AVG(d.CONFIDENCE_SCORE) AS AVG_CONFIDENCE
FROM DEFECT_LOGS d
LEFT JOIN PCB_METADATA m ON d.BOARD_ID = m.BOARD_ID
GROUP BY COALESCE(m.FACTORY_LINE_ID, 'UNKNOWN'), d.DETECTED_CLASS;

--------------------------------------------------------------------------------
-- 5. Insert Sample Data for Demo (synthetic factory lines)
--------------------------------------------------------------------------------

-- Insert sample PCB metadata for demo purposes
INSERT INTO PCB_METADATA (BOARD_ID, MANUFACTURING_DATE, FACTORY_LINE_ID, PRODUCT_TYPE)
SELECT 
    'PCB_' || SEQ4() AS BOARD_ID,
    DATEADD('HOUR', -UNIFORM(0, 720, RANDOM()), CURRENT_TIMESTAMP()) AS MANUFACTURING_DATE,
    CASE UNIFORM(1, 4, RANDOM())
        WHEN 1 THEN 'SHANGHAI_L1'
        WHEN 2 THEN 'SHANGHAI_L2'
        WHEN 3 THEN 'SHENZHEN_L1'
        ELSE 'AUSTIN_L1'
    END AS FACTORY_LINE_ID,
    CASE UNIFORM(1, 3, RANDOM())
        WHEN 1 THEN 'CONSUMER_ELECTRONICS'
        WHEN 2 THEN 'AUTOMOTIVE'
        ELSE 'INDUSTRIAL'
    END AS PRODUCT_TYPE
FROM TABLE(GENERATOR(ROWCOUNT => 100));

--------------------------------------------------------------------------------
-- Summary
--------------------------------------------------------------------------------
-- Created:
--   - Database: $FULL_PREFIX
--   - Schema: $PROJECT_SCHEMA
--   - Stages: MODEL_STAGE, NOTEBOOKS
--   - Tables: PCB_METADATA, DEFECT_LOGS
--   - Views: DEFECT_SUMMARY, DAILY_DEFECT_TRENDS, FACTORY_LINE_DEFECTS
--------------------------------------------------------------------------------

