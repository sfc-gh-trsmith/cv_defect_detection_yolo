--------------------------------------------------------------------------------
-- 01_account_setup.sql - Account-Level Setup for CV Defect Detection YOLO Demo
--
-- Run as: ACCOUNTADMIN
-- Purpose: Create role, warehouse, GPU compute pool, network rules, and EAI
--
-- Session variables (set by deploy.sh before execution):
--   FULL_PREFIX, PROJECT_ROLE, PROJECT_WH, PROJECT_COMPUTE_POOL,
--   PROJECT_NETWORK_RULE, PROJECT_EXTERNAL_ACCESS
--------------------------------------------------------------------------------

USE ROLE ACCOUNTADMIN;

--------------------------------------------------------------------------------
-- 1. Create Project Role
--------------------------------------------------------------------------------
CREATE ROLE IF NOT EXISTS IDENTIFIER($PROJECT_ROLE)
    COMMENT = 'Role for CV Defect Detection YOLO demo';

-- Grant role to current user for testing
GRANT ROLE IDENTIFIER($PROJECT_ROLE) TO ROLE ACCOUNTADMIN;

--------------------------------------------------------------------------------
-- 2. Create Warehouse
--------------------------------------------------------------------------------
CREATE WAREHOUSE IF NOT EXISTS IDENTIFIER($PROJECT_WH)
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for CV Defect Detection YOLO queries';

GRANT USAGE ON WAREHOUSE IDENTIFIER($PROJECT_WH) TO ROLE IDENTIFIER($PROJECT_ROLE);
GRANT OPERATE ON WAREHOUSE IDENTIFIER($PROJECT_WH) TO ROLE IDENTIFIER($PROJECT_ROLE);

--------------------------------------------------------------------------------
-- 3. Create GPU Compute Pool for Container Runtime
--------------------------------------------------------------------------------
CREATE COMPUTE POOL IF NOT EXISTS IDENTIFIER($PROJECT_COMPUTE_POOL)
    MIN_NODES = 1
    MAX_NODES = 1
    INSTANCE_FAMILY = GPU_NV_M
    AUTO_RESUME = TRUE
    AUTO_SUSPEND_SECS = 600
    COMMENT = 'GPU compute pool for YOLOv12 training';

GRANT USAGE ON COMPUTE POOL IDENTIFIER($PROJECT_COMPUTE_POOL) TO ROLE IDENTIFIER($PROJECT_ROLE);
GRANT MONITOR ON COMPUTE POOL IDENTIFIER($PROJECT_COMPUTE_POOL) TO ROLE IDENTIFIER($PROJECT_ROLE);

--------------------------------------------------------------------------------
-- 4. Create Network Rule for External Access (PyPI, GitHub, HuggingFace)
--------------------------------------------------------------------------------
-- Network rules require a database context, create temporary if needed
CREATE DATABASE IF NOT EXISTS IDENTIFIER($FULL_PREFIX);
GRANT OWNERSHIP ON DATABASE IDENTIFIER($FULL_PREFIX) TO ROLE IDENTIFIER($PROJECT_ROLE) COPY CURRENT GRANTS;

USE DATABASE IDENTIFIER($FULL_PREFIX);
CREATE SCHEMA IF NOT EXISTS PUBLIC;

CREATE OR REPLACE NETWORK RULE IDENTIFIER($PROJECT_NETWORK_RULE)
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = (
        'pypi.org',
        'files.pythonhosted.org',
        'github.com',
        'raw.githubusercontent.com',
        'objects.githubusercontent.com',
        'github-releases.githubusercontent.com',
        'github-cloud.s3.amazonaws.com',
        'github.githubassets.com',
        'huggingface.co',
        'cdn-lfs.huggingface.co',
        'cdn-lfs-us-1.huggingface.co'
    )
    COMMENT = 'Allow outbound access to PyPI, GitHub, and HuggingFace for ML packages';

--------------------------------------------------------------------------------
-- 5. Create External Access Integration
-- Note: ALLOWED_NETWORK_RULES requires fully qualified name literal, not IDENTIFIER()
--------------------------------------------------------------------------------
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION IDENTIFIER($PROJECT_EXTERNAL_ACCESS)
    ALLOWED_NETWORK_RULES = (CV_DEFECT_DETECTION_YOLO.PUBLIC.CV_DEFECT_DETECTION_YOLO_EGRESS_RULE)
    ENABLED = TRUE
    COMMENT = 'External access for pip install and model downloads';

GRANT USAGE ON INTEGRATION IDENTIFIER($PROJECT_EXTERNAL_ACCESS) TO ROLE IDENTIFIER($PROJECT_ROLE);

--------------------------------------------------------------------------------
-- 6. Grant Bind Service Endpoint (required for Streamlit in Notebook)
--------------------------------------------------------------------------------
GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE IDENTIFIER($PROJECT_ROLE);

--------------------------------------------------------------------------------
-- 7. Grant Create Database privilege (for schema setup)
--------------------------------------------------------------------------------
GRANT CREATE DATABASE ON ACCOUNT TO ROLE IDENTIFIER($PROJECT_ROLE);

--------------------------------------------------------------------------------
-- Summary
--------------------------------------------------------------------------------
-- Created:
--   - Role: $PROJECT_ROLE
--   - Warehouse: $PROJECT_WH
--   - Compute Pool: $PROJECT_COMPUTE_POOL (GPU_NV_M)
--   - Network Rule: $PROJECT_NETWORK_RULE
--   - External Access Integration: $PROJECT_EXTERNAL_ACCESS
--------------------------------------------------------------------------------

