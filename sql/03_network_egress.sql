--------------------------------------------------------------------------------
-- 03_network_egress.sql - External Network Access for GitHub
--
-- Run as: ACCOUNTADMIN (network rules require elevated privileges)
-- Purpose: Enable notebook to download Deep PCB dataset from GitHub
--
-- Reference: https://docs.snowflake.com/en/sql-reference/sql/create-network-rule
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- 1. Create Network Rule for GitHub Egress
--------------------------------------------------------------------------------
-- Allow outbound HTTPS traffic to GitHub domains for cloning repositories
-- and downloading raw files

CREATE OR REPLACE NETWORK RULE github_egress_rule
  TYPE = HOST_PORT
  MODE = EGRESS
  VALUE_LIST = (
    'github.com:443',
    'raw.githubusercontent.com:443',
    'objects.githubusercontent.com:443',
    'codeload.github.com:443'
  )
  COMMENT = 'Allow egress to GitHub for Deep PCB dataset download';

--------------------------------------------------------------------------------
-- 2. Create External Access Integration
--------------------------------------------------------------------------------
-- Bundle the network rule into an integration that can be attached to notebooks

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION github_access_integration
  ALLOWED_NETWORK_RULES = (github_egress_rule)
  ENABLED = TRUE
  COMMENT = 'External access integration for GitHub repository access';

--------------------------------------------------------------------------------
-- 3. Grant Usage to Project Role
--------------------------------------------------------------------------------
-- Allow the project role to use this integration

GRANT USAGE ON INTEGRATION github_access_integration TO ROLE IDENTIFIER($PROJECT_ROLE);

--------------------------------------------------------------------------------
-- 4. Attach Integration to Notebook (if notebook exists)
--------------------------------------------------------------------------------
-- Note: This must be run after the notebook is created
-- Uncomment and adjust notebook name as needed

-- ALTER NOTEBOOK IDENTIFIER($NOTEBOOK_NAME)
--   SET EXTERNAL_ACCESS_INTEGRATIONS = (github_access_integration);

--------------------------------------------------------------------------------
-- Summary
--------------------------------------------------------------------------------
-- Created:
--   - Network Rule: github_egress_rule (GitHub domains on port 443)
--   - Integration: github_access_integration
--   - Grant: Usage to $PROJECT_ROLE
--
-- Next Steps:
--   1. Run this script as ACCOUNTADMIN
--   2. Attach integration to notebook using ALTER NOTEBOOK command
--------------------------------------------------------------------------------
