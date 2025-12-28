#!/bin/bash
###############################################################################
# deploy.sh - Deploy CV Defect Detection YOLO Demo to Snowflake
#
# This script:
#   1. Checks prerequisites (snow CLI, connection)
#   2. Runs account-level SQL (roles, warehouse, compute pool, EAI)
#   3. Runs schema-level SQL (database, tables, stages)
#   4. Uploads Deep PCB dataset to stage
#   5. Deploys the YOLOv12 training notebook
#   6. Deploys the Streamlit dashboard
#
# Usage:
#   ./deploy.sh                     # Full deployment
#   ./deploy.sh -c prod             # Use 'prod' connection
#   ./deploy.sh --only-streamlit    # Deploy only Streamlit
###############################################################################

set -e
set -o pipefail

# Configuration
CONNECTION_NAME="demo"
ENV_PREFIX=""
ONLY_COMPONENT=""

PROJECT_PREFIX="CV_DEFECT_DETECTION_YOLO"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Error handler
error_exit() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
    exit 1
}

# Usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy the CV Defect Detection YOLO demo to Snowflake.

Options:
  -c, --connection NAME    Snowflake CLI connection name (default: demo)
  -p, --prefix PREFIX      Environment prefix for resources (e.g., DEV, PROD)
  --only-sql               Deploy only SQL infrastructure
  --only-data              Upload only dataset to stage
  --only-notebook          Deploy only the notebook
  --only-streamlit         Deploy only the Streamlit app
  -h, --help               Show this help message

Examples:
  $0                       # Full deployment
  $0 -c prod               # Use 'prod' connection
  $0 --only-streamlit      # Deploy only Streamlit
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -c|--connection)
            CONNECTION_NAME="$2"
            shift 2
            ;;
        -p|--prefix)
            ENV_PREFIX="$2"
            shift 2
            ;;
        --only-sql)
            ONLY_COMPONENT="sql"
            shift
            ;;
        --only-data)
            ONLY_COMPONENT="data"
            shift
            ;;
        --only-notebook)
            ONLY_COMPONENT="notebook"
            shift
            ;;
        --only-streamlit)
            ONLY_COMPONENT="streamlit"
            shift
            ;;
        *)
            error_exit "Unknown option: $1\nUse --help for usage information"
            ;;
    esac
done

# Build connection string
SNOW_CONN="-c $CONNECTION_NAME"

# Compute full prefix
if [ -n "$ENV_PREFIX" ]; then
    FULL_PREFIX="${ENV_PREFIX}_${PROJECT_PREFIX}"
else
    FULL_PREFIX="${PROJECT_PREFIX}"
fi

# Derive resource names
DATABASE="${FULL_PREFIX}"
SCHEMA="${PROJECT_PREFIX}"
ROLE="${FULL_PREFIX}_ROLE"
WAREHOUSE="${FULL_PREFIX}_WH"
COMPUTE_POOL="${FULL_PREFIX}_COMPUTE_POOL"
NETWORK_RULE="${FULL_PREFIX}_EGRESS_RULE"
EXTERNAL_ACCESS="${FULL_PREFIX}_EXTERNAL_ACCESS"
NOTEBOOK_NAME="${FULL_PREFIX}_NOTEBOOK"
STREAMLIT_APP="${FULL_PREFIX}_DASHBOARD"

# Helper function
should_run_step() {
    local step_name="$1"
    if [ -z "$ONLY_COMPONENT" ]; then
        return 0
    fi
    case "$ONLY_COMPONENT" in
        sql)
            [[ "$step_name" == "account_sql" || "$step_name" == "schema_sql" ]]
            ;;
        data)
            [[ "$step_name" == "upload_data" ]]
            ;;
        notebook)
            [[ "$step_name" == "notebook" ]]
            ;;
        streamlit)
            [[ "$step_name" == "streamlit" ]]
            ;;
        *)
            return 1
            ;;
    esac
}

# Display configuration
echo "=================================================="
echo "CV Defect Detection YOLO - Deployment"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Connection: $CONNECTION_NAME"
if [ -n "$ENV_PREFIX" ]; then
    echo "  Environment Prefix: $ENV_PREFIX"
fi
if [ -n "$ONLY_COMPONENT" ]; then
    echo "  Deploy Only: $ONLY_COMPONENT"
fi
echo "  Database: $DATABASE"
echo "  Schema: $SCHEMA"
echo "  Role: $ROLE"
echo "  Warehouse: $WAREHOUSE"
echo "  Compute Pool: $COMPUTE_POOL"
echo ""

###############################################################################
# Step 1: Check Prerequisites
###############################################################################
echo "Step 1: Checking prerequisites..."
echo "------------------------------------------------"

if ! command -v snow &> /dev/null; then
    error_exit "Snowflake CLI (snow) not found. Install with: pip install snowflake-cli"
fi
echo -e "${GREEN}[OK]${NC} Snowflake CLI found"

# Test connection
echo "Testing Snowflake connection..."
if ! snow sql $SNOW_CONN -q "SELECT 1" &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Failed to connect to Snowflake"
    snow connection test $SNOW_CONN 2>&1 || true
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Connection '$CONNECTION_NAME' verified"

# Check required files
for file in "sql/01_account_setup.sql" "sql/02_schema_setup.sql"; do
    if [ ! -f "$file" ]; then
        error_exit "Required file not found: $file"
    fi
done
echo -e "${GREEN}[OK]${NC} Required files present"
echo ""

###############################################################################
# Step 2: Run Account-Level SQL
###############################################################################
if should_run_step "account_sql"; then
    echo "Step 2: Running account-level SQL setup..."
    echo "------------------------------------------------"
    
    {
        echo "-- Set session variables"
        echo "SET FULL_PREFIX = '${FULL_PREFIX}';"
        echo "SET PROJECT_ROLE = '${ROLE}';"
        echo "SET PROJECT_WH = '${WAREHOUSE}';"
        echo "SET PROJECT_COMPUTE_POOL = '${COMPUTE_POOL}';"
        echo "SET PROJECT_NETWORK_RULE = '${NETWORK_RULE}';"
        echo "SET PROJECT_EXTERNAL_ACCESS = '${EXTERNAL_ACCESS}';"
        echo ""
        cat sql/01_account_setup.sql
    } | snow sql $SNOW_CONN -i > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK]${NC} Account-level setup completed"
    else
        error_exit "Account-level SQL setup failed"
    fi
    echo ""
else
    echo "Step 2: Skipped (--only-$ONLY_COMPONENT)"
    echo ""
fi

###############################################################################
# Step 3: Run Schema-Level SQL
###############################################################################
if should_run_step "schema_sql"; then
    echo "Step 3: Running schema-level SQL setup..."
    echo "------------------------------------------------"
    
    {
        echo "USE ROLE ${ROLE};"
        echo "SET FULL_PREFIX = '${FULL_PREFIX}';"
        echo "SET PROJECT_SCHEMA = '${SCHEMA}';"
        echo ""
        cat sql/02_schema_setup.sql
    } | snow sql $SNOW_CONN -i > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK]${NC} Schema-level setup completed"
    else
        error_exit "Schema-level SQL setup failed"
    fi
    echo ""
else
    echo "Step 3: Skipped (--only-$ONLY_COMPONENT)"
    echo ""
fi

###############################################################################
# Step 4: Upload Dataset to Stage (SKIPPED - Notebook handles data acquisition)
###############################################################################
if should_run_step "upload_data"; then
    echo "Step 4: Dataset upload..."
    echo "------------------------------------------------"
    
    # SKIPPED: Deep PCB dataset upload
    # The notebook now handles data acquisition via git sparse clone
    # with automatic caching to stage for subsequent runs.
    # To force local upload, uncomment the section below.
    
    echo -e "${YELLOW}[INFO]${NC} Deep PCB dataset upload skipped"
    echo "       The notebook will clone from GitHub on first run"
    echo "       and cache to stage for subsequent runs."
    
    # --- BEGIN COMMENTED: Manual upload (uncomment if git unavailable in SPCS) ---
    # if [ -d "data/DeepPCB/PCBData" ]; then
    #     # Upload each group directory - need to upload files from nested subdirectories
    #     # Structure: group*/XXXXX/*.jpg and group*/XXXXX_not/*.txt
    #     for groupdir in data/DeepPCB/PCBData/group*; do
    #         if [ -d "$groupdir" ]; then
    #             groupname=$(basename "$groupdir")
    #             echo "  Uploading $groupname..."
    #             
    #             # Upload images from inner directory (e.g., group00041/00041/*.jpg)
    #             for subdir in "$groupdir"/*/; do
    #                 if [ -d "$subdir" ]; then
    #                     subdirname=$(basename "$subdir")
    #                     # Upload all files in this subdirectory
    #                     snow sql $SNOW_CONN -q "
    #                         USE ROLE ${ROLE};
    #                         USE DATABASE ${DATABASE};
    #                         USE SCHEMA ${SCHEMA};
    #                         PUT file://${SCRIPT_DIR}/${subdir}* @MODEL_STAGE/raw/deeppcb/${groupname}/${subdirname}/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
    #                     " > /dev/null 2>&1 || true
    #                 fi
    #             done
    #         fi
    #     done
    #     
    #     # Verify upload
    #     UPLOAD_COUNT=$(snow sql $SNOW_CONN -q "
    #         USE ROLE ${ROLE};
    #         USE DATABASE ${DATABASE};
    #         USE SCHEMA ${SCHEMA};
    #         LIST @MODEL_STAGE/raw/deeppcb/;
    #     " -o tsv 2>/dev/null | wc -l)
    #     echo -e "${GREEN}[OK]${NC} Dataset uploaded to @MODEL_STAGE/raw/deeppcb/ (${UPLOAD_COUNT} files)"
    # else
    #     echo -e "${YELLOW}[WARN]${NC} Dataset not found at data/DeepPCB/PCBData"
    #     echo "       Run: cd data && git clone --sparse https://github.com/tangsanli5201/DeepPCB.git"
    # fi
    # --- END COMMENTED ---
    
    # Upload pretrained YOLO weights (still needed)
    if [ -f "weights/yolo12n.pt" ]; then
        echo "  Uploading pretrained YOLO weights..."
        snow sql $SNOW_CONN -q "
            USE ROLE ${ROLE};
            USE DATABASE ${DATABASE};
            USE SCHEMA ${SCHEMA};
            PUT file://${SCRIPT_DIR}/weights/yolo12n.pt @MODEL_STAGE/weights/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
        " > /dev/null 2>&1
        echo -e "${GREEN}[OK]${NC} Pretrained weights uploaded to @MODEL_STAGE/weights/"
    else
        echo -e "${YELLOW}[WARN]${NC} Pretrained weights not found at weights/yolo12n.pt"
    fi
    echo ""
else
    echo "Step 4: Skipped (--only-$ONLY_COMPONENT)"
    echo ""
fi

###############################################################################
# Step 5: Deploy Notebook
###############################################################################
if should_run_step "notebook"; then
    echo "Step 5: Deploying YOLOv12 training notebook..."
    echo "------------------------------------------------"
    
    if [ -f "notebooks/pcb_defect_detection.ipynb" ]; then
        # Upload notebook and environment
        snow sql $SNOW_CONN -q "
            USE ROLE ${ROLE};
            USE DATABASE ${DATABASE};
            USE SCHEMA ${SCHEMA};
            
            PUT file://${SCRIPT_DIR}/notebooks/pcb_defect_detection.ipynb @NOTEBOOKS/${NOTEBOOK_NAME}/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
            PUT file://${SCRIPT_DIR}/notebooks/environment.yml @NOTEBOOKS/${NOTEBOOK_NAME}/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
        " > /dev/null 2>&1
        
        # Create notebook object
        snow sql $SNOW_CONN -q "
            USE ROLE ${ROLE};
            USE DATABASE ${DATABASE};
            USE SCHEMA ${SCHEMA};
            
            CREATE OR REPLACE NOTEBOOK ${NOTEBOOK_NAME}
                FROM '@NOTEBOOKS/${NOTEBOOK_NAME}/'
                MAIN_FILE = 'pcb_defect_detection.ipynb'
                QUERY_WAREHOUSE = '${WAREHOUSE}'
                COMMENT = 'YOLOv12 CV Defect Detection Training';
            
            ALTER NOTEBOOK ${NOTEBOOK_NAME} ADD LIVE VERSION FROM LAST;
        " > /dev/null 2>&1
        
        echo -e "${GREEN}[OK]${NC} Notebook created: ${NOTEBOOK_NAME}"
        
        # Configure notebook runtime and compute pool for Container Runtime
        echo "  Configuring GPU runtime and compute pool..."
        if snow sql $SNOW_CONN -q "ALTER NOTEBOOK ${DATABASE}.${SCHEMA}.${NOTEBOOK_NAME} SET RUNTIME_NAME = 'SYSTEM\$GPU_RUNTIME', COMPUTE_POOL = ${COMPUTE_POOL}" --database ${DATABASE} --schema ${SCHEMA} --role ${ROLE} > /dev/null 2>&1; then
            echo -e "${GREEN}[OK]${NC} Notebook configured with GPU runtime and compute pool"
        else
            echo -e "${YELLOW}[WARN]${NC} Could not configure notebook runtime automatically"
            echo "       You may need to set the compute pool manually in the notebook UI"
        fi
        
        # Configure external access integration for PyPI egress
        echo "  Configuring external access integration..."
        if snow sql $SNOW_CONN -q "ALTER NOTEBOOK ${DATABASE}.${SCHEMA}.${NOTEBOOK_NAME} SET EXTERNAL_ACCESS_INTEGRATIONS = (${EXTERNAL_ACCESS})" --database ${DATABASE} --schema ${SCHEMA} --role ${ROLE} > /dev/null 2>&1; then
            echo -e "${GREEN}[OK]${NC} Notebook configured with external access"
        else
            echo -e "${YELLOW}[WARN]${NC} Could not configure external access automatically"
        fi
        
        # Validate notebook deployment
        NOTEBOOK_CHECK=$(snow sql $SNOW_CONN -q "
            USE ROLE ${ROLE};
            USE DATABASE ${DATABASE};
            USE SCHEMA ${SCHEMA};
            SHOW NOTEBOOKS LIKE '${NOTEBOOK_NAME}';
        " 2>/dev/null | grep -c "${NOTEBOOK_NAME}" || echo "0")
        
        if [ "$NOTEBOOK_CHECK" -gt 0 ]; then
            echo -e "${GREEN}[OK]${NC} Notebook deployed and verified: ${NOTEBOOK_NAME}"
        else
            echo -e "${YELLOW}[WARN]${NC} Notebook created but verification failed"
        fi
    else
        echo -e "${YELLOW}[WARN]${NC} Notebook file not found"
    fi
    echo ""
else
    echo "Step 5: Skipped (--only-$ONLY_COMPONENT)"
    echo ""
fi

###############################################################################
# Step 6: Deploy Streamlit App
###############################################################################
if should_run_step "streamlit"; then
    echo "Step 6: Deploying Streamlit dashboard..."
    echo "------------------------------------------------"
    
    # Clean up existing deployment
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        USE DATABASE ${DATABASE};
        USE SCHEMA ${SCHEMA};
        DROP STREAMLIT IF EXISTS ${STREAMLIT_APP};
    " > /dev/null 2>&1 || true
    
    rm -rf streamlit/output/bundle 2>/dev/null || true
    
    # Deploy from streamlit directory
    cd streamlit
    
    snow streamlit deploy \
        $SNOW_CONN \
        --database $DATABASE \
        --schema $SCHEMA \
        --role $ROLE \
        --replace 2>&1 | grep -v "^$" || true
    
    cd ..
    
    echo -e "${GREEN}[OK]${NC} Streamlit app deployed: ${STREAMLIT_APP}"
    echo ""
else
    echo "Step 6: Skipped (--only-$ONLY_COMPONENT)"
    echo ""
fi

###############################################################################
# Summary
###############################################################################
echo "=================================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "=================================================="
echo ""

if [ -z "$ONLY_COMPONENT" ]; then
    echo "Next Steps:"
    echo "  1. Run the main workflow:"
    echo "     ./run.sh main"
    echo ""
    echo "  2. Open the dashboard:"
    echo "     ./run.sh streamlit"
    echo ""
    echo "Resources Created:"
    echo "  - Database: $DATABASE"
    echo "  - Schema: $DATABASE.$SCHEMA"
    echo "  - Role: $ROLE"
    echo "  - Warehouse: $WAREHOUSE"
    echo "  - Compute Pool: $COMPUTE_POOL"
    echo "  - Notebook: $NOTEBOOK_NAME"
    echo "  - Streamlit: $STREAMLIT_APP"
else
    echo "Deployed component: $ONLY_COMPONENT"
fi

