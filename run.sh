#!/bin/bash
###############################################################################
# run.sh - Runtime Operations for CV Defect Detection YOLO Demo
#
# Commands:
#   main       - Execute the YOLOv12 training notebook
#   test       - Run query test suite (validates all Streamlit queries)
#   status     - Check status of all resources
#   streamlit  - Get Streamlit app URL
#   suspend    - Suspend the compute pool (stops billing)
#   resume     - Resume the compute pool
#
# Usage:
#   ./run.sh main              # Execute notebook
#   ./run.sh test              # Run query tests
#   ./run.sh status            # Check resources
#   ./run.sh streamlit         # Get dashboard URL
#   ./run.sh suspend           # Suspend compute pool
#   ./run.sh resume            # Resume compute pool
###############################################################################

set -e
set -o pipefail

# Configuration
CONNECTION_NAME="demo"
COMMAND=""
ENV_PREFIX=""

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
Usage: $0 [OPTIONS] COMMAND

Run operations for the CV Defect Detection YOLO demo.

Commands:
  main       Execute the YOLOv12 training notebook
  test       Run query test suite (validates all Streamlit queries)
  status     Check status of all resources
  streamlit  Get Streamlit app URL
  suspend    Suspend the compute pool (stops billing)
  resume     Resume the compute pool

Options:
  -c, --connection NAME    Snowflake CLI connection name (default: demo)
  -p, --prefix PREFIX      Environment prefix for resources
  -h, --help               Show this help message

Examples:
  $0 main                  # Execute notebook
  $0 test                  # Run query tests
  $0 status                # Check resources
  $0 -c prod streamlit     # Get URL using 'prod' connection
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
        main|test|status|streamlit|suspend|resume)
            COMMAND="$1"
            shift
            ;;
        *)
            error_exit "Unknown option: $1\nUse --help for usage information"
            ;;
    esac
done

# Require a command
if [ -z "$COMMAND" ]; then
    usage
fi

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
NOTEBOOK_NAME="${FULL_PREFIX}_NOTEBOOK"
STREAMLIT_APP="${FULL_PREFIX}_DASHBOARD"

###############################################################################
# Command: main - Execute Notebook
###############################################################################
cmd_main() {
    echo "=================================================="
    echo "Executing YOLOv12 Training Notebook"
    echo "=================================================="
    echo ""
    echo "Configuration:"
    echo "  Notebook: ${NOTEBOOK_NAME}"
    echo "  Compute Pool: ${COMPUTE_POOL}"
    echo ""
    
    # Stop any existing services on the compute pool
    echo "Stopping existing compute pool services..."
    snow sql $SNOW_CONN -q "
        USE ROLE ACCOUNTADMIN;
        ALTER COMPUTE POOL ${COMPUTE_POOL} STOP ALL;
    " 2>/dev/null || true
    
    echo "Waiting for compute pool to be ready..."
    sleep 5
    
    # Execute notebook
    echo ""
    echo "Executing notebook (this may take 10-30 minutes)..."
    echo "------------------------------------------------"
    
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        USE DATABASE ${DATABASE};
        USE SCHEMA ${SCHEMA};
        
        EXECUTE NOTEBOOK ${NOTEBOOK_NAME}();
    " 2>&1
    
    RESULT=$?
    
    echo ""
    if [ $RESULT -eq 0 ]; then
        echo -e "${GREEN}[OK]${NC} Notebook execution completed successfully"
        
        # Verify outputs
        echo ""
        echo "Verifying outputs..."
        DEFECT_COUNT=$(snow sql $SNOW_CONN -q "
            USE ROLE ${ROLE};
            USE DATABASE ${DATABASE};
            USE SCHEMA ${SCHEMA};
            SELECT COUNT(*) FROM DEFECT_LOGS;
        " -o tsv 2>/dev/null | tail -1)
        
        echo "  DEFECT_LOGS rows: ${DEFECT_COUNT:-0}"
    else
        echo -e "${RED}[FAIL]${NC} Notebook execution failed"
        exit 1
    fi
}

###############################################################################
# Command: test - Run Query Test Suite
###############################################################################
cmd_test() {
    echo "=================================================="
    echo "Running Query Test Suite"
    echo "=================================================="
    echo ""
    
    # Test each registered query
    QUERIES=(
        "defect_summary:SELECT DETECTED_CLASS, COUNT(*) FROM DEFECT_LOGS GROUP BY DETECTED_CLASS"
        "daily_trends:SELECT DATE_TRUNC('DAY', INFERENCE_TIMESTAMP), COUNT(*) FROM DEFECT_LOGS GROUP BY 1 LIMIT 10"
        "factory_line:SELECT COALESCE(m.FACTORY_LINE_ID, 'X'), COUNT(*) FROM DEFECT_LOGS d LEFT JOIN PCB_METADATA m ON d.BOARD_ID=m.BOARD_ID GROUP BY 1"
        "total_defects:SELECT COUNT(*) FROM DEFECT_LOGS"
        "pcb_count:SELECT COUNT(*) FROM PCB_METADATA"
    )
    
    PASS=0
    FAIL=0
    FAILED_QUERIES=""
    
    for query_def in "${QUERIES[@]}"; do
        name="${query_def%%:*}"
        sql="${query_def#*:}"
        
        result=$(snow sql $SNOW_CONN -q "
            USE ROLE ${ROLE};
            USE DATABASE ${DATABASE};
            USE SCHEMA ${SCHEMA};
            ${sql};
        " 2>&1)
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}PASS${NC}: ${name}"
            ((PASS++))
        else
            echo -e "${RED}FAIL${NC}: ${name}"
            FAILED_QUERIES="${FAILED_QUERIES}\n  - ${name}"
            ((FAIL++))
        fi
    done
    
    echo ""
    echo "------------------------------------------------"
    
    if [ $FAIL -eq 0 ]; then
        echo -e "${GREEN}All ${PASS} queries passed${NC}"
        exit 0
    else
        echo -e "${RED}FAILED: ${FAIL}/${PASS} queries${NC}"
        echo -e "Failed queries:${FAILED_QUERIES}"
        exit 1
    fi
}

###############################################################################
# Command: status - Check Resources
###############################################################################
cmd_status() {
    echo "=================================================="
    echo "CV Defect Detection YOLO - Status"
    echo "=================================================="
    echo ""
    
    echo "Compute Pool:"
    snow sql $SNOW_CONN -q "
        SHOW COMPUTE POOLS LIKE '${COMPUTE_POOL}';
    " 2>/dev/null || echo "  Not found or no access"
    
    echo ""
    echo "Warehouse:"
    snow sql $SNOW_CONN -q "
        SHOW WAREHOUSES LIKE '${WAREHOUSE}';
    " 2>/dev/null || echo "  Not found or no access"
    
    echo ""
    echo "Table Row Counts:"
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        USE DATABASE ${DATABASE};
        USE SCHEMA ${SCHEMA};
        SELECT 'DEFECT_LOGS' as TABLE_NAME, COUNT(*) as ROWS FROM DEFECT_LOGS
        UNION ALL SELECT 'PCB_METADATA', COUNT(*) FROM PCB_METADATA;
    " 2>/dev/null || echo "  Error querying tables"
    
    echo ""
    echo "Notebook:"
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        SHOW NOTEBOOKS LIKE '${NOTEBOOK_NAME}' IN SCHEMA ${DATABASE}.${SCHEMA};
    " 2>/dev/null || echo "  Not found or no access"
    
    echo ""
    echo "Streamlit App:"
    snow sql $SNOW_CONN -q "
        USE ROLE ${ROLE};
        SHOW STREAMLITS LIKE '${STREAMLIT_APP}' IN SCHEMA ${DATABASE}.${SCHEMA};
    " 2>/dev/null || echo "  Not found or no access"
}

###############################################################################
# Command: streamlit - Get Streamlit URL
###############################################################################
cmd_streamlit() {
    echo "=================================================="
    echo "CV Defect Detection YOLO - Streamlit Dashboard"
    echo "=================================================="
    echo ""
    
    # Try to get URL
    URL=$(snow streamlit get-url ${STREAMLIT_APP} \
        $SNOW_CONN \
        --database $DATABASE \
        --schema $SCHEMA \
        --role $ROLE 2>/dev/null) || true
    
    if [ -n "$URL" ]; then
        echo "Streamlit Dashboard URL:"
        echo ""
        echo "  $URL"
    else
        echo "Could not retrieve URL automatically."
        echo ""
        echo "To open the dashboard:"
        echo "1. Go to Snowsight (https://app.snowflake.com)"
        echo "2. Navigate to: Projects > Streamlit"
        echo "3. Open: ${STREAMLIT_APP}"
    fi
}

###############################################################################
# Command: suspend - Suspend Compute Pool
###############################################################################
cmd_suspend() {
    echo "=================================================="
    echo "Suspending Compute Pool"
    echo "=================================================="
    echo ""
    echo "Compute Pool: ${COMPUTE_POOL}"
    echo ""
    
    snow sql $SNOW_CONN -q "
        USE ROLE ACCOUNTADMIN;
        ALTER COMPUTE POOL ${COMPUTE_POOL} SUSPEND;
    "
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}[OK]${NC} Compute pool suspended (billing stopped)"
    else
        echo ""
        echo -e "${RED}[FAIL]${NC} Failed to suspend compute pool"
        exit 1
    fi
}

###############################################################################
# Command: resume - Resume Compute Pool
###############################################################################
cmd_resume() {
    echo "=================================================="
    echo "Resuming Compute Pool"
    echo "=================================================="
    echo ""
    echo "Compute Pool: ${COMPUTE_POOL}"
    echo ""
    
    snow sql $SNOW_CONN -q "
        USE ROLE ACCOUNTADMIN;
        ALTER COMPUTE POOL ${COMPUTE_POOL} RESUME;
    "
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}[OK]${NC} Compute pool resumed (billing active)"
    else
        echo ""
        echo -e "${RED}[FAIL]${NC} Failed to resume compute pool"
        exit 1
    fi
}

###############################################################################
# Execute Command
###############################################################################
case $COMMAND in
    main)
        cmd_main
        ;;
    test)
        cmd_test
        ;;
    status)
        cmd_status
        ;;
    streamlit)
        cmd_streamlit
        ;;
    suspend)
        cmd_suspend
        ;;
    resume)
        cmd_resume
        ;;
    *)
        error_exit "Unknown command: $COMMAND"
        ;;
esac

