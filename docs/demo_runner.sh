#!/bin/bash
# Demo runner for asciinema recording — clean output
set -e

echo '$ research-agent "What are the assumptions of double machine learning?"'
echo ""

# Run agent, suppress MCP server noise, show only the report
RESEARCH_KB_PATH=/home/brandon_behring/Claude/research-kb \
  research-agent "What are the assumptions of double machine learning?" 2>/dev/null
