#!/usr/bin/env bash
RUN_DIR="$( dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" )"
cd ${RUN_DIR}

grid-search -x train -e 15 -b 128 -w 8 000-mobilenet-v3-small/config.yaml 000-mobilenet-v3-small/grid-vars.yaml

