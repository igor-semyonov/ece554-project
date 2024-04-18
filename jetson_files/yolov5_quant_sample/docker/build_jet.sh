#!/bin/bash
docker build --network=host . --rm --pull --no-cache -t mypytorch_jet