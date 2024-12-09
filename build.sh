#!/bin/sh

# Script to build & deploy your docker image
docker build -t eidos-service.di.unito.it/cassano/pxs:1.0 . -f Dockerfile
docker push eidos-service.di.unito.it/cassano/pxs:1.0
