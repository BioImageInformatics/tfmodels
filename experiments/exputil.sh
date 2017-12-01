#/bin/bash

rootdir=$1
echo "Checking for experiment: $rootdir"

if [ -d "$rootdir" ]; then
  echo "Found $rootdir exists; removing it"
  rm -r $rootdir
fi

echo "Making folders under $rootdir"

echo Making $rootdir/ && mkdir $rootdir/
echo Making $rootdir/logs && mkdir $rootdir/logs
echo Making $rootdir/snapshots && mkdir $rootdir/snapshots
echo Making $rootdir/debug && mkdir $rootdir/debug
echo Making $rootdir/inference && mkdir $rootdir/inference
