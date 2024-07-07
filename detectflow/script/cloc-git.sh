#!/usr/bin/env bash
# First install cloc with: winget install cloc or sudo apt-get install cloc
# Then move the file somewhere to PATH and make it executable with: chmod +x cloc-git.sh (linux)
# Run with: bash cloc-git.sh https://github.com/ChlupacTheBosmer/DetectFlow
# Or run manually cloc <path-to-repo>

git clone --depth 1 "$1" temp-linecount-repo &&
  printf "('temp-linecount-repo' will be deleted automatically)\n\n\n" &&
  cloc temp-linecount-repo &&
  rm -rf temp-linecount-repo