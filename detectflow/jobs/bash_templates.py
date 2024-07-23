
# Basic template which is used to create a bash script for a job.
# Assumes that a keywarg 'config_path' is passed to the python script.
BASIC_TEMPLATE = """#!/bin/bash
#PBS -N {{ job_name }}
#PBS -q {{ queue }}
#PBS -l walltime={{ walltime }},select=1:ncpus={{ ncpus }}:mem={{ mem }}:ngpus={{ ngpus }}{{ gpu_cap }}:scratch_local={{ scratch_local }}

#PBS -o "{{ task_folder }}/{{ job_name }}.out"
#PBS -e "{{ task_folder }}/{{ job_name }}.err"

HOMEDIR="{{ home_dir }}"
JOBDIR="{{ task_folder }}"
CONFIG="{{ config_path }}"
SOURCE_FILE="{{ python_script_path }}"
SING_IMAGE="/storage/projects/yolo_group/singularity/DetectFlow:24.04_03.sif"

# Redirect TMP directory into scratch
export TMPDIR=$SCRATCHDIR

# Check if the CONFIG variable is set and not empty
if [ -z "$CONFIG" ]; then
    echo "Variable CONFIG is not set!" >&2
    exit 1
fi

echo "Config is set to: $CONFIG"

# Append a line to a file "jobs_info.txt" containing the ID of the job, the 
# hostname of node it is run on and the path to a scratch directory. This 
# information helps to find a scratch directory in case the job fails and you 
# need to remove the scratch directory manually.

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $HOMEDIR/jobs_info.txt

# Load modules here

# Test if scratch directory is set. If scratch directory is not set,
# issue an error message and exit.
test -n "$SCRATCHDIR" || {
    echo "Variable SCRATCHDIR is not set!" >&2
    exit 1
}

################################################################################
# CALCULATIONS

singularity run -B $SCRATCHDIR:/auto/brno2/home/$USER $SING_IMAGE python $SOURCE_FILE --config_path $CONFIG --scratch_path $SCRATCHDIR

################################################################################

# Copy everything from scratch directory to $JOBDIR
echo "Listing contents of scratch directory:"
ls -l $SCRATCHDIR

# Make sure that the output folder exists
mkdir -p "$JOBDIR"

if [ -d "$SCRATCHDIR" ] && [ "$(ls -A $SCRATCHDIR)" ]; then
   echo "Copying files from scratch directory to job output directory."
   cp -r "$SCRATCHDIR"/* "$JOBDIR" || echo "Failed to copy files."
else
   echo "Scratch directory is empty or does not exist."
fi

clean_scratch
"""